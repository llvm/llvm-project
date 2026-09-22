//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Watchpoint-based data race detector for GPU targets.
///
//===----------------------------------------------------------------------===//

#include <gpuintrin.h>

#include "csan_offload_packet.h"
#include "csan_watch.h"
#include "sanitizer_common/sanitizer_internal_defs.h"
#include "shared/rpc.h"

using namespace __sanitizer;

#define INTERFACE extern "C" SANITIZER_INTERFACE_ATTRIBUTE

extern "C" {
// Externally initialized by the sanitizer, keeps one table per active device.
[[gnu::visibility("protected")]] u64 *__csan_watchpoint_table = nullptr;
}

static constexpr u64 SAMPLE_DELAY_MIN_NS = 1000;
static constexpr u64 SAMPLE_DELAY_MAX_NS = 10000;
static constexpr u32 WP_CHANCE = 8;
static_assert((CSAN_WATCHPOINT_TABLE_ENTRIES &
               (CSAN_WATCHPOINT_TABLE_ENTRIES - 1)) == 0,
              "watchpoint table size must be a power of two");
static_assert(WP_CHANCE >= 2 && (WP_CHANCE & (WP_CHANCE - 1)) == 0,
              "WP_CHANCE must be a power of two");

// The GPU case does
static constexpr u32 GPU_MAX_ACCESS_SIZE = 16;
static constexpr u32 GPU_WATCHPOINT_ENTRIES = CSAN_WATCHPOINT_TABLE_ENTRIES;
static constexpr u32 GPU_CHECK_ADJACENT_SLOTS = 0;
static_assert(GPU_WATCHPOINT_ENTRIES * sizeof(u64) == 2 * 1024 * 1024,
              "GPU watchpoint table must be 2 MiB");
using GpuWatchpointTable =
    __csan::WatchpointTable<GPU_MAX_ACCESS_SIZE, GPU_CHECK_ADJACENT_SLOTS>;

static GpuWatchpointTable get_watchpoints() {
  return GpuWatchpointTable(__csan_watchpoint_table);
}

// LDS addresses share the global watchpoint table. The original address is
// combined with the block's linear ID to create a unique global address.
static constexpr u32 LDS_OFFSET_BITS = 20;
static constexpr u64 LDS_OFFSET_MASK = (1ull << LDS_OFFSET_BITS) - 1;
static constexpr u64 LDS_FLAG = 1ull << (GpuWatchpointTable::AddressBits - 1);
static constexpr u64 GLOBAL_ADDRESS_MASK = LDS_FLAG - 1;
static constexpr u64 LDS_MAX_BLOCKS =
    1ull << (GpuWatchpointTable::AddressBits - 1 - LDS_OFFSET_BITS);
static_assert((LDS_FLAG | ((LDS_MAX_BLOCKS - 1) << LDS_OFFSET_BITS) |
               LDS_OFFSET_MASK) == GpuWatchpointTable::AddressMask,
              "LDS key fields must fill the address bits");

[[gnu::visibility("protected"),
  gnu::weak]] rpc::Client client asm("__llvm_rpc_client");

static u64 __csan_num_data_races = 0;

INTERFACE u64 __csan_get_num_data_races() {
  return __atomic_load_n(&__csan_num_data_races, __ATOMIC_RELAXED);
}

// Shallow deduplication check to save the host thread work. Keyed on both the
// PC and the race kind so each distinct kind of race at a PC is reported once.
static bool should_report(void *pc, unsigned kind) {
  static u64 seen[64] = {};
  const u64 token = (reinterpret_cast<uptr>(pc) >> 4) ^
                    (static_cast<u64>(kind) * 0x9E3779B97F4A7C15ull);
  u64 idx = (token * 0x9E3779B97F4A7C15ull) >> 58;
  u64 last = __scoped_atomic_exchange_n(&seen[idx], token, __ATOMIC_RELAXED,
                                        __MEMORY_SCOPE_DEVICE);
  return last != token;
}

[[gnu::cold, gnu::noinline]] static void
report(unsigned kind, uptr addr, u32 size, int access_type, uptr pc,
       void *peer = nullptr, int peer_access = 0, u32 peer_size = 0,
       u8 peer_lane = 0) {
  pc = pc ? pc : GET_CALLER_PC();
  if (!should_report(reinterpret_cast<void *>(pc), kind))
    return;

  __csan_gpu_race rep = {};
  rep.pc = pc;
  rep.peer_pc = reinterpret_cast<uptr>(peer);
  rep.addr = addr;
  rep.size = size;
  rep.access_type = static_cast<unsigned>(access_type);
  rep.kind = kind;
  rep.block[0] = __gpu_block_id(__GPU_X_DIM);
  rep.block[1] = __gpu_block_id(__GPU_Y_DIM);
  rep.block[2] = __gpu_block_id(__GPU_Z_DIM);
  rep.thread[0] = __gpu_thread_id(__GPU_X_DIM);
  rep.thread[1] = __gpu_thread_id(__GPU_Y_DIM);
  rep.thread[2] = __gpu_thread_id(__GPU_Z_DIM);
  rep.lane = __gpu_lane_id();
  rep.peer_lane = peer_lane;
  rep.peer_access_type = static_cast<u8>(peer_access);
  rep.peer_size = static_cast<u8>(peer_size);

  rpc::Client::Port Port = client.open<SANITIZER_OFFLOAD_CSAN>();
  Port.send([&](rpc::Buffer *buf, u32) {
    __builtin_memcpy(buf->data, &rep, sizeof(rep));
  });
  static_assert(sizeof(__csan_gpu_race) <= sizeof(rpc::Buffer),
                "Report must fit in a single packet");

  __scoped_atomic_fetch_add(&__csan_num_data_races, 1, __ATOMIC_RELAXED,
                            __MEMORY_SCOPE_DEVICE);
}

#if defined(__AMDGPU__)
// AMDGPU does not have a single set frequency. Different architectures and
// cards can have different values. A frequency of 100MHz is most common so we
// use it, if it is wrong it just means we sleep longer than expected.
static constexpr u64 CLOCK_FREQ_HZ = 100000000UL;
#else
static constexpr u64 CLOCK_FREQ_HZ = 1000000000UL;
#endif
static constexpr u64 TICKS_PER_SEC = 1000000000UL;

// FIXME: Avoids emitting an unresolved reference to the OCLC ABI version.
static u32 num_blocks(int dim) {
#ifdef __AMDGPU__
  return ((const u32 __gpu_constant *)__builtin_amdgcn_implicitarg_ptr())[dim];
#else
  return __gpu_num_blocks(dim);
#endif
}

static u64 lds_block_index() {
  return (u64)__gpu_block_id(__GPU_X_DIM) +
         (u64)num_blocks(__GPU_X_DIM) *
             ((u64)__gpu_block_id(__GPU_Y_DIM) +
              (u64)num_blocks(__GPU_Y_DIM) * (u64)__gpu_block_id(__GPU_Z_DIM));
}

static bool lds_fits() {
  u64 nxy, nxyz;
  if (__builtin_mul_overflow((u64)num_blocks(__GPU_X_DIM),
                             (u64)num_blocks(__GPU_Y_DIM), &nxy) ||
      __builtin_mul_overflow(nxy, (u64)num_blocks(__GPU_Z_DIM), &nxyz))
    return false;
  return nxyz <= LDS_MAX_BLOCKS;
}

// Stateless PRNG hashes cycle counter and global thread ID using SplitMix64.
static u64 rng() {
  u64 z = __builtin_readcyclecounter();
  z ^= (u64(__gpu_block_id(__GPU_X_DIM)) << 32 | __gpu_thread_id(__GPU_X_DIM)) *
       0xD1B54A32D192ED03ull;
  z += 0x9E3779B97F4A7C15ull;
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
  return z ^ (z >> 31);
}

namespace {
template <typename> struct is_ptr_local {
  static constexpr bool value = false;
};
template <typename T> struct is_ptr_local<T __gpu_local *> {
  static constexpr bool value = true;
};
} // namespace

template <typename PtrTy> static uptr report_address(PtrTy addr) {
  if constexpr (is_ptr_local<PtrTy>::value)
    return reinterpret_cast<uptr>((const volatile void *)addr);
  return reinterpret_cast<uptr>(addr);
}

template <typename PtrTy> static bool uses_table() {
  if constexpr (is_ptr_local<PtrTy>::value)
    return lds_fits();
  return true;
}

template <typename PtrTy> static u64 watch_key(uptr addr) {
  if constexpr (is_ptr_local<PtrTy>::value)
    return LDS_FLAG | (lds_block_index() << LDS_OFFSET_BITS) |
           (addr & LDS_OFFSET_MASK);
  return addr & GLOBAL_ADDRESS_MASK;
}

static u32 watchpoint_slot(u64 key) {
  key ^= key >> LDS_OFFSET_BITS;
  return (key / GPU_MAX_ACCESS_SIZE) & (GPU_WATCHPOINT_ENTRIES - 1);
}

static bool should_watch(u64 lane_mask, u32 access_type) {
  // If every access is atomic we cannot have a race.
  if (!__gpu_ballot(lane_mask, !(access_type & CSAN_ACCESS_ATOMIC)))
    return false;

  constexpr unsigned N = __builtin_ctzg(WP_CHANCE);
  return __gpu_read_first_lane_u32(lane_mask, (rng() >> (64 - N)) == 0);
}

template <typename PtrTy>
static u64 *find_watchpoint(uptr addr, u32 size, bool expect_write,
                            u64 &encoded) {
  const u64 key = watch_key<PtrTy>(addr);
  return get_watchpoints().Find(key, size, expect_write, watchpoint_slot(key),
                                encoded);
}

// FNV-1a digest of a byte range so wide accesses can reuse the value comparison
// semantics.
template <typename BytePtr, typename WordPtr>
static u64 read_range(BytePtr bytes, WordPtr, u32 size) {
  u64 sum = 0xcbf29ce484222325ull;
  u32 i = 0;

  for (; i < size && ((reinterpret_cast<uptr>(bytes) + i) & 7u); ++i)
    sum = (sum ^ bytes[i]) * 0x100000001b3ull;
  for (; i + 8 <= size; i += 8)
    sum = (sum ^ *reinterpret_cast<WordPtr>(bytes + i)) * 0x100000001b3ull;
  for (; i < size; ++i)
    sum = (sum ^ bytes[i]) * 0x100000001b3ull;
  return sum;
}

// Snapshot the watched location for value-change detection. Larger sizes get
// converted into a single checksum.
static u64 read_instrumented_memory(const volatile __gpu_global void *ptr,
                                    u32 size) {
  const uptr addr =
      reinterpret_cast<uptr>(const_cast<const __gpu_global void *>(ptr));
  if ((addr & (size - 1)) == 0) {
    switch (size) {
    case 1:
      return *(const volatile __gpu_global u8 *)ptr;
    case 2:
      return *(const volatile __gpu_global u16 *)ptr;
    case 4:
      return *(const volatile __gpu_global u32 *)ptr;
    case 8:
      return *(const volatile __gpu_global u64 *)ptr;
    }
  }
  return read_range((const volatile __gpu_global u8 *)ptr,
                    (const volatile __gpu_global u64 *)ptr, size);
}

static u64 read_instrumented_memory(const volatile __gpu_local void *ptr,
                                    u32 size) {
  const uptr addr =
      reinterpret_cast<uptr>(const_cast<const __gpu_local void *>(ptr));
  if ((addr & (size - 1)) == 0) {
    switch (size) {
    case 1:
      return *(const volatile __gpu_local u8 *)ptr;
    case 2:
      return *(const volatile __gpu_local u16 *)ptr;
    case 4:
      return *(const volatile __gpu_local u32 *)ptr;
    case 8:
      return *(const volatile __gpu_local u64 *)ptr;
    }
  }
  return read_range((const volatile __gpu_local u8 *)ptr,
                    (const volatile __gpu_local u64 *)ptr, size);
}

static bool intra_wave_race(u64 lane_mask, uptr addr, int access_type,
                            u8 &peer_lane) {
  const bool is_write = (access_type & CSAN_ACCESS_WRITE) != 0;
  const bool is_atomic = (access_type & CSAN_ACCESS_ATOMIC) != 0;
  const u64 writers = __gpu_ballot(lane_mask, is_write);
  const u64 nonatomic = __gpu_ballot(lane_mask, !is_atomic);
  if (!writers || !nonatomic)
    return false;

  const u64 same_addr = __gpu_match_any_u64(lane_mask, addr);
  const bool is_race = __builtin_popcountg(same_addr) >= 2 &&
                       (same_addr & writers) && (same_addr & nonatomic);
  if (!is_race || !__gpu_is_first_in_lane(same_addr))
    return false;
  peer_lane = static_cast<u8>(63u - __builtin_clzg(same_addr));
  return true;
}

static void delay_ns(u64 nsecs) {
  const u64 tick_rate = TICKS_PER_SEC / CLOCK_FREQ_HZ;
  const u64 start = __builtin_readsteadycounter();
  const u64 end = start + (nsecs + tick_rate - 1) / tick_rate;
#if defined(__AMDGPU__)
  __builtin_amdgcn_s_sleep(2);
  while (__builtin_readsteadycounter() < end)
    __builtin_amdgcn_s_sleep(15);
#else
  while (__builtin_readsteadycounter() < end)
    __gpu_thread_suspend();
#endif
}

static void sample_delay(u64 lane_mask) {
  u64 nsecs = SAMPLE_DELAY_MIN_NS;
  if (__gpu_is_first_in_lane(lane_mask))
    nsecs += (rng() >> 32) % (SAMPLE_DELAY_MAX_NS - SAMPLE_DELAY_MIN_NS);
  delay_ns(__gpu_read_first_lane_u64(lane_mask, nsecs));
}

[[gnu::cold, gnu::noinline]] static void
found_watchpoint(u64 *wp, u64 encoded, uptr pc, bool is_write, u32 size) {
  pc = pc ? pc : GET_CALLER_PC();
  get_watchpoints().TryConsume(wp, encoded, pc, is_write, size);
}

// The slow path, sets a watchpoint in the table and waits to see if any other
// thread tripped it. Returns a CSAN_RACE_* kind, or -1 if none.
template <typename PtrTy>
static int watch(u64 lane_mask, const PtrTy addr, u32 size, int access_type,
                 uptr pc, uptr &report_pc, void *&peer, int &peer_access,
                 u32 &peer_size) {
  report_pc = pc ? pc : GET_CALLER_PC();
  const bool is_write = (access_type & CSAN_ACCESS_WRITE) != 0;
  const uptr iaddr = reinterpret_cast<uptr>(addr);

  const u32 wp_size = GpuWatchpointTable::EncodeSize(size);
  const bool armable = uses_table<PtrTy>() &&
                       !(access_type & CSAN_ACCESS_ATOMIC) &&
                       (is_ptr_local<PtrTy>::value || iaddr != 0);
  const u64 key = watch_key<PtrTy>(iaddr);
  u64 *wp = armable ? get_watchpoints().Insert(key, wp_size, is_write,
                                               watchpoint_slot(key))
                    : nullptr;

  const u64 old = read_instrumented_memory(addr, size);
  sample_delay(lane_mask);
  const u64 now = read_instrumented_memory(addr, size);

  peer = nullptr;
  peer_access = 0;
  peer_size = 0;
  int kind = -1;
  if (wp && !get_watchpoints().Consume(wp, peer, peer_access, peer_size))
    kind = CSAN_RACE_DATA;
  else if (old != now)
    kind = CSAN_RACE_UNKNOWN_ORIGIN;

  if (wp)
    get_watchpoints().Remove(wp);
  return kind;
}

template <typename PtrTy>
static void check_access_impl(u64 lane_mask, const PtrTy addr, u32 size,
                              int access_type, uptr pc) {
  pc = pc ? pc : GET_CALLER_PC();
  if (uses_table<PtrTy>()) {
    const bool is_write = (access_type & CSAN_ACCESS_WRITE) != 0;
    u64 encoded;
    u64 *wp = find_watchpoint<PtrTy>((u64)addr, size, !is_write, encoded);
    if (wp)
      found_watchpoint(wp, encoded, pc, is_write, size);
  }

  if (!should_watch(lane_mask, access_type))
    return;

  const uptr iaddr = reinterpret_cast<uptr>(addr);
  u8 peer_lane;
  if (intra_wave_race(lane_mask, iaddr, access_type, peer_lane))
    report(CSAN_RACE_INTRA_WAVE, report_address(addr), size, access_type, pc,
           nullptr, 0, 0, peer_lane);

  if (access_type & CSAN_ACCESS_ATOMIC)
    return;

  uptr report_pc;
  void *peer;
  int peer_access;
  u32 peer_size;
  int kind = watch(lane_mask, addr, size, access_type, pc, report_pc, peer,
                   peer_access, peer_size);
  if (kind >= 0)
    report(static_cast<unsigned>(kind), report_address(addr), size, access_type,
           report_pc, peer, peer_access, peer_size);
}

static void check_access(const volatile void *addr, uptr size, int access_type,
                         uptr pc) {
  pc = pc ? pc : GET_CALLER_PC();
  if (__gpu_is_ptr_private(const_cast<void *>(addr)) || !size)
    return;
  if (size > ~u32(0))
    size = ~u32(0);

  if (__gpu_is_ptr_local(const_cast<void *>(addr)))
    return check_access_impl(__gpu_lane_mask(),
                             (const volatile __gpu_local void *)addr, size,
                             access_type, pc);
  check_access_impl(__gpu_lane_mask(), (const volatile __gpu_global void *)addr,
                    size, access_type, pc);
}

//===----------------------------------------------------------------------===//
// Public ABI (emitted by the ConcurrencySanitizer pass)
//===----------------------------------------------------------------------===//

// Using `sanitize_concurrency_no_checking_at_run_time` ignored on the device,
INTERFACE void __csan_init() {}
INTERFACE void __csan_func_entry(void *) {}
INTERFACE void __csan_func_exit() {}
INTERFACE void __csan_ignore_thread_begin() {}
INTERFACE void __csan_ignore_thread_end() {}

static int access_flags(int flags, bool is_write) {
  return flags | (is_write ? CSAN_ACCESS_WRITE : 0);
}

#define CSAN_PROBE(name, N, is_write)                                          \
  INTERFACE void name(void *addr, int flags) {                                 \
    check_access(addr, N, access_flags(flags, is_write), GET_CALLER_PC());     \
  }

#define CSAN_ACCESS(N)                                                         \
  CSAN_PROBE(__csan_read##N, N, false)                                         \
  CSAN_PROBE(__csan_unaligned_read##N, N, false)                               \
  CSAN_PROBE(__csan_volatile_read##N, N, false)                                \
  CSAN_PROBE(__csan_unaligned_volatile_read##N, N, false)                      \
  CSAN_PROBE(__csan_write##N, N, true)                                         \
  CSAN_PROBE(__csan_unaligned_write##N, N, true)                               \
  CSAN_PROBE(__csan_volatile_write##N, N, true)                                \
  CSAN_PROBE(__csan_unaligned_volatile_write##N, N, true)                      \
  CSAN_PROBE(__csan_read_write##N, N, true)                                    \
  CSAN_PROBE(__csan_unaligned_read_write##N, N, true)

CSAN_ACCESS(1)
CSAN_ACCESS(2)
CSAN_ACCESS(4)
CSAN_ACCESS(8)
CSAN_ACCESS(16)

INTERFACE void __csan_read_range(void *addr, uptr size, int flags) {
  check_access(addr, size, access_flags(flags, false), GET_CALLER_PC());
}

INTERFACE void __csan_write_range(void *addr, uptr size, int flags) {
  check_access(addr, size, access_flags(flags, true), GET_CALLER_PC());
}

INTERFACE void __csan_atomic_thread_fence(int) {}
INTERFACE void __csan_atomic_signal_fence(int) {}
