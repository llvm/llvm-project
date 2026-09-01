//===-- csan_gpu.cpp - GPU ConcurrencySanitizer runtime -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Watchpoint-based data race detector for GPU targets, modelled on the Linux
// kernel's KCSAN.
//
//===----------------------------------------------------------------------===//

#include <gpuintrin.h>
#include <stdint.h>

#include "sanitizer/csan_interface.h"
#include "sanitizer_common/sanitizer_internal_defs.h"
#include "shared/rpc.h"

[[gnu::visibility("protected"),
  gnu::weak]] rpc::Client client asm("__llvm_rpc_client");

// Running total of races this device has reported.
extern "C" SANITIZER_INTERFACE_ATTRIBUTE uint64_t __tsan_num_data_races = 0;

// Shallow deduplication check to save the host thread work. Keyed on both the
// PC and the race kind so each distinct kind of race at a PC is reported once.
static bool should_report(void* pc, unsigned kind) {
  static uint64_t seen[64] = {};
  const uint64_t token = (reinterpret_cast<uintptr_t>(pc) >> 4) ^
                         (static_cast<uint64_t>(kind) * 0x9E3779B97F4A7C15ull);
  uint64_t idx = (token * 0x9E3779B97F4A7C15ull) >> 58;
  uint64_t last = __scoped_atomic_exchange_n(
      &seen[idx], token, __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE);
  return last != token;
}

// Report a data race to the RPC server so it can be symbolized and presented.
[[gnu::cold]] static void report(unsigned kind, uintptr_t addr, uint32_t size,
                                 int access_type, void* pc,
                                 void* peer = nullptr, uint8_t peer_lane = 0,
                                 const uint16_t (&peer_thread)[3] = {}) {
  if (!should_report(pc, kind))
    return;

  __tsan_gpu_race rep;
  rep.pc = reinterpret_cast<uintptr_t>(pc);
  rep.peer_pc = reinterpret_cast<uintptr_t>(peer);
  rep.addr = addr;
  rep.size = size;
  rep.access_type = static_cast<unsigned>(access_type);
  rep.block[0] = __gpu_block_id(__GPU_X_DIM);
  rep.block[1] = __gpu_block_id(__GPU_Y_DIM);
  rep.block[2] = __gpu_block_id(__GPU_Z_DIM);
  rep.thread[0] = __gpu_thread_id(__GPU_X_DIM);
  rep.thread[1] = __gpu_thread_id(__GPU_Y_DIM);
  rep.thread[2] = __gpu_thread_id(__GPU_Z_DIM);
  rep.lane = __gpu_lane_id();
  rep.peer_lane = peer_lane;
  rep.peer_thread[0] = peer_thread ? peer_thread[0] : 0;
  rep.peer_thread[1] = peer_thread ? peer_thread[1] : 0;
  rep.peer_thread[2] = peer_thread ? peer_thread[2] : 0;
  rep.kind = kind;

  rpc::Client::Port Port = client.open<TSAN_GPU_REPORT_OPCODE>();
  Port.send([&](rpc::Buffer* buf, uint32_t) {
    __builtin_memcpy(buf->data, &rep, sizeof(rep));
  });
  static_assert(sizeof(__tsan_gpu_race) <= sizeof(rpc::Buffer),
                "Report must fit in a single packet");

  __scoped_atomic_fetch_add(&__tsan_num_data_races, 1, __ATOMIC_RELAXED,
                            __MEMORY_SCOPE_DEVICE);
}

#if defined(__AMDGPU__)
// AMDGPU does not have a single set frequency. Different architectures and
// cards can have different values. A frequency of 100MHz is most common so we
// use it, if it is wrong it just means we sleep longer than expected.
static constexpr uint64_t CLOCK_FREQ_HZ = 100000000UL;
#else
static constexpr uint64_t CLOCK_FREQ_HZ = 1000000000UL;
#endif
static constexpr uint64_t TICKS_PER_SEC = 1000000000UL;

// Randomized bounds, in nanoseconds, for the watchpoint stall window. A wider
// window catches more concurrent races at the cost of extra runtime.
static constexpr uint64_t SAMPLE_DELAY_MIN_NS = 1000;
static constexpr uint64_t SAMPLE_DELAY_MAX_NS = 5000;

// Watchpoint table capacity, in bytes.
static constexpr uint64_t WP_TABLE_SIZE = /*2 MiB=*/2 * 1024ul * 1024ul;

// Number of slots in the watchpoint table, must be a power of two;
static constexpr uint64_t WP_TABLE_SLOTS = WP_TABLE_SIZE / sizeof(uint64_t);

// The probability that we set up a watchpoint at any given access.
static constexpr uint32_t WP_CHANCE = 8;

// Whether global accesses use the watchpoint table.
static constexpr bool WP_ENABLE_TABLE = true;

// The largest size we encode. We check a very narrow range to widen
// watchpoints. Wide accesses are relatively uncommon on global memory.
static constexpr uint32_t WP_MAX_SIZE = 8;

static constexpr uint32_t WP_SIZE_BITS =
    __builtin_popcountg(WP_MAX_SIZE - 1u) + 1;
static constexpr uint32_t WP_ADDR_BITS = 64 - 2 - WP_SIZE_BITS;

//===----------------------------------------------------------------------===//
// Watchpoint encoding:
//   [63]                       is_write
//   [62]                       consumed
//   [61 : WP_ADDR_BITS]        size
//   [WP_ADDR_BITS-1 : 0]       address
//===----------------------------------------------------------------------===//

static constexpr uint64_t WP_INVALID = 0;
static constexpr uint64_t WP_CONSUMED_MASK = 1ull << 62;
static constexpr uint64_t WP_WRITE_MASK = 1ull << 63;
static constexpr uint64_t WP_ADDR_MASK = (1ull << WP_ADDR_BITS) - 1;
static constexpr uint64_t WP_SIZE_MASK = ((1ull << WP_SIZE_BITS) - 1)
                                         << WP_ADDR_BITS;
static_assert((WP_MAX_SIZE & (WP_MAX_SIZE - 1)) == 0,
              "WP_MAX_SIZE must be a power of two");
static_assert((WP_WRITE_MASK ^ WP_CONSUMED_MASK ^ WP_SIZE_MASK ^
               WP_ADDR_MASK) == ~0ull,
              "watchpoint fields must partition the 64-bit word");

static constexpr uint64_t encode_watchpoint(uint64_t addr, uint32_t size,
                                            bool is_write) {
  return (is_write ? WP_WRITE_MASK : 0) | ((uint64_t)size << WP_ADDR_BITS) |
         (addr & WP_ADDR_MASK);
}

static constexpr bool decode_watchpoint(uint64_t wp, uint64_t& addr,
                                        uint32_t& size, bool& is_write) {
  if (wp == WP_INVALID || (wp & WP_CONSUMED_MASK))
    return false;
  is_write = (wp & WP_WRITE_MASK) != 0;
  size = (wp & WP_SIZE_MASK) >> WP_ADDR_BITS;
  addr = wp & WP_ADDR_MASK;
  return true;
}

static constexpr bool ranges_overlap(uint64_t a1, uint32_t s1, uint64_t a2,
                                     uint32_t s2) {
  return a1 < a2 + s2 && a2 < a1 + s1;
}

static_assert((WP_TABLE_SLOTS & (WP_TABLE_SLOTS - 1)) == 0,
              "WP_TABLE_SLOTS must be a power of two");

static constexpr uint32_t watchpoint_slot(uint64_t addr) {
  const uint64_t word = addr >> (WP_SIZE_BITS - 1);
  return word & (WP_TABLE_SLOTS - 1);
}

static inline uint32_t xorshift32(uint32_t& state) {
  state ^= state << 13;
  state ^= state >> 17;
  state ^= state << 5;
  return state * 0x9e3779bb;
}

// Wave-uniform Bernoulli trial that is true with probability 1 / N.
template <uint32_t N>
static bool bernoulli(uint32_t& rand) {
  static_assert((N & (N - 1)) == 0,
                "sample denominator must be a power of two");
  [[maybe_unused]] const uint32_t r = xorshift32(rand);
  if constexpr (N == 0)
    return false;
  else if constexpr (N == 1)
    return true;
  else
    return (r >> (32 - __builtin_ctzg(N))) == 0;
}

static uint64_t watchpoints[WP_TABLE_SLOTS] = {};

// Flatten the block-local thread index into a linear lane index.
static uint32_t flat_thread_id() {
  return __gpu_thread_id(__GPU_X_DIM) +
         __gpu_num_threads(__GPU_X_DIM) *
             (__gpu_thread_id(__GPU_Y_DIM) +
              __gpu_num_threads(__GPU_Y_DIM) * __gpu_thread_id(__GPU_Z_DIM));
}

// PRNG seed based off of the current thread's IDs and clock cycle.
static uint32_t entropy() {
  return (static_cast<uint32_t>(__builtin_readcyclecounter() | 1u) ^
          flat_thread_id() ^ (__gpu_block_id(__GPU_X_DIM) * 0x9e3779b9u));
}

namespace {
// Per-wavefront analysis state, seeded once at kernel entry and carried for the
// lifetime of the wave.
struct Ctx {
  uint32_t rand;
};
}  // namespace

// Return the calling wavefront's context from the global LDS.
static __gpu_local Ctx& get_ctx() {
  static constexpr uint32_t MAX_WAVEFRONTS = 1024 / 32;
  static __gpu_local Ctx ctx[MAX_WAVEFRONTS]
      __attribute__((loader_uninitialized));
  return ctx[flat_thread_id() / __gpu_num_lanes()];
}

// Type trait helpers for address spaces.
template <typename>
struct is_ptr_local {
  static constexpr bool value = false;
};
template <typename T>
struct is_ptr_local<T __gpu_local*> {
  static constexpr bool value = true;
};

template <typename PtrTy>
static constexpr bool uses_table() {
  return WP_ENABLE_TABLE && !is_ptr_local<PtrTy>::value;
}

// Returns if a wavefront should sample this access for detection. Each eligible
// access is sampled with a wave-uniform probability of 1/SAMPLE_CHANCE.
static bool should_watch(uint64_t lane_mask, uint32_t access_type) {
  // If every access is atomic we cannot have a race.
  if (!__gpu_ballot(lane_mask, !(access_type & TSAN_GPU_ACCESS_ATOMIC)))
    return false;

  bool sample = false;
  if (__gpu_is_first_in_lane(lane_mask))
    sample = bernoulli<WP_CHANCE>(get_ctx().rand);
  return __gpu_read_first_lane_u32(lane_mask, sample);
}

// Scan the watchpoint table to see if there are any points set for this
// address. Cheap lookup done for each lane in the wavefront.
template <typename PtrTy>
static uint64_t* find_watchpoint(uintptr_t addr, uint32_t size,
                                 bool expect_write, uint64_t& encoded) {
  // Local addresses never use the global watchpoint table.
  if constexpr (!uses_table<PtrTy>())
    return nullptr;

  uint64_t* wp = &watchpoints[watchpoint_slot(addr)];
  encoded = __scoped_atomic_load_n(wp, __ATOMIC_ACQUIRE, __MEMORY_SCOPE_DEVICE);

  uint64_t wp_addr;
  uint32_t wp_size;
  bool is_write;
  if (!decode_watchpoint(encoded, wp_addr, wp_size, is_write))
    return nullptr;
  if (expect_write && !is_write)
    return nullptr;
  if (ranges_overlap(wp_addr, wp_size, addr & WP_ADDR_MASK, size))
    return wp;
  return nullptr;
}

// Claim a free slot for this access by atomically flipping it from INVALID to
// the encoded watchpoint. Returns nullptr if every candidate slot is taken.
static uint64_t* insert_watchpoint(uint64_t addr, uint32_t size,
                                   bool is_write) {
  uint64_t* wp = &watchpoints[watchpoint_slot(addr)];
  const uint64_t encoded = encode_watchpoint(addr, size, is_write);
  uint64_t expected = WP_INVALID;
  if (__scoped_atomic_compare_exchange_n(wp, &expected, encoded, false,
                                         __ATOMIC_RELEASE, __ATOMIC_RELAXED,
                                         __MEMORY_SCOPE_DEVICE))
    return wp;
  return nullptr;
}

// A finder consumes a watchpoint to signal the setter that a race was observed,
// storing its own PC into the address bits so the setter can report it as the
// peer. Succeeds only if the slot still holds the value the finder matched.
static bool try_consume_watchpoint(uint64_t* wp, uint64_t encoded, void* pc) {
  uint64_t consumed =
      WP_CONSUMED_MASK | (reinterpret_cast<uintptr_t>(pc) & WP_ADDR_MASK);
  return __scoped_atomic_compare_exchange_n(wp, &encoded, consumed, false,
                                            __ATOMIC_RELEASE, __ATOMIC_RELAXED,
                                            __MEMORY_SCOPE_DEVICE);
}

// A setter tears down its own watchpoint and checks if another thread consumed
// it and triggered a race, recovering the finder's stashed PC as 'peer'.
static bool consume_watchpoint(uint64_t* wp, void*& peer) {
  uint64_t old = __scoped_atomic_exchange_n(
      wp, WP_CONSUMED_MASK, __ATOMIC_ACQUIRE, __MEMORY_SCOPE_DEVICE);
  peer = reinterpret_cast<void*>(old & WP_ADDR_MASK);
  return !(old & WP_CONSUMED_MASK);
}

// Release the slot back to the pool so it may be reused immediately after.
static void remove_watchpoint(uint64_t* wp) {
  __scoped_atomic_store_n(wp, WP_INVALID, __ATOMIC_RELAXED,
                          __MEMORY_SCOPE_DEVICE);
}

// FNV-1a digest of a byte range so wide accesses can reuse the 64-bit value
// comparison semantics.
template <typename BytePtr, typename WordPtr>
static uint64_t read_range(BytePtr bytes, [[maybe_unused]] WordPtr words,
                           uint32_t size, int scope) {
  uint64_t sum = 0xcbf29ce484222325ull;
  uint32_t i = 0;

  // Digest the unaligned prefix byte-by-byte up to a word.
  for (; i < size && ((reinterpret_cast<uintptr_t>(bytes) + i) & 7u); ++i)
    sum = (sum ^ __scoped_atomic_load_n(bytes + i, __ATOMIC_RELAXED, scope)) *
          0x100000001b3ull;

  // Digest the aligned interior with wide loads.
  for (; i + 8 <= size; i += 8)
    sum = (sum ^ __scoped_atomic_load_n(reinterpret_cast<WordPtr>(bytes + i),
                                        __ATOMIC_RELAXED, scope)) *
          0x100000001b3ull;

  // Digest the trailing bytes that do not fill a word.
  for (; i < size; ++i)
    sum = (sum ^ __scoped_atomic_load_n(bytes + i, __ATOMIC_RELAXED, scope)) *
          0x100000001b3ull;
  return sum;
}

// Snapshot the watched location for value-change detection. Larger sizes get
// converted into a single checksum.
static uint64_t read_instrumented_memory(const volatile __gpu_global void* ptr,
                                         uint32_t size) {
  const uintptr_t addr =
      reinterpret_cast<uintptr_t>(const_cast<const __gpu_global void*>(ptr));
  if ((addr & (size - 1)) == 0) {
    switch (size) {
      case 1:
        return __scoped_atomic_load_n((const volatile __gpu_global uint8_t*)ptr,
                                      __ATOMIC_RELAXED, __MEMORY_SCOPE_SYSTEM);
      case 2:
        return __scoped_atomic_load_n(
            (const volatile __gpu_global uint16_t*)ptr, __ATOMIC_RELAXED,
            __MEMORY_SCOPE_SYSTEM);
      case 4:
        return __scoped_atomic_load_n(
            (const volatile __gpu_global uint32_t*)ptr, __ATOMIC_RELAXED,
            __MEMORY_SCOPE_SYSTEM);
      case 8:
        return __scoped_atomic_load_n(
            (const volatile __gpu_global uint64_t*)ptr, __ATOMIC_RELAXED,
            __MEMORY_SCOPE_SYSTEM);
    }
  }
  return read_range((const volatile __gpu_global uint8_t*)ptr,
                    (const volatile __gpu_global uint64_t*)ptr, size,
                    __MEMORY_SCOPE_SYSTEM);
}

static uint64_t read_instrumented_memory(const volatile __gpu_local void* ptr,
                                         uint32_t size) {
  // LDS is only coherent within the workgroup, so the scope narrows to match.
  const uintptr_t addr =
      reinterpret_cast<uintptr_t>(const_cast<const __gpu_local void*>(ptr));
  if ((addr & (size - 1)) == 0) {
    switch (size) {
      case 1:
        return __scoped_atomic_load_n((const volatile __gpu_local uint8_t*)ptr,
                                      __ATOMIC_RELAXED, __MEMORY_SCOPE_WRKGRP);
      case 2:
        return __scoped_atomic_load_n((const volatile __gpu_local uint16_t*)ptr,
                                      __ATOMIC_RELAXED, __MEMORY_SCOPE_WRKGRP);
      case 4:
        return __scoped_atomic_load_n((const volatile __gpu_local uint32_t*)ptr,
                                      __ATOMIC_RELAXED, __MEMORY_SCOPE_WRKGRP);
      case 8:
        return __scoped_atomic_load_n((const volatile __gpu_local uint64_t*)ptr,
                                      __ATOMIC_RELAXED, __MEMORY_SCOPE_WRKGRP);
    }
  }
  return read_range((const volatile __gpu_local uint8_t*)ptr,
                    (const volatile __gpu_local uint64_t*)ptr, size,
                    __MEMORY_SCOPE_WRKGRP);
}

// A conflicting watchpoint already exists. Try to consume it so the thread that
// created the watchpoint will see the conflict.
template <typename PtrTy>
[[gnu::cold]] static void found_watchpoint(const PtrTy addr, uint32_t size,
                                           int access_type, void* pc,
                                           uint64_t* wp, uint64_t encoded) {
  try_consume_watchpoint(wp, encoded, pc);
}

// Trivial check to catch races between lanes inside of the same wave.
static bool intra_wave_race(uint64_t lane_mask, uintptr_t addr, int access_type,
                            uint8_t& peer_lane, uint16_t peer_thread[3]) {
  const bool is_write = (access_type & TSAN_GPU_ACCESS_WRITE) != 0;
  const bool is_atomic = (access_type & TSAN_GPU_ACCESS_ATOMIC) != 0;

  const uint64_t same_addr = __gpu_match_any_u64(lane_mask, addr);
  const uint64_t writers = __gpu_ballot(lane_mask, is_write);
  const uint64_t nonatomic = __gpu_ballot(lane_mask, !is_atomic);

  // Every lane reads the peer's coordinates so the source lane is always live.
  const uint32_t peer = 63u - __builtin_clzg(same_addr);
  const uint32_t width = __gpu_num_lanes();
  peer_lane = static_cast<uint8_t>(peer);
  peer_thread[0] = __gpu_shuffle_idx_u32(lane_mask, peer,
                                         __gpu_thread_id(__GPU_X_DIM), width);
  peer_thread[1] = __gpu_shuffle_idx_u32(lane_mask, peer,
                                         __gpu_thread_id(__GPU_Y_DIM), width);
  peer_thread[2] = __gpu_shuffle_idx_u32(lane_mask, peer,
                                         __gpu_thread_id(__GPU_Z_DIM), width);

  // The first lane of each racing group reports its own conflict.
  const bool is_race = __builtin_popcountg(same_addr) >= 2 &&
                       (same_addr & writers) && (same_addr & nonatomic);
  return is_race && __gpu_is_first_in_lane(same_addr);
}

static void delay_ns(uint64_t nsecs) {
  const uint64_t tick_rate = TICKS_PER_SEC / CLOCK_FREQ_HZ;
  const uint64_t start = __builtin_readsteadycounter();
  const uint64_t end = start + (nsecs + tick_rate - 1) / tick_rate;
#if defined(__AMDGPU__)
  __builtin_amdgcn_s_sleep(2);
  while (__builtin_readsteadycounter() < end) __builtin_amdgcn_s_sleep(15);
#else
  while (__builtin_readsteadycounter() < end) __gpu_thread_suspend();
#endif
}

// Draw a wave-uniform random delay in the range [lo, hi) nanoseconds.
static uint64_t random_delay(uint64_t lane_mask, uint64_t lo, uint64_t hi) {
  uint64_t nsecs = lo;
  if (__gpu_is_first_in_lane(lane_mask))
    nsecs = lo + xorshift32(get_ctx().rand) % (hi - lo);
  return __gpu_read_first_lane_u64(lane_mask, nsecs);
}

static void sample_delay(uint64_t lane_mask) {
  delay_ns(random_delay(lane_mask, SAMPLE_DELAY_MIN_NS, SAMPLE_DELAY_MAX_NS));
}

// The slow path, sets a watchpoint in the table and waits to see if any other
// thread tripped it while finding a watchpoint.
template <typename PtrTy>
static void watch(uint64_t lane_mask, const PtrTy addr, uint32_t size,
                  int access_type, void* pc) {
  const bool is_write = (access_type & TSAN_GPU_ACCESS_WRITE) != 0;
  const uintptr_t iaddr = reinterpret_cast<uintptr_t>(addr);

  uint8_t peer_lane;
  uint16_t peer_thread[3];
  if (intra_wave_race(lane_mask, iaddr, access_type, peer_lane, peer_thread))
    report(TSAN_GPU_INTRA_WAVE, iaddr, size, access_type, pc, nullptr,
           peer_lane, peer_thread);

  // Oversized accesses watch their base; atomics and LDS cannot be armed, and
  // nothing is armed when the table is disabled (value-only mode).
  const uint32_t wp_size = size < WP_MAX_SIZE ? size : WP_MAX_SIZE;
  const bool armable = uses_table<PtrTy>() &&
                       !(access_type & TSAN_GPU_ACCESS_ATOMIC) && iaddr != 0;
  uint64_t* wp =
      armable ? insert_watchpoint(iaddr, wp_size, is_write) : nullptr;

  const uint64_t old = read_instrumented_memory(addr, size);
  sample_delay(lane_mask);
  const uint64_t now = read_instrumented_memory(addr, size);

  void* peer = nullptr;
  if (wp && !consume_watchpoint(wp, peer))
    // A finder consumed the watchpoint, report a race with a known peer.
    report(TSAN_GPU_DATA_RACE, iaddr, size, access_type, pc, peer);
  else if (old != now)
    // The value moved under us with no finder a race of unknown origin.
    report(TSAN_GPU_UNKNOWN_ORIGIN, iaddr, size, access_type, pc);

  if (wp)
    remove_watchpoint(wp);
}

template <typename PtrTy>
static void check_access_impl(uint64_t lane_mask, const PtrTy addr,
                              uint32_t size, int access_type, void* pc) {
  // Finder side, every access probes for a conflicting watchpoint.
  if constexpr (uses_table<PtrTy>()) {
    const bool is_write = (access_type & TSAN_GPU_ACCESS_WRITE) != 0;
    uint64_t encoded;
    uint64_t* wp =
        find_watchpoint<PtrTy>((uint64_t)addr, size, !is_write, encoded);
    if (wp)
      found_watchpoint(addr, size, access_type, pc, wp, encoded);
  }

  // Single sampled path, decided per wave so the stall is reached in lockstep.
  if (should_watch(lane_mask, access_type))
    watch(lane_mask, addr, size, access_type, pc);
}

static void check_access(const volatile void* addr, uintptr_t size,
                         int access_type, void* pc) {
  // Private addresses or empty accesses by definition cannot race.
  if (__gpu_is_ptr_private(const_cast<void*>(addr)) || !size)
    return;

  if (__gpu_is_ptr_local(const_cast<void*>(addr)))
    return check_access_impl(__gpu_lane_mask(),
                             (const volatile __gpu_local void*)addr, size,
                             access_type, pc);
  check_access_impl(__gpu_lane_mask(), (const volatile __gpu_global void*)addr,
                    size, access_type, pc);
}

// Hooks to track weak memory ordering with additional scoping information.
// Currently unused with questionable practicality for the GPU case.
[[maybe_unused]] static void barrier_mb(int scope) { /* TODO */ }
[[maybe_unused]] static void barrier_wmb(int scope) { /* TODO */ }
[[maybe_unused]] static void barrier_rmb(int scope) { /* TODO */ }
[[maybe_unused]] static void barrier_release(int scope) { /* TODO */ }

// Translate an atomic builtin memory order into the barrier it implies.
static void atomic_memorder(int order, int scope) {
  if (order == __ATOMIC_RELEASE || order == __ATOMIC_ACQ_REL ||
      order == __ATOMIC_SEQ_CST)
    barrier_release(scope);
}

// Translate a standalone fence memory order into the implied barrier.
static void fence_memorder(int order, int scope) {
  switch (order) {
    case __ATOMIC_SEQ_CST:
    case __ATOMIC_ACQ_REL:
      barrier_mb(scope);
      break;
    case __ATOMIC_RELEASE:
      barrier_release(scope);
      break;
    case __ATOMIC_ACQUIRE:
    case __ATOMIC_CONSUME:
      barrier_rmb(scope);
      break;
    default:
      break;
  }
}

// Runs at the start of every instrumented kernel to seed the per-wave context
// once, so all later sampling and delays draw from a single entropy source.
static void init_watchpoints() {
  uint64_t lane_mask = __gpu_lane_mask();
  if (__gpu_is_first_in_lane(lane_mask))
    get_ctx().rand = entropy();
  __gpu_sync_threads();
}

//===----------------------------------------------------------------------===//
// Public API (ABI emitted by the ThreadSanitizer instrumentation pass)
//===----------------------------------------------------------------------===//

#define INTERFACE extern "C" SANITIZER_INTERFACE_ATTRIBUTE

INTERFACE void __tsan_kernel_entry() { init_watchpoints(); }

INTERFACE void __tsan_func_entry(void* pc) {}
INTERFACE void __tsan_func_exit() {}

INTERFACE void __tsan_ignore_thread_begin() {}
INTERFACE void __tsan_ignore_thread_end() {}

#define TSAN_READ(N)                                             \
  INTERFACE void __tsan_read##N(void* addr) {                    \
    check_access(addr, N, 0, __builtin_return_address(0));       \
  }                                                              \
  INTERFACE void __tsan_unaligned_read##N(void* addr) {          \
    check_access(addr, N, 0, __builtin_return_address(0));       \
  }                                                              \
  INTERFACE void __tsan_volatile_read##N(void* addr) {           \
    check_access(addr, N, 0, __builtin_return_address(0));       \
  }                                                              \
  INTERFACE void __tsan_unaligned_volatile_read##N(void* addr) { \
    check_access(addr, N, 0, __builtin_return_address(0));       \
  }

#define TSAN_WRITE(N)                                                          \
  INTERFACE void __tsan_write##N(void* addr) {                                 \
    check_access(addr, N, TSAN_GPU_ACCESS_WRITE, __builtin_return_address(0)); \
  }                                                                            \
  INTERFACE void __tsan_unaligned_write##N(void* addr) {                       \
    check_access(addr, N, TSAN_GPU_ACCESS_WRITE, __builtin_return_address(0)); \
  }                                                                            \
  INTERFACE void __tsan_volatile_write##N(void* addr) {                        \
    check_access(addr, N, TSAN_GPU_ACCESS_WRITE, __builtin_return_address(0)); \
  }                                                                            \
  INTERFACE void __tsan_unaligned_volatile_write##N(void* addr) {              \
    check_access(addr, N, TSAN_GPU_ACCESS_WRITE, __builtin_return_address(0)); \
  }

#define TSAN_READ_WRITE(N)                                                  \
  INTERFACE void __tsan_read_write##N(void* addr) {                         \
    check_access(addr, N, TSAN_GPU_ACCESS_WRITE | TSAN_GPU_ACCESS_COMPOUND, \
                 __builtin_return_address(0));                              \
  }                                                                         \
  INTERFACE void __tsan_unaligned_read_write##N(void* addr) {               \
    check_access(addr, N, TSAN_GPU_ACCESS_WRITE | TSAN_GPU_ACCESS_COMPOUND, \
                 __builtin_return_address(0));                              \
  }

#define TSAN_ACCESS(N) \
  TSAN_READ(N)         \
  TSAN_WRITE(N)        \
  TSAN_READ_WRITE(N)

TSAN_ACCESS(1)
TSAN_ACCESS(2)
TSAN_ACCESS(4)
TSAN_ACCESS(8)
TSAN_ACCESS(16)

INTERFACE void __tsan_read_range(void* addr, uintptr_t size) {
  check_access(addr, size, 0, __builtin_return_address(0));
}

INTERFACE void __tsan_write_range(void* addr, uintptr_t size) {
  check_access(addr, size, TSAN_GPU_ACCESS_WRITE, __builtin_return_address(0));
}

#define TSAN_ATOMIC_LOAD(N)                                                    \
  INTERFACE uint##N##_t __tsan_atomic##N##_load(const volatile uint##N##_t* a, \
                                                int order, int scope) {        \
    atomic_memorder(order, scope);                                             \
    check_access(a, N / 8, TSAN_GPU_ACCESS_ATOMIC,                             \
                 __builtin_return_address(0));                                 \
    return __scoped_atomic_load_n(a, order, scope);                            \
  }

#define TSAN_ATOMIC_STORE(N)                                               \
  INTERFACE void __tsan_atomic##N##_store(                                 \
      volatile uint##N##_t* a, uint##N##_t v, int order, int scope) {      \
    atomic_memorder(order, scope);                                         \
    check_access(a, N / 8, TSAN_GPU_ACCESS_ATOMIC | TSAN_GPU_ACCESS_WRITE, \
                 __builtin_return_address(0));                             \
    __scoped_atomic_store_n(a, v, order, scope);                           \
  }

#define TSAN_ATOMIC_EXCHANGE(N)                                            \
  INTERFACE uint##N##_t __tsan_atomic##N##_exchange(                       \
      volatile uint##N##_t* a, uint##N##_t v, int order, int scope) {      \
    atomic_memorder(order, scope);                                         \
    check_access(a, N / 8, TSAN_GPU_ACCESS_ATOMIC | TSAN_GPU_ACCESS_WRITE, \
                 __builtin_return_address(0));                             \
    return __scoped_atomic_exchange_n(a, v, order, scope);                 \
  }

#define TSAN_ATOMIC_FETCH_OP(N, op)                                        \
  INTERFACE uint##N##_t __tsan_atomic##N##_fetch_##op(                     \
      volatile uint##N##_t* a, uint##N##_t v, int order, int scope) {      \
    atomic_memorder(order, scope);                                         \
    check_access(a, N / 8, TSAN_GPU_ACCESS_ATOMIC | TSAN_GPU_ACCESS_WRITE, \
                 __builtin_return_address(0));                             \
    return __scoped_atomic_fetch_##op(a, v, order, scope);                 \
  }

#define TSAN_ATOMIC_CAS(N)                                                   \
  INTERFACE uint##N##_t __tsan_atomic##N##_compare_exchange_val(             \
      volatile uint##N##_t* a, uint##N##_t c, uint##N##_t v, int order_succ, \
      int order_fail, int scope) {                                           \
    atomic_memorder(order_succ, scope);                                      \
    check_access(a, N / 8, TSAN_GPU_ACCESS_ATOMIC | TSAN_GPU_ACCESS_WRITE,   \
                 __builtin_return_address(0));                               \
    __scoped_atomic_compare_exchange_n(a, &c, v, false, order_succ,          \
                                       order_fail, scope);                   \
    return c;                                                                \
  }

#define TSAN_ATOMIC_OPS(N)      \
  TSAN_ATOMIC_LOAD(N)           \
  TSAN_ATOMIC_STORE(N)          \
  TSAN_ATOMIC_EXCHANGE(N)       \
  TSAN_ATOMIC_FETCH_OP(N, add)  \
  TSAN_ATOMIC_FETCH_OP(N, sub)  \
  TSAN_ATOMIC_FETCH_OP(N, and)  \
  TSAN_ATOMIC_FETCH_OP(N, or)   \
  TSAN_ATOMIC_FETCH_OP(N, xor)  \
  TSAN_ATOMIC_FETCH_OP(N, nand) \
  TSAN_ATOMIC_CAS(N)

TSAN_ATOMIC_OPS(8)
TSAN_ATOMIC_OPS(16)
TSAN_ATOMIC_OPS(32)
TSAN_ATOMIC_OPS(64)

INTERFACE void __tsan_atomic_thread_fence(int order, int scope) {
  fence_memorder(order, scope);
  __scoped_atomic_thread_fence(order, scope);
}

INTERFACE void __tsan_atomic_signal_fence(int order, int scope) {
  fence_memorder(order, scope);
  __atomic_signal_fence(order);
}

INTERFACE void __tsan_vptr_update(void* vptr_p, void* new_val) {}
INTERFACE void __tsan_vptr_read(void* vptr_p) {}

INTERFACE void* __tsan_memmove(void* dst, const void* src, uintptr_t sz) {
  void* pc = __builtin_return_address(0);
  check_access(dst, sz, TSAN_GPU_ACCESS_WRITE, pc);
  check_access(src, sz, 0, pc);
  return __builtin_memmove(dst, src, sz);
}

INTERFACE void* __tsan_memcpy(void* dst, const void* src, uintptr_t sz) {
  void* pc = __builtin_return_address(0);
  check_access(dst, sz, TSAN_GPU_ACCESS_WRITE, pc);
  check_access(src, sz, 0, pc);
  return __builtin_memcpy(dst, src, sz);
}

INTERFACE void* __tsan_memset(void* dst, int c, uintptr_t sz) {
  check_access(dst, sz, TSAN_GPU_ACCESS_WRITE, __builtin_return_address(0));
  return __builtin_memset(dst, c, sz);
}

// FIXME: Required to resolve the workgroup size without a set ABI version.
#ifdef __AMDGPU__
extern "C" const inline uint32_t __oclc_ABI_version = 0;
[[gnu::alias("__oclc_ABI_version")]] const uint32_t __oclc_ABI_version__;
#endif
