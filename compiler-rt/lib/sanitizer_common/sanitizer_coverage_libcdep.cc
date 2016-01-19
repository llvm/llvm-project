//===-- sanitizer_coverage.cc ---------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Sanitizer Coverage.
// This file implements run-time support for a poor man's coverage tool.
//
// Compiler instrumentation:
// For every interesting basic block the compiler injects the following code:
// if (Guard < 0) {
//    __sanitizer_cov(&Guard);
// }
// At the module start up time __sanitizer_cov_module_init sets the guards
// to consecutive negative numbers (-1, -2, -3, ...).
// It's fine to call __sanitizer_cov more than once for a given block.
//
// Run-time:
//  - __sanitizer_cov(): record that we've executed the PC (GET_CALLER_PC).
//    and atomically set Guard to -Guard.
//  - __sanitizer_cov_dump: dump the coverage data to disk.
//  For every module of the current process that has coverage data
//  this will create a file module_name.PID.sancov.
//
// The file format is simple: the first 8 bytes is the magic,
// one of 0xC0BFFFFFFFFFFF64 and 0xC0BFFFFFFFFFFF32. The last byte of the
// magic defines the size of the following offsets.
// The rest of the data is the offsets in the module.
//
// Eventually, this coverage implementation should be obsoleted by a more
// powerful general purpose Clang/LLVM coverage instrumentation.
// Consider this implementation as prototype.
//
// FIXME: support (or at least test with) dlclose.
//===----------------------------------------------------------------------===//

#include "sanitizer_allocator_internal.h"
#include "sanitizer_common.h"
#include "sanitizer_libc.h"
#include "sanitizer_mutex.h"
#include "sanitizer_procmaps.h"
#include "sanitizer_stacktrace.h"
#include "sanitizer_symbolizer.h"
#include "sanitizer_flags.h"

static const u64 kMagic64 = 0xC0BFFFFFFFFFFF64ULL;
static const u64 kMagic32 = 0xC0BFFFFFFFFFFF32ULL;
static const uptr kNumWordsForMagic = SANITIZER_WORDSIZE == 64 ? 1 : 2;
static const u64 kMagic = SANITIZER_WORDSIZE == 64 ? kMagic64 : kMagic32;

static atomic_uint32_t dump_once_guard;  // Ensure that CovDump runs only once.

static atomic_uintptr_t coverage_counter;
static atomic_uintptr_t caller_callee_counter;

static void ResetGlobalCounters() {
  return atomic_store(&coverage_counter, 0, memory_order_relaxed);
  return atomic_store(&caller_callee_counter, 0, memory_order_relaxed);
}

// pc_array is the array containing the covered PCs.
// To make the pc_array thread- and async-signal-safe it has to be large enough.
// 128M counters "ought to be enough for anybody" (4M on 32-bit).

// With coverage_direct=1 in ASAN_OPTIONS, pc_array memory is mapped to a file.
// In this mode, __sanitizer_cov_dump does nothing, and CovUpdateMapping()
// dump current memory layout to another file.

static bool cov_sandboxed = false;
static fd_t cov_fd = kInvalidFd;
static unsigned int cov_max_block_size = 0;
static bool coverage_enabled = false;
static const char *coverage_dir;

namespace __sanitizer {

class CoverageData {
 public:
  void Init();
  void Enable();
  void Disable();
  void ReInit();
  void BeforeFork();
  void AfterFork(int child_pid);
  void Extend(uptr npcs);
  void Add(uptr pc, u32 *guard);
  void IndirCall(uptr caller, uptr callee, uptr callee_cache[],
                 uptr cache_size);
  void DumpCallerCalleePairs();
  void DumpTrace();
  void DumpAsBitSet();
  void DumpCounters();
  void DumpOffsets();
  void DumpAll();

  ALWAYS_INLINE
  void TraceBasicBlock(u32 *id);

  void InitializeGuardArray(s32 *guards);
  void InitializeGuards(s32 *guards, uptr n, const char *module_name,
                        uptr caller_pc);
  void InitializeCounters(u8 *counters, uptr n);
  void ReinitializeGuards();
  uptr GetNumberOf8bitCounters();
  uptr Update8bitCounterBitsetAndClearCounters(u8 *bitset);

  uptr *data();
  uptr size() const;
  uptr *buffer() const { return pc_buffer; }

 private:
  struct NamedPcRange {
    const char *copied_module_name;
    uptr beg, end; // elements [beg,end) in pc_array.
  };

  void DirectOpen();
  void UpdateModuleNameVec(uptr caller_pc, uptr range_beg, uptr range_end);
  void GetRangeOffsets(const NamedPcRange& r, Symbolizer* s,
      InternalMmapVector<uptr>* offsets) const;

  // Maximal size pc array may ever grow.
  // We MmapNoReserve this space to ensure that the array is contiguous.
  static const uptr kPcArrayMaxSize = FIRST_32_SECOND_64(
      1 << (SANITIZER_ANDROID ? 24 : (SANITIZER_WINDOWS ? 27 : 26)),
      1 << 27);
  // The amount file mapping for the pc array is grown by.
  static const uptr kPcArrayMmapSize = 64 * 1024;

  // pc_array is allocated with MmapNoReserveOrDie and so it uses only as
  // much RAM as it really needs.
  uptr *pc_array;
  // Index of the first available pc_array slot.
  atomic_uintptr_t pc_array_index;
  // Array size.
  atomic_uintptr_t pc_array_size;
  // Current file mapped size of the pc array.
  uptr pc_array_mapped_size;
  // Descriptor of the file mapped pc array.
  fd_t pc_fd;

  uptr *pc_buffer;

  // Vector of coverage guard arrays, protected by mu.
  InternalMmapVectorNoCtor<s32*> guard_array_vec;

  // Vector of module and compilation unit pc ranges.
  InternalMmapVectorNoCtor<NamedPcRange> comp_unit_name_vec;
  InternalMmapVectorNoCtor<NamedPcRange> module_name_vec;

  struct CounterAndSize {
    u8 *counters;
    uptr n;
  };

  InternalMmapVectorNoCtor<CounterAndSize> counters_vec;
  uptr num_8bit_counters;

  // Caller-Callee (cc) array, size and current index.
  static const uptr kCcArrayMaxSize = FIRST_32_SECOND_64(1 << 18, 1 << 24);
  uptr **cc_array;
  atomic_uintptr_t cc_array_index;
  atomic_uintptr_t cc_array_size;

  // Tracing event array, size and current pointer.
  // We record all events (basic block entries) in a global buffer of u32
  // values. Each such value is the index in pc_array.
  // So far the tracing is highly experimental:
  //   - not thread-safe;
  //   - does not support long traces;
  //   - not tuned for performance.
  static const uptr kTrEventArrayMaxSize = FIRST_32_SECOND_64(1 << 22, 1 << 30);
  u32 *tr_event_array;
  uptr tr_event_array_size;
  u32 *tr_event_pointer;
  static const uptr kTrPcArrayMaxSize    = FIRST_32_SECOND_64(1 << 22, 1 << 27);

  StaticSpinMutex mu;
};

static CoverageData coverage_data;

void CovUpdateMapping(const char *path, uptr caller_pc = 0);

void CoverageData::DirectOpen() {
  InternalScopedString path(kMaxPathLength);
  internal_snprintf((char *)path.data(), path.size(), "%s/%zd.sancov.raw",
                    coverage_dir, internal_getpid());
  pc_fd = OpenFile(path.data(), RdWr);
  if (pc_fd == kInvalidFd) {
    Report("Coverage: failed to open %s for reading/writing\n", path.data());
    Die();
  }

  pc_array_mapped_size = 0;
  CovUpdateMapping(coverage_dir);
}

void CoverageData::Init() {
  pc_fd = kInvalidFd;
}

void CoverageData::Enable() {
  if (pc_array)
    return;
  pc_array = reinterpret_cast<uptr *>(
      MmapNoReserveOrDie(sizeof(uptr) * kPcArrayMaxSize, "CovInit"));
  atomic_store(&pc_array_index, 0, memory_order_relaxed);
  if (common_flags()->coverage_direct) {
    atomic_store(&pc_array_size, 0, memory_order_relaxed);
  } else {
    atomic_store(&pc_array_size, kPcArrayMaxSize, memory_order_relaxed);
  }

  pc_buffer = nullptr;
  if (common_flags()->coverage_pc_buffer)
    pc_buffer = reinterpret_cast<uptr *>(MmapNoReserveOrDie(
        sizeof(uptr) * kPcArrayMaxSize, "CovInit::pc_buffer"));

  cc_array = reinterpret_cast<uptr **>(MmapNoReserveOrDie(
      sizeof(uptr *) * kCcArrayMaxSize, "CovInit::cc_array"));
  atomic_store(&cc_array_size, kCcArrayMaxSize, memory_order_relaxed);
  atomic_store(&cc_array_index, 0, memory_order_relaxed);

  // Allocate tr_event_array with a guard page at the end.
  tr_event_array = reinterpret_cast<u32 *>(MmapNoReserveOrDie(
      sizeof(tr_event_array[0]) * kTrEventArrayMaxSize + GetMmapGranularity(),
      "CovInit::tr_event_array"));
  MprotectNoAccess(
      reinterpret_cast<uptr>(&tr_event_array[kTrEventArrayMaxSize]),
      GetMmapGranularity());
  tr_event_array_size = kTrEventArrayMaxSize;
  tr_event_pointer = tr_event_array;

  num_8bit_counters = 0;
}

void CoverageData::InitializeGuardArray(s32 *guards) {
  Enable();  // Make sure coverage is enabled at this point.
  s32 n = guards[0];
  for (s32 j = 1; j <= n; j++) {
    uptr idx = atomic_load_relaxed(&pc_array_index);
    atomic_store_relaxed(&pc_array_index, idx + 1);
    guards[j] = -static_cast<s32>(idx + 1);
  }
}

void CoverageData::Disable() {
  if (pc_array) {
    UnmapOrDie(pc_array, sizeof(uptr) * kPcArrayMaxSize);
    pc_array = nullptr;
  }
  if (cc_array) {
    UnmapOrDie(cc_array, sizeof(uptr *) * kCcArrayMaxSize);
    cc_array = nullptr;
  }
  if (pc_buffer) {
    UnmapOrDie(pc_buffer, sizeof(uptr) * kPcArrayMaxSize);
    pc_buffer = nullptr;
  }
  if (tr_event_array) {
    UnmapOrDie(tr_event_array,
               sizeof(tr_event_array[0]) * kTrEventArrayMaxSize +
                   GetMmapGranularity());
    tr_event_array = nullptr;
    tr_event_pointer = nullptr;
  }
  if (pc_fd != kInvalidFd) {
    CloseFile(pc_fd);
    pc_fd = kInvalidFd;
  }
}

void CoverageData::ReinitializeGuards() {
  // Assuming single thread.
  atomic_store(&pc_array_index, 0, memory_order_relaxed);
  for (uptr i = 0; i < guard_array_vec.size(); i++)
    InitializeGuardArray(guard_array_vec[i]);
}

void CoverageData::ReInit() {
  Disable();
  if (coverage_enabled) {
    if (common_flags()->coverage_direct) {
      // In memory-mapped mode we must extend the new file to the known array
      // size.
      uptr size = atomic_load(&pc_array_size, memory_order_relaxed);
      uptr npcs = size / sizeof(uptr);
      Enable();
      if (size) Extend(npcs);
      if (coverage_enabled) CovUpdateMapping(coverage_dir);
    } else {
      Enable();
    }
  }
  // Re-initialize the guards.
  // We are single-threaded now, no need to grab any lock.
  CHECK_EQ(atomic_load(&pc_array_index, memory_order_relaxed), 0);
  ReinitializeGuards();
}

void CoverageData::BeforeFork() {
  mu.Lock();
}

void CoverageData::AfterFork(int child_pid) {
  // We are single-threaded so it's OK to release the lock early.
  mu.Unlock();
  if (child_pid == 0) ReInit();
}

// Extend coverage PC array to fit additional npcs elements.
void CoverageData::Extend(uptr npcs) {
  if (!common_flags()->coverage_direct) return;
  SpinMutexLock l(&mu);

  uptr size = atomic_load(&pc_array_size, memory_order_relaxed);
  size += npcs * sizeof(uptr);

  if (coverage_enabled && size > pc_array_mapped_size) {
    if (pc_fd == kInvalidFd) DirectOpen();
    CHECK_NE(pc_fd, kInvalidFd);

    uptr new_mapped_size = pc_array_mapped_size;
    while (size > new_mapped_size) new_mapped_size += kPcArrayMmapSize;
    CHECK_LE(new_mapped_size, sizeof(uptr) * kPcArrayMaxSize);

    // Extend the file and map the new space at the end of pc_array.
    uptr res = internal_ftruncate(pc_fd, new_mapped_size);
    int err;
    if (internal_iserror(res, &err)) {
      Printf("failed to extend raw coverage file: %d\n", err);
      Die();
    }

    uptr next_map_base = ((uptr)pc_array) + pc_array_mapped_size;
    void *p = MapWritableFileToMemory((void *)next_map_base,
                                      new_mapped_size - pc_array_mapped_size,
                                      pc_fd, pc_array_mapped_size);
    CHECK_EQ((uptr)p, next_map_base);
    pc_array_mapped_size = new_mapped_size;
  }

  atomic_store(&pc_array_size, size, memory_order_release);
}

void CoverageData::InitializeCounters(u8 *counters, uptr n) {
  if (!counters) return;
  CHECK_EQ(reinterpret_cast<uptr>(counters) % 16, 0);
  n = RoundUpTo(n, 16); // The compiler must ensure that counters is 16-aligned.
  SpinMutexLock l(&mu);
  counters_vec.push_back({counters, n});
  num_8bit_counters += n;
}

void CoverageData::UpdateModuleNameVec(uptr caller_pc, uptr range_beg,
                                       uptr range_end) {
  auto sym = Symbolizer::GetOrInit();
  if (!sym)
    return;
  const char *module_name = sym->GetModuleNameForPc(caller_pc);
  if (!module_name) return;
  if (module_name_vec.empty() ||
      module_name_vec.back().copied_module_name != module_name)
    module_name_vec.push_back({module_name, range_beg, range_end});
  else
    module_name_vec.back().end = range_end;
}

void CoverageData::InitializeGuards(s32 *guards, uptr n,
                                    const char *comp_unit_name,
                                    uptr caller_pc) {
  // The array 'guards' has n+1 elements, we use the element zero
  // to store 'n'.
  CHECK_LT(n, 1 << 30);
  guards[0] = static_cast<s32>(n);
  InitializeGuardArray(guards);
  SpinMutexLock l(&mu);
  uptr range_end = atomic_load(&pc_array_index, memory_order_relaxed);
  uptr range_beg = range_end - n;
  comp_unit_name_vec.push_back({comp_unit_name, range_beg, range_end});
  guard_array_vec.push_back(guards);
  UpdateModuleNameVec(caller_pc, range_beg, range_end);
}

static const uptr kBundleCounterBits = 16;

// When coverage_order_pcs==true and SANITIZER_WORDSIZE==64
// we insert the global counter into the first 16 bits of the PC.
uptr BundlePcAndCounter(uptr pc, uptr counter) {
  if (SANITIZER_WORDSIZE != 64 || !common_flags()->coverage_order_pcs)
    return pc;
  static const uptr kMaxCounter = (1 << kBundleCounterBits) - 1;
  if (counter > kMaxCounter)
    counter = kMaxCounter;
  CHECK_EQ(0, pc >> (SANITIZER_WORDSIZE - kBundleCounterBits));
  return pc | (counter << (SANITIZER_WORDSIZE - kBundleCounterBits));
}

uptr UnbundlePc(uptr bundle) {
  if (SANITIZER_WORDSIZE != 64 || !common_flags()->coverage_order_pcs)
    return bundle;
  return (bundle << kBundleCounterBits) >> kBundleCounterBits;
}

uptr UnbundleCounter(uptr bundle) {
  if (SANITIZER_WORDSIZE != 64 || !common_flags()->coverage_order_pcs)
    return 0;
  return bundle >> (SANITIZER_WORDSIZE - kBundleCounterBits);
}

// If guard is negative, atomically set it to -guard and store the PC in
// pc_array.
void CoverageData::Add(uptr pc, u32 *guard) {
  atomic_uint32_t *atomic_guard = reinterpret_cast<atomic_uint32_t*>(guard);
  s32 guard_value = atomic_load(atomic_guard, memory_order_relaxed);
  if (guard_value >= 0) return;

  atomic_store(atomic_guard, -guard_value, memory_order_relaxed);
  if (!pc_array) return;

  uptr idx = -guard_value - 1;
  if (idx >= atomic_load(&pc_array_index, memory_order_acquire))
    return;  // May happen after fork when pc_array_index becomes 0.
  CHECK_LT(idx * sizeof(uptr),
           atomic_load(&pc_array_size, memory_order_acquire));
  uptr counter = atomic_fetch_add(&coverage_counter, 1, memory_order_relaxed);
  pc_array[idx] = BundlePcAndCounter(pc, counter);
  if (pc_buffer) pc_buffer[counter] = pc;
}

// Registers a pair caller=>callee.
// When a given caller is seen for the first time, the callee_cache is added
// to the global array cc_array, callee_cache[0] is set to caller and
// callee_cache[1] is set to cache_size.
// Then we are trying to add callee to callee_cache [2,cache_size) if it is
// not there yet.
// If the cache is full we drop the callee (may want to fix this later).
void CoverageData::IndirCall(uptr caller, uptr callee, uptr callee_cache[],
                             uptr cache_size) {
  if (!cc_array) return;
  atomic_uintptr_t *atomic_callee_cache =
      reinterpret_cast<atomic_uintptr_t *>(callee_cache);
  uptr zero = 0;
  if (atomic_compare_exchange_strong(&atomic_callee_cache[0], &zero, caller,
                                     memory_order_seq_cst)) {
    uptr idx = atomic_fetch_add(&cc_array_index, 1, memory_order_relaxed);
    CHECK_LT(idx * sizeof(uptr),
             atomic_load(&cc_array_size, memory_order_acquire));
    callee_cache[1] = cache_size;
    cc_array[idx] = callee_cache;
  }
  CHECK_EQ(atomic_load(&atomic_callee_cache[0], memory_order_relaxed), caller);
  for (uptr i = 2; i < cache_size; i++) {
    uptr was = 0;
    if (atomic_compare_exchange_strong(&atomic_callee_cache[i], &was, callee,
                                       memory_order_seq_cst)) {
      atomic_fetch_add(&caller_callee_counter, 1, memory_order_relaxed);
      return;
    }
    if (was == callee)  // Already have this callee.
      return;
  }
}

uptr CoverageData::GetNumberOf8bitCounters() {
  return num_8bit_counters;
}

// Map every 8bit counter to a 8-bit bitset and clear the counter.
uptr CoverageData::Update8bitCounterBitsetAndClearCounters(u8 *bitset) {
  uptr num_new_bits = 0;
  uptr cur = 0;
  // For better speed we map 8 counters to 8 bytes of bitset at once.
  static const uptr kBatchSize = 8;
  CHECK_EQ(reinterpret_cast<uptr>(bitset) % kBatchSize, 0);
  for (uptr i = 0, len = counters_vec.size(); i < len; i++) {
    u8 *c = counters_vec[i].counters;
    uptr n = counters_vec[i].n;
    CHECK_EQ(n % 16, 0);
    CHECK_EQ(cur % kBatchSize, 0);
    CHECK_EQ(reinterpret_cast<uptr>(c) % kBatchSize, 0);
    if (!bitset) {
      internal_bzero_aligned16(c, n);
      cur += n;
      continue;
    }
    for (uptr j = 0; j < n; j += kBatchSize, cur += kBatchSize) {
      CHECK_LT(cur, num_8bit_counters);
      u64 *pc64 = reinterpret_cast<u64*>(c + j);
      u64 *pb64 = reinterpret_cast<u64*>(bitset + cur);
      u64 c64 = *pc64;
      u64 old_bits_64 = *pb64;
      u64 new_bits_64 = old_bits_64;
      if (c64) {
        *pc64 = 0;
        for (uptr k = 0; k < kBatchSize; k++) {
          u64 x = (c64 >> (8 * k)) & 0xff;
          if (x) {
            u64 bit = 0;
            /**/ if (x >= 128) bit = 128;
            else if (x >= 32) bit = 64;
            else if (x >= 16) bit = 32;
            else if (x >= 8) bit = 16;
            else if (x >= 4) bit = 8;
            else if (x >= 3) bit = 4;
            else if (x >= 2) bit = 2;
            else if (x >= 1) bit = 1;
            u64 mask = bit << (8 * k);
            if (!(new_bits_64 & mask)) {
              num_new_bits++;
              new_bits_64 |= mask;
            }
          }
        }
        *pb64 = new_bits_64;
      }
    }
  }
  CHECK_EQ(cur, num_8bit_counters);
  return num_new_bits;
}

uptr *CoverageData::data() {
  return pc_array;
}

uptr CoverageData::size() const {
  return atomic_load(&pc_array_index, memory_order_relaxed);
}

// Block layout for packed file format: header, followed by module name (no
// trailing zero), followed by data blob.
struct CovHeader {
  int pid;
  unsigned int module_name_length;
  unsigned int data_length;
};

static void CovWritePacked(int pid, const char *module, const void *blob,
                           unsigned int blob_size) {
  if (cov_fd == kInvalidFd) return;
  unsigned module_name_length = internal_strlen(module);
  CovHeader header = {pid, module_name_length, blob_size};

  if (cov_max_block_size == 0) {
    // Writing to a file. Just go ahead.
    WriteToFile(cov_fd, &header, sizeof(header));
    WriteToFile(cov_fd, module, module_name_length);
    WriteToFile(cov_fd, blob, blob_size);
  } else {
    // Writing to a socket. We want to split the data into appropriately sized
    // blocks.
    InternalScopedBuffer<char> block(cov_max_block_size);
    CHECK_EQ((uptr)block.data(), (uptr)(CovHeader *)block.data());
    uptr header_size_with_module = sizeof(header) + module_name_length;
    CHECK_LT(header_size_with_module, cov_max_block_size);
    unsigned int max_payload_size =
        cov_max_block_size - header_size_with_module;
    char *block_pos = block.data();
    internal_memcpy(block_pos, &header, sizeof(header));
    block_pos += sizeof(header);
    internal_memcpy(block_pos, module, module_name_length);
    block_pos += module_name_length;
    char *block_data_begin = block_pos;
    const char *blob_pos = (const char *)blob;
    while (blob_size > 0) {
      unsigned int payload_size = Min(blob_size, max_payload_size);
      blob_size -= payload_size;
      internal_memcpy(block_data_begin, blob_pos, payload_size);
      blob_pos += payload_size;
      ((CovHeader *)block.data())->data_length = payload_size;
      WriteToFile(cov_fd, block.data(), header_size_with_module + payload_size);
    }
  }
}

// If packed = false: <name>.<pid>.<sancov> (name = module name).
// If packed = true and name == 0: <pid>.<sancov>.<packed>.
// If packed = true and name != 0: <name>.<sancov>.<packed> (name is
// user-supplied).
static fd_t CovOpenFile(InternalScopedString *path, bool packed,
                       const char *name, const char *extension = "sancov") {
  path->clear();
  if (!packed) {
    CHECK(name);
    path->append("%s/%s.%zd.%s", coverage_dir, name, internal_getpid(),
                extension);
  } else {
    if (!name)
      path->append("%s/%zd.%s.packed", coverage_dir, internal_getpid(),
                  extension);
    else
      path->append("%s/%s.%s.packed", coverage_dir, name, extension);
  }
  error_t err;
  fd_t fd = OpenFile(path->data(), WrOnly, &err);
  if (fd == kInvalidFd)
    Report("SanitizerCoverage: failed to open %s for writing (reason: %d)\n",
           path->data(), err);
  return fd;
}

// Dump trace PCs and trace events into two separate files.
void CoverageData::DumpTrace() {
  uptr max_idx = tr_event_pointer - tr_event_array;
  if (!max_idx) return;
  auto sym = Symbolizer::GetOrInit();
  if (!sym)
    return;
  InternalScopedString out(32 << 20);
  for (uptr i = 0, n = size(); i < n; i++) {
    const char *module_name = "<unknown>";
    uptr module_address = 0;
    sym->GetModuleNameAndOffsetForPC(UnbundlePc(pc_array[i]), &module_name,
                                     &module_address);
    out.append("%s 0x%zx\n", module_name, module_address);
  }
  InternalScopedString path(kMaxPathLength);
  fd_t fd = CovOpenFile(&path, false, "trace-points");
  if (fd == kInvalidFd) return;
  WriteToFile(fd, out.data(), out.length());
  CloseFile(fd);

  fd = CovOpenFile(&path, false, "trace-compunits");
  if (fd == kInvalidFd) return;
  out.clear();
  for (uptr i = 0; i < comp_unit_name_vec.size(); i++)
    out.append("%s\n", comp_unit_name_vec[i].copied_module_name);
  WriteToFile(fd, out.data(), out.length());
  CloseFile(fd);

  fd = CovOpenFile(&path, false, "trace-events");
  if (fd == kInvalidFd) return;
  uptr bytes_to_write = max_idx * sizeof(tr_event_array[0]);
  u8 *event_bytes = reinterpret_cast<u8*>(tr_event_array);
  // The trace file could be huge, and may not be written with a single syscall.
  while (bytes_to_write) {
    uptr actually_written;
    if (WriteToFile(fd, event_bytes, bytes_to_write, &actually_written) &&
        actually_written <= bytes_to_write) {
      bytes_to_write -= actually_written;
      event_bytes += actually_written;
    } else {
      break;
    }
  }
  CloseFile(fd);
  VReport(1, " CovDump: Trace: %zd PCs written\n", size());
  VReport(1, " CovDump: Trace: %zd Events written\n", max_idx);
}

// This function dumps the caller=>callee pairs into a file as a sequence of
// lines like "module_name offset".
void CoverageData::DumpCallerCalleePairs() {
  uptr max_idx = atomic_load(&cc_array_index, memory_order_relaxed);
  if (!max_idx) return;
  auto sym = Symbolizer::GetOrInit();
  if (!sym)
    return;
  InternalScopedString out(32 << 20);
  uptr total = 0;
  for (uptr i = 0; i < max_idx; i++) {
    uptr *cc_cache = cc_array[i];
    CHECK(cc_cache);
    uptr caller = cc_cache[0];
    uptr n_callees = cc_cache[1];
    const char *caller_module_name = "<unknown>";
    uptr caller_module_address = 0;
    sym->GetModuleNameAndOffsetForPC(caller, &caller_module_name,
                                     &caller_module_address);
    for (uptr j = 2; j < n_callees; j++) {
      uptr callee = cc_cache[j];
      if (!callee) break;
      total++;
      const char *callee_module_name = "<unknown>";
      uptr callee_module_address = 0;
      sym->GetModuleNameAndOffsetForPC(callee, &callee_module_name,
                                       &callee_module_address);
      out.append("%s 0x%zx\n%s 0x%zx\n", caller_module_name,
                 caller_module_address, callee_module_name,
                 callee_module_address);
    }
  }
  InternalScopedString path(kMaxPathLength);
  fd_t fd = CovOpenFile(&path, false, "caller-callee");
  if (fd == kInvalidFd) return;
  WriteToFile(fd, out.data(), out.length());
  CloseFile(fd);
  VReport(1, " CovDump: %zd caller-callee pairs written\n", total);
}

// Record the current PC into the event buffer.
// Every event is a u32 value (index in tr_pc_array_index) so we compute
// it once and then cache in the provided 'cache' storage.
//
// This function will eventually be inlined by the compiler.
void CoverageData::TraceBasicBlock(u32 *id) {
  // Will trap here if
  //  1. coverage is not enabled at run-time.
  //  2. The array tr_event_array is full.
  *tr_event_pointer = *id - 1;
  tr_event_pointer++;
}

void CoverageData::DumpCounters() {
  if (!common_flags()->coverage_counters) return;
  uptr n = coverage_data.GetNumberOf8bitCounters();
  if (!n) return;
  InternalScopedBuffer<u8> bitset(n);
  coverage_data.Update8bitCounterBitsetAndClearCounters(bitset.data());
  InternalScopedString path(kMaxPathLength);

  for (uptr m = 0; m < module_name_vec.size(); m++) {
    auto r = module_name_vec[m];
    CHECK(r.copied_module_name);
    CHECK_LE(r.beg, r.end);
    CHECK_LE(r.end, size());
    const char *base_name = StripModuleName(r.copied_module_name);
    fd_t fd =
        CovOpenFile(&path, /* packed */ false, base_name, "counters-sancov");
    if (fd == kInvalidFd) return;
    WriteToFile(fd, bitset.data() + r.beg, r.end - r.beg);
    CloseFile(fd);
    VReport(1, " CovDump: %zd counters written for '%s'\n", r.end - r.beg,
            base_name);
  }
}

void CoverageData::DumpAsBitSet() {
  if (!common_flags()->coverage_bitset) return;
  if (!size()) return;
  InternalScopedBuffer<char> out(size());
  InternalScopedString path(kMaxPathLength);
  for (uptr m = 0; m < module_name_vec.size(); m++) {
    uptr n_set_bits = 0;
    auto r = module_name_vec[m];
    CHECK(r.copied_module_name);
    CHECK_LE(r.beg, r.end);
    CHECK_LE(r.end, size());
    for (uptr i = r.beg; i < r.end; i++) {
      uptr pc = UnbundlePc(pc_array[i]);
      out[i] = pc ? '1' : '0';
      if (pc)
        n_set_bits++;
    }
    const char *base_name = StripModuleName(r.copied_module_name);
    fd_t fd = CovOpenFile(&path, /* packed */false, base_name, "bitset-sancov");
    if (fd == kInvalidFd) return;
    WriteToFile(fd, out.data() + r.beg, r.end - r.beg);
    CloseFile(fd);
    VReport(1,
            " CovDump: bitset of %zd bits written for '%s', %zd bits are set\n",
            r.end - r.beg, base_name, n_set_bits);
  }
}


void CoverageData::GetRangeOffsets(const NamedPcRange& r, Symbolizer* sym,
    InternalMmapVector<uptr>* offsets) const {
  offsets->clear();
  for (uptr i = 0; i < kNumWordsForMagic; i++)
    offsets->push_back(0);
  CHECK(r.copied_module_name);
  CHECK_LE(r.beg, r.end);
  CHECK_LE(r.end, size());
  for (uptr i = r.beg; i < r.end; i++) {
    uptr pc = UnbundlePc(pc_array[i]);
    uptr counter = UnbundleCounter(pc_array[i]);
    if (!pc) continue; // Not visited.
    uptr offset = 0;
    sym->GetModuleNameAndOffsetForPC(pc, nullptr, &offset);
    offsets->push_back(BundlePcAndCounter(offset, counter));
  }

  CHECK_GE(offsets->size(), kNumWordsForMagic);
  SortArray(offsets->data(), offsets->size());
  for (uptr i = 0; i < offsets->size(); i++)
    (*offsets)[i] = UnbundlePc((*offsets)[i]);
}

void CoverageData::DumpOffsets() {
  auto sym = Symbolizer::GetOrInit();
  if (!common_flags()->coverage_pcs) return;
  CHECK_NE(sym, nullptr);
  InternalMmapVector<uptr> offsets(0);
  InternalScopedString path(kMaxPathLength);
  for (uptr m = 0; m < module_name_vec.size(); m++) {
    auto r = module_name_vec[m];
    GetRangeOffsets(r, sym, &offsets);

    uptr num_offsets = offsets.size() - kNumWordsForMagic;
    u64 *magic_p = reinterpret_cast<u64*>(offsets.data());
    CHECK_EQ(*magic_p, 0ULL);
    // FIXME: we may want to write 32-bit offsets even in 64-mode
    // if all the offsets are small enough.
    *magic_p = kMagic;

    const char *module_name = StripModuleName(r.copied_module_name);
    if (cov_sandboxed) {
      if (cov_fd != kInvalidFd) {
        CovWritePacked(internal_getpid(), module_name, offsets.data(),
                       offsets.size() * sizeof(offsets[0]));
        VReport(1, " CovDump: %zd PCs written to packed file\n", num_offsets);
      }
    } else {
      // One file per module per process.
      fd_t fd = CovOpenFile(&path, false /* packed */, module_name);
      if (fd == kInvalidFd) continue;
      WriteToFile(fd, offsets.data(), offsets.size() * sizeof(offsets[0]));
      CloseFile(fd);
      VReport(1, " CovDump: %s: %zd PCs written\n", path.data(), num_offsets);
    }
  }
  if (cov_fd != kInvalidFd)
    CloseFile(cov_fd);
}

void CoverageData::DumpAll() {
  if (!coverage_enabled || common_flags()->coverage_direct) return;
  if (atomic_fetch_add(&dump_once_guard, 1, memory_order_relaxed))
    return;
  DumpAsBitSet();
  DumpCounters();
  DumpTrace();
  DumpOffsets();
  DumpCallerCalleePairs();
}

void CovPrepareForSandboxing(__sanitizer_sandbox_arguments *args) {
  if (!args) return;
  if (!coverage_enabled) return;
  cov_sandboxed = args->coverage_sandboxed;
  if (!cov_sandboxed) return;
  cov_max_block_size = args->coverage_max_block_size;
  if (args->coverage_fd >= 0) {
    cov_fd = (fd_t)args->coverage_fd;
  } else {
    InternalScopedString path(kMaxPathLength);
    // Pre-open the file now. The sandbox won't allow us to do it later.
    cov_fd = CovOpenFile(&path, true /* packed */, nullptr);
  }
}

fd_t MaybeOpenCovFile(const char *name) {
  CHECK(name);
  if (!coverage_enabled) return kInvalidFd;
  InternalScopedString path(kMaxPathLength);
  return CovOpenFile(&path, true /* packed */, name);
}

void CovBeforeFork() {
  coverage_data.BeforeFork();
}

void CovAfterFork(int child_pid) {
  coverage_data.AfterFork(child_pid);
}

static void MaybeDumpCoverage() {
  if (common_flags()->coverage)
    __sanitizer_cov_dump();
}

void InitializeCoverage(bool enabled, const char *dir) {
  if (coverage_enabled)
    return;  // May happen if two sanitizer enable coverage in the same process.
  coverage_enabled = enabled;
  coverage_dir = dir;
  coverage_data.Init();
  if (enabled) coverage_data.Enable();
  if (!common_flags()->coverage_direct) Atexit(__sanitizer_cov_dump);
  AddDieCallback(MaybeDumpCoverage);
}

void ReInitializeCoverage(bool enabled, const char *dir) {
  coverage_enabled = enabled;
  coverage_dir = dir;
  coverage_data.ReInit();
}

void CoverageUpdateMapping() {
  if (coverage_enabled)
    CovUpdateMapping(coverage_dir);
}

} // namespace __sanitizer

extern "C" {
SANITIZER_INTERFACE_ATTRIBUTE void __sanitizer_cov(u32 *guard) {
  coverage_data.Add(StackTrace::GetPreviousInstructionPc(GET_CALLER_PC()),
                    guard);
}
SANITIZER_INTERFACE_ATTRIBUTE void __sanitizer_cov_with_check(u32 *guard) {
  atomic_uint32_t *atomic_guard = reinterpret_cast<atomic_uint32_t*>(guard);
  if (static_cast<s32>(
          __sanitizer::atomic_load(atomic_guard, memory_order_relaxed)) < 0)
    __sanitizer_cov(guard);
}
SANITIZER_INTERFACE_ATTRIBUTE void
__sanitizer_cov_indir_call16(uptr callee, uptr callee_cache16[]) {
  coverage_data.IndirCall(StackTrace::GetPreviousInstructionPc(GET_CALLER_PC()),
                          callee, callee_cache16, 16);
}
SANITIZER_INTERFACE_ATTRIBUTE void __sanitizer_cov_init() {
  coverage_enabled = true;
  coverage_dir = common_flags()->coverage_dir;
  coverage_data.Init();
}
SANITIZER_INTERFACE_ATTRIBUTE void __sanitizer_cov_dump() {
  coverage_data.DumpAll();
}
SANITIZER_INTERFACE_ATTRIBUTE void
__sanitizer_cov_module_init(s32 *guards, uptr npcs, u8 *counters,
                            const char *comp_unit_name) {
  coverage_data.InitializeGuards(guards, npcs, comp_unit_name, GET_CALLER_PC());
  coverage_data.InitializeCounters(counters, npcs);
  if (!common_flags()->coverage_direct) return;
  if (SANITIZER_ANDROID && coverage_enabled) {
    // dlopen/dlclose interceptors do not work on Android, so we rely on
    // Extend() calls to update .sancov.map.
    CovUpdateMapping(coverage_dir, GET_CALLER_PC());
  }
  coverage_data.Extend(npcs);
}
SANITIZER_INTERFACE_ATTRIBUTE
sptr __sanitizer_maybe_open_cov_file(const char *name) {
  return (sptr)MaybeOpenCovFile(name);
}
SANITIZER_INTERFACE_ATTRIBUTE
uptr __sanitizer_get_total_unique_coverage() {
  return atomic_load(&coverage_counter, memory_order_relaxed);
}

SANITIZER_INTERFACE_ATTRIBUTE
uptr __sanitizer_get_total_unique_caller_callee_pairs() {
  return atomic_load(&caller_callee_counter, memory_order_relaxed);
}

SANITIZER_INTERFACE_ATTRIBUTE
void __sanitizer_cov_trace_func_enter(u32 *id) {
  __sanitizer_cov_with_check(id);
  coverage_data.TraceBasicBlock(id);
}
SANITIZER_INTERFACE_ATTRIBUTE
void __sanitizer_cov_trace_basic_block(u32 *id) {
  __sanitizer_cov_with_check(id);
  coverage_data.TraceBasicBlock(id);
}
SANITIZER_INTERFACE_ATTRIBUTE
void __sanitizer_reset_coverage() {
  ResetGlobalCounters();
  coverage_data.ReinitializeGuards();
  internal_bzero_aligned16(
      coverage_data.data(),
      RoundUpTo(coverage_data.size() * sizeof(coverage_data.data()[0]), 16));
}
SANITIZER_INTERFACE_ATTRIBUTE
uptr __sanitizer_get_coverage_guards(uptr **data) {
  *data = coverage_data.data();
  return coverage_data.size();
}

SANITIZER_INTERFACE_ATTRIBUTE
uptr __sanitizer_get_coverage_pc_buffer(uptr **data) {
  *data = coverage_data.buffer();
  return __sanitizer_get_total_unique_coverage();
}

SANITIZER_INTERFACE_ATTRIBUTE
uptr __sanitizer_get_number_of_counters() {
  return coverage_data.GetNumberOf8bitCounters();
}

SANITIZER_INTERFACE_ATTRIBUTE
uptr __sanitizer_update_counter_bitset_and_clear_counters(u8 *bitset) {
  return coverage_data.Update8bitCounterBitsetAndClearCounters(bitset);
}
// Default empty implementations (weak). Users should redefine them.
SANITIZER_INTERFACE_ATTRIBUTE SANITIZER_WEAK_ATTRIBUTE
void __sanitizer_cov_trace_cmp() {}
SANITIZER_INTERFACE_ATTRIBUTE SANITIZER_WEAK_ATTRIBUTE
void __sanitizer_cov_trace_switch() {}
} // extern "C"
