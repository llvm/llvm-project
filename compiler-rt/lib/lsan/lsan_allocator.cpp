//=-- lsan_allocator.cpp --------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file is a part of LeakSanitizer.
// See lsan_allocator.h for details.
//
//===----------------------------------------------------------------------===//

#include "lsan_allocator.h"

#include "lsan.h"
#include "lsan_common.h"
#include "sanitizer_common/sanitizer_allocator.h"
#include "sanitizer_common/sanitizer_allocator_checks.h"
#include "sanitizer_common/sanitizer_allocator_interface.h"
#include "sanitizer_common/sanitizer_allocator_report.h"
#include "sanitizer_common/sanitizer_atomic.h"
#include "sanitizer_common/sanitizer_dense_map.h"
#include "sanitizer_common/sanitizer_errno.h"
#include "sanitizer_common/sanitizer_internal_defs.h"
#include "sanitizer_common/sanitizer_placement_new.h"
#include "sanitizer_common/sanitizer_report_decorator.h"
#include "sanitizer_common/sanitizer_stackdepot.h"
#include "sanitizer_common/sanitizer_stacktrace.h"

extern "C" void *memset(void *ptr, int value, uptr num);

namespace __lsan {
#if defined(__i386__) || defined(__arm__)
static const uptr kMaxAllowedMallocSize = 1ULL << 30;
#elif defined(__mips64)
static const uptr kMaxAllowedMallocSize = 4ULL << 30;
#else
static const uptr kMaxAllowedMallocSize = 1ULL << 40;
#endif

static Allocator allocator;

static uptr max_malloc_size;

// Double-free detection keeps two stacks per chunk. Primary-allocator chunks
// keep their metadata mapped after a free, so both stacks live inline in
// ChunkMetadata and the check is a lock-free compare-exchange. Secondary
// chunks are unmapped along with their metadata, so their stacks go into the
// side table below.

namespace {

struct SecondaryFreeRecord {
  u32 alloc_stack_id;
  u32 free_stack_id;
  // Index of this address in secondary_free_fifo, or kNoFifoSlot when the
  // table is unbounded and nothing is ever evicted. Storing it here makes
  // eviction and invalidation O(1).
  u32 fifo_slot;
};

using SecondaryFreeMap = DenseMap<uptr, SecondaryFreeRecord>;

const u32 kNoFifoSlot = ~0u;

// Upper bound for double_free_max_entries, 4M entries. It keeps the worst-case
// side table in the low hundreds of megabytes and turns a mistyped flag into a
// startup warning instead of an out-of-memory death.
const int kMaxSecondaryFreeEntries = 1 << 22;

}  // namespace

// Guards the side table and its FIFO.
static Mutex secondary_free_mutex;

// compiler-rt runtimes must not emit static constructors, so the map is
// placement-new'd into this buffer and never destroyed.
alignas(64) static char secondary_free_placeholder[sizeof(SecondaryFreeMap)];
static SecondaryFreeMap *secondary_free_map;

// Addresses in first-free order. Eviction takes the oldest entry, so a bounded
// table always remembers the most recent frees.
static uptr *secondary_free_fifo;
static u32 secondary_free_fifo_size;
static u32 secondary_free_fifo_head;

void LockDoubleFree() { secondary_free_mutex.Lock(); }

void UnlockDoubleFree() { secondary_free_mutex.Unlock(); }

static void InitializeDoubleFree() {
  if (!flags()->detect_double_free)
    return;
  secondary_free_map = new (secondary_free_placeholder) SecondaryFreeMap();
  int max_entries = flags()->double_free_max_entries;
  if (max_entries < 0) {
    Report(
        "WARNING: LeakSanitizer: double_free_max_entries=%d is negative, "
        "using 0 (unlimited) instead.\n",
        max_entries);
    max_entries = 0;
  } else if (max_entries > kMaxSecondaryFreeEntries) {
    Report(
        "WARNING: LeakSanitizer: double_free_max_entries=%d is too large, "
        "capping it at %d.\n",
        max_entries, kMaxSecondaryFreeEntries);
    max_entries = kMaxSecondaryFreeEntries;
  }
  if (max_entries > 0) {
    secondary_free_fifo_size = static_cast<u32>(max_entries);
    secondary_free_fifo = static_cast<uptr *>(
        MmapOrDie(secondary_free_fifo_size * sizeof(uptr), "DoubleFreeFifo"));
  }
}

static void RemoveSecondaryFreeRecordLocked(uptr chunk) {
  auto *entry = secondary_free_map->find(chunk);
  if (!entry)
    return;
  if (entry->second.fifo_slot != kNoFifoSlot)
    secondary_free_fifo[entry->second.fifo_slot] = 0;
  secondary_free_map->erase(chunk);
}

static void AddSecondaryFreeRecordLocked(uptr chunk, u32 alloc_stack_id,
                                         u32 free_stack_id) {
  // Drop any previous record for this address so that it does not leave a
  // dangling FIFO slot that would later evict the record added here.
  RemoveSecondaryFreeRecordLocked(chunk);

  u32 fifo_slot = kNoFifoSlot;
  if (secondary_free_fifo) {
    fifo_slot = secondary_free_fifo_head;
    if (uptr evicted = secondary_free_fifo[fifo_slot])
      secondary_free_map->erase(evicted);
    secondary_free_fifo[fifo_slot] = chunk;
    if (++secondary_free_fifo_head == secondary_free_fifo_size)
      secondary_free_fifo_head = 0;
  }
  (*secondary_free_map)[chunk] = {alloc_stack_id, free_stack_id, fifo_slot};
}

static bool FindSecondaryFreeRecordLocked(uptr chunk,
                                          SecondaryFreeRecord *record) {
  auto *entry = secondary_free_map->find(chunk);
  if (!entry)
    return false;
  *record = entry->second;
  return true;
}

void InitializeAllocator() {
  SetAllocatorMayReturnNull(common_flags()->allocator_may_return_null);
  allocator.InitLinkerInitialized(
      common_flags()->allocator_release_to_os_interval_ms);
  if (common_flags()->max_allocation_size_mb)
    max_malloc_size = Min(common_flags()->max_allocation_size_mb << 20,
                          kMaxAllowedMallocSize);
  else
    max_malloc_size = kMaxAllowedMallocSize;
  InitializeDoubleFree();
}

void AllocatorThreadStart() { allocator.InitCache(GetAllocatorCache()); }

void AllocatorThreadFinish() {
  allocator.SwallowCache(GetAllocatorCache());
  allocator.DestroyCache(GetAllocatorCache());
}

static ChunkMetadata *Metadata(const void *p) {
  return reinterpret_cast<ChunkMetadata *>(allocator.GetMetaData(p));
}

static atomic_uint8_t *ChunkStateOf(ChunkMetadata *m) {
  return reinterpret_cast<atomic_uint8_t *>(m);
}

namespace {

class Decorator : public __sanitizer::SanitizerCommonDecorator {
 public:
  Decorator() : SanitizerCommonDecorator() {}
  const char *Error() { return Red(); }
};

}  // namespace

static void PrintStackById(const char *label, u32 stack_id) {
  if (!stack_id)
    return;
  Printf("%s", label);
  StackDepotGet(stack_id).Print();
}

// Renders " in thread Tn", or an empty string when the thread has no LSan
// context yet. The wording follows AddressSanitizer's reports.
static const char *ThreadSuffix(char *buffer, uptr size) {
  const u32 tid = GetCurrentThreadId();
  if (tid == static_cast<u32>(kInvalidTid))
    internal_strncpy(buffer, "", size);
  else
    internal_snprintf(buffer, size, " in thread T%u", tid);
  return buffer;
}

// Both reporters hold the error-report lock across Die(), the way ASan's
// ScopedInErrorReport does, so a thread that reaches a report while another is
// printing blocks on the lock instead of appending a truncated second report.
static void NORETURN ReportDoubleFree(uptr addr, u32 alloc_stack_id,
                                      u32 free_stack_id,
                                      const StackTrace *second_free_stack) {
  ScopedErrorReportLock lock;
  char thread[64];
  Decorator d;
  Printf("%s", d.Error());
  Report("ERROR: LeakSanitizer: attempting double-free on %p%s:\n",
         (void *)addr, ThreadSuffix(thread, sizeof(thread)));
  Printf("%s", d.Default());

  Printf("The second free occurred here:\n");
  if (second_free_stack)
    second_free_stack->Print();
  else
    Printf("    <empty stack>\n\n");

  PrintStackById("The first free occurred here:\n", free_stack_id);
  PrintStackById("The memory was allocated here:\n", alloc_stack_id);

  if (second_free_stack)
    ReportErrorSummary("double-free", second_free_stack);
  else
    ReportErrorSummary("double-free");
  Die();
}

static void NORETURN ReportInvalidFree(uptr addr, const StackTrace *stack) {
  ScopedErrorReportLock lock;
  char thread[64];
  Decorator d;
  Printf("%s", d.Error());
  Report(
      "ERROR: LeakSanitizer: attempting free on address which was not "
      "malloc()-ed: %p%s\n",
      (void *)addr, ThreadSuffix(thread, sizeof(thread)));
  Printf("%s", d.Default());

  if (stack) {
    stack->Print();
    ReportErrorSummary("bad-free", stack);
  } else {
    ReportErrorSummary("bad-free");
  }
  Die();
}

// Must be called with secondary_free_mutex released, since symbolizing can
// re-enter the allocator.
static void NORETURN ReportNonLiveFree(uptr chunk,
                                       const SecondaryFreeRecord &record,
                                       bool remembered,
                                       const StackTrace *stack) {
  if (remembered)
    ReportDoubleFree(chunk, record.alloc_stack_id, record.free_stack_id, stack);
  ReportInvalidFree(chunk, stack);
}

static void RegisterAllocation(const StackTrace &stack, void *p, uptr size) {
  if (!p) return;
  ChunkMetadata *m = Metadata(p);
  CHECK(m);
  if (flags()->detect_double_free && !allocator.FromPrimary(p)) {
    // This address may still be recorded from an earlier large allocation that
    // was freed and unmapped; the new allocation supersedes it.
    Lock l(&secondary_free_mutex);
    RemoveSecondaryFreeRecordLocked(reinterpret_cast<uptr>(p));
  }
  m->tag = DisabledInThisThread() ? kIgnored : kDirectlyLeaked;
  m->stack_trace_id = StackDepotPut(stack);
  m->free_stack_id = 0;
  m->requested_size = size;
  // These fields are only acquired by the double-free checks, which run when
  // detect_double_free is on. Paying for a release store only then avoids a
  // barrier on every malloc() elsewhere.
  atomic_store(ChunkStateOf(m), kChunkAllocated,
               flags()->detect_double_free ? memory_order_release
                                           : memory_order_relaxed);
  RunMallocHooks(p, size);
}

static void DeallocateCheckedPrimary(void *p, const StackTrace *free_stack) {
  const uptr chunk = reinterpret_cast<uptr>(p);
  ChunkMetadata *m = nullptr;
  if (LIKELY(allocator.GetBlockBegin(p) == p)) {
    m = Metadata(p);
    CHECK(m);
  }
  // The check above rejects an interior pointer and an address the primary
  // never mapped, but a mapped slot that was never handed out passes it. Such
  // a slot has no requested size, and freeing it is an invalid free, not a
  // double free.
  if (UNLIKELY(!m || m->requested_size == 0))
    ReportInvalidFree(chunk, free_stack);

  u8 expected = kChunkAllocated;
  if (UNLIKELY(!atomic_compare_exchange_strong(
          ChunkStateOf(m), &expected, kChunkFreeing, memory_order_acq_rel))) {
    // Lost the race, or the chunk was already free. The compare-exchange has
    // acquire semantics, so a kChunkFree state means free_stack_id is visible.
    // kChunkFreeing means the winner has not published it yet.
    ReportDoubleFree(chunk, m->stack_trace_id,
                     expected == kChunkFree ? m->free_stack_id : 0, free_stack);
  }
  m->free_stack_id = free_stack ? StackDepotPut(*free_stack) : 0;
  atomic_store(ChunkStateOf(m), kChunkFree, memory_order_release);
  allocator.Deallocate(GetAllocatorCache(), p);
}

// Releasing a secondary chunk unmaps its metadata, so the stacks are copied
// into the side table first and the whole sequence holds secondary_free_mutex,
// or a concurrent free could unmap it mid-check.
static void DeallocateCheckedSecondary(void *p, const StackTrace *free_stack) {
  const uptr chunk = reinterpret_cast<uptr>(p);
  SecondaryFreeRecord record = {};
  bool remembered = false;
  {
    Lock l(&secondary_free_mutex);
    if (LIKELY(allocator.GetBlockBegin(p) == p)) {
      ChunkMetadata *m = Metadata(p);
      CHECK(m);
      const u32 free_stack_id = free_stack ? StackDepotPut(*free_stack) : 0;
      AddSecondaryFreeRecordLocked(chunk, m->stack_trace_id, free_stack_id);
      atomic_store(ChunkStateOf(m), kChunkFree, memory_order_relaxed);
      allocator.Deallocate(GetAllocatorCache(), p);
      return;
    }
    // A large chunk is unmapped as soon as it is freed, so a second free can
    // only be recognized from the side table.
    remembered = FindSecondaryFreeRecordLocked(chunk, &record);
  }
  ReportNonLiveFree(chunk, record, remembered, free_stack);
}

static void *ReportAllocationSizeTooBig(uptr size, const StackTrace &stack) {
  if (AllocatorMayReturnNull()) {
    Report("WARNING: LeakSanitizer failed to allocate 0x%zx bytes\n", size);
    return nullptr;
  }
  ReportAllocationSizeTooBig(size, max_malloc_size, &stack);
}

void *Allocate(const StackTrace &stack, uptr size, uptr alignment,
               bool cleared) {
  if (size == 0)
    size = 1;
  if (size > max_malloc_size)
    return ReportAllocationSizeTooBig(size, stack);
  if (UNLIKELY(IsRssLimitExceeded())) {
    if (AllocatorMayReturnNull())
      return nullptr;
    ReportRssLimitExceeded(&stack);
  }
  void *p = allocator.Allocate(GetAllocatorCache(), size, alignment);
  if (UNLIKELY(!p)) {
    SetAllocatorOutOfMemory();
    if (AllocatorMayReturnNull())
      return nullptr;
    ReportOutOfMemory(size, &stack);
  }
  // Do not rely on the allocator to clear the memory (it's slow).
  if (cleared && allocator.FromPrimary(p))
    memset(p, 0, size);
  RegisterAllocation(stack, p, size);
  return p;
}

static void *Calloc(uptr nmemb, uptr size, const StackTrace &stack) {
  if (UNLIKELY(CheckForCallocOverflow(size, nmemb))) {
    if (AllocatorMayReturnNull())
      return nullptr;
    ReportCallocOverflow(nmemb, size, &stack);
  }
  size *= nmemb;
  return Allocate(stack, size, 1, true);
}

static void DeallocateUnchecked(void *p) {
  ChunkMetadata *m = Metadata(p);
  CHECK(m);
  atomic_store(ChunkStateOf(m), kChunkFree, memory_order_relaxed);
  allocator.Deallocate(GetAllocatorCache(), p);
}

void Deallocate(void *p, const StackTrace *free_stack) {
  if (!p) return;
  // Run the hooks before the chunk state changes, so a hook still observes p as
  // owned. AddressSanitizer runs them at the same point, before it validates p.
  RunFreeHooks(p);
  if (flags()->detect_double_free) {
    if (allocator.FromPrimary(p))
      DeallocateCheckedPrimary(p, free_stack);
    else
      DeallocateCheckedSecondary(p, free_stack);
    return;
  }
  DeallocateUnchecked(p);
}

// Returns the requested size of the live chunk starting at `p`, and reports
// without returning if `p` is not one. A secondary chunk is unmapped as soon as
// it is freed, so this has to run before the old chunk's metadata is read.
static uptr CheckedReallocSourceSize(void *p, const StackTrace *stack) {
  const uptr chunk = reinterpret_cast<uptr>(p);
  if (allocator.FromPrimary(p)) {
    if (LIKELY(allocator.GetBlockBegin(p) == p)) {
      ChunkMetadata *m = Metadata(p);
      CHECK(m);
      const u8 state = atomic_load(ChunkStateOf(m), memory_order_acquire);
      if (LIKELY(state == kChunkAllocated && m->requested_size != 0))
        return m->requested_size;
      if (m->requested_size != 0)
        ReportDoubleFree(chunk, m->stack_trace_id,
                         state == kChunkFree ? m->free_stack_id : 0, stack);
    }
    ReportInvalidFree(chunk, stack);
  }

  SecondaryFreeRecord record = {};
  bool remembered = false;
  {
    Lock l(&secondary_free_mutex);
    if (LIKELY(allocator.GetBlockBegin(p) == p)) {
      ChunkMetadata *m = Metadata(p);
      CHECK(m);
      return m->requested_size;
    }
    remembered = FindSecondaryFreeRecordLocked(chunk, &record);
  }
  ReportNonLiveFree(chunk, record, remembered, stack);
}

void *Reallocate(const StackTrace &stack, void *p, uptr new_size,
                 uptr alignment) {
  if (new_size > max_malloc_size) {
    ReportAllocationSizeTooBig(new_size, stack);
    return nullptr;
  }
  if (!p)
    return Allocate(stack, new_size, alignment, false);
  if (!new_size) {
    Deallocate(p, &stack);
    return nullptr;
  }
  uptr old_size;
  if (flags()->detect_double_free) {
    old_size = CheckedReallocSourceSize(p, &stack);
  } else {
    ChunkMetadata *m = Metadata(p);
    CHECK(m);
    old_size = m->requested_size;
  }
  // Allocate the replacement first. If it fails, p must be left completely
  // untouched: it is still owned by the caller, so its metadata must keep the
  // original size and allocation stack, and no free hook may be reported.
  void *new_p = Allocate(stack, new_size, alignment, false);
  if (!new_p)
    return nullptr;
  internal_memcpy(new_p, p, Min(new_size, old_size));
  Deallocate(p, &stack);
  return new_p;
}

void GetAllocatorCacheRange(uptr *begin, uptr *end) {
  *begin = (uptr)GetAllocatorCache();
  *end = *begin + sizeof(AllocatorCache);
}

static const void *GetMallocBegin(const void *p) {
  if (!p)
    return nullptr;
  void *beg = allocator.GetBlockBegin(p);
  if (!beg)
    return nullptr;
  ChunkMetadata *m = Metadata(beg);
  if (!m)
    return nullptr;
  if (!m->allocated)
    return nullptr;
  if (m->requested_size == 0)
    return nullptr;
  return (const void *)beg;
}

uptr GetMallocUsableSize(const void *p) {
  if (!p)
    return 0;
  ChunkMetadata *m = Metadata(p);
  if (!m) return 0;
  return m->requested_size;
}

uptr GetMallocUsableSizeFast(const void *p) {
  return Metadata(p)->requested_size;
}

int lsan_posix_memalign(void **memptr, uptr alignment, uptr size,
                        const StackTrace &stack) {
  if (UNLIKELY(!CheckPosixMemalignAlignment(alignment))) {
    if (AllocatorMayReturnNull())
      return errno_EINVAL;
    ReportInvalidPosixMemalignAlignment(alignment, &stack);
  }
  void *ptr = Allocate(stack, size, alignment, kAlwaysClearMemory);
  if (UNLIKELY(!ptr))
    // OOM error is already taken care of by Allocate.
    return errno_ENOMEM;
  CHECK(IsAligned((uptr)ptr, alignment));
  *memptr = ptr;
  return 0;
}

void *lsan_aligned_alloc(uptr alignment, uptr size, const StackTrace &stack) {
  if (UNLIKELY(!CheckAlignedAllocAlignmentAndSize(alignment, size))) {
    errno = errno_EINVAL;
    if (AllocatorMayReturnNull())
      return nullptr;
    ReportInvalidAlignedAllocAlignment(size, alignment, &stack);
  }
  return SetErrnoOnNull(Allocate(stack, size, alignment, kAlwaysClearMemory));
}

void *lsan_memalign(uptr alignment, uptr size, const StackTrace &stack) {
  if (UNLIKELY(!IsPowerOfTwo(alignment))) {
    errno = errno_EINVAL;
    if (AllocatorMayReturnNull())
      return nullptr;
    ReportInvalidAllocationAlignment(alignment, &stack);
  }
  return SetErrnoOnNull(Allocate(stack, size, alignment, kAlwaysClearMemory));
}

void *lsan_malloc(uptr size, const StackTrace &stack) {
  return SetErrnoOnNull(Allocate(stack, size, 1, kAlwaysClearMemory));
}

void lsan_free(void *p, const StackTrace *free_stack) {
  Deallocate(p, free_stack);
}

void lsan_free_sized(void *p, uptr, const StackTrace *free_stack) {
  Deallocate(p, free_stack);
}

void lsan_free_aligned_sized(void *p, uptr, uptr,
                             const StackTrace *free_stack) {
  Deallocate(p, free_stack);
}

void NOINLINE lsan_free_with_stack(void *p, uptr pc, uptr bp) {
  GET_STACK_TRACE_FREE_AT(pc, bp);
  Deallocate(p, &stack);
}

void NOINLINE lsan_free_sized_with_stack(void *p, uptr, uptr pc, uptr bp) {
  GET_STACK_TRACE_FREE_AT(pc, bp);
  Deallocate(p, &stack);
}

void NOINLINE lsan_free_aligned_sized_with_stack(void *p, uptr, uptr, uptr pc,
                                                 uptr bp) {
  GET_STACK_TRACE_FREE_AT(pc, bp);
  Deallocate(p, &stack);
}

void *lsan_realloc(void *p, uptr size, const StackTrace &stack) {
  return SetErrnoOnNull(Reallocate(stack, p, size, 1));
}

void *lsan_reallocarray(void *ptr, uptr nmemb, uptr size,
                        const StackTrace &stack) {
  if (UNLIKELY(CheckForCallocOverflow(size, nmemb))) {
    errno = errno_ENOMEM;
    if (AllocatorMayReturnNull())
      return nullptr;
    ReportReallocArrayOverflow(nmemb, size, &stack);
  }
  return lsan_realloc(ptr, nmemb * size, stack);
}

void *lsan_calloc(uptr nmemb, uptr size, const StackTrace &stack) {
  return SetErrnoOnNull(Calloc(nmemb, size, stack));
}

void *lsan_valloc(uptr size, const StackTrace &stack) {
  return SetErrnoOnNull(
      Allocate(stack, size, GetPageSizeCached(), kAlwaysClearMemory));
}

void *lsan_pvalloc(uptr size, const StackTrace &stack) {
  uptr PageSize = GetPageSizeCached();
  if (UNLIKELY(CheckForPvallocOverflow(size, PageSize))) {
    errno = errno_ENOMEM;
    if (AllocatorMayReturnNull())
      return nullptr;
    ReportPvallocOverflow(size, &stack);
  }
  // pvalloc(0) should allocate one page.
  size = size ? RoundUpTo(size, PageSize) : PageSize;
  return SetErrnoOnNull(Allocate(stack, size, PageSize, kAlwaysClearMemory));
}

uptr lsan_mz_size(const void *p) {
  return GetMallocUsableSize(p);
}

///// Interface to the common LSan module. /////

void LockAllocator() {
  allocator.ForceLock();
}

void UnlockAllocator() {
  allocator.ForceUnlock();
}

void GetAllocatorGlobalRange(uptr *begin, uptr *end) {
  *begin = (uptr)&allocator;
  *end = *begin + sizeof(allocator);
}

uptr PointsIntoChunk(void* p) {
  uptr addr = reinterpret_cast<uptr>(p);
  uptr chunk = reinterpret_cast<uptr>(allocator.GetBlockBeginFastLocked(p));
  if (!chunk) return 0;
  // LargeMmapAllocator considers pointers to the meta-region of a chunk to be
  // valid, but we don't want that.
  if (addr < chunk) return 0;
  ChunkMetadata *m = Metadata(reinterpret_cast<void *>(chunk));
  CHECK(m);
  if (!m->allocated)
    return 0;
  if (addr < chunk + m->requested_size)
    return chunk;
  if (IsSpecialCaseOfOperatorNew0(chunk, m->requested_size, addr))
    return chunk;
  return 0;
}

uptr GetUserBegin(uptr chunk) {
  return chunk;
}

uptr GetUserAddr(uptr chunk) {
  return chunk;
}

LsanMetadata::LsanMetadata(uptr chunk) {
  metadata_ = Metadata(reinterpret_cast<void *>(chunk));
  CHECK(metadata_);
}

bool LsanMetadata::allocated() const {
  return reinterpret_cast<ChunkMetadata *>(metadata_)->allocated;
}

ChunkTag LsanMetadata::tag() const {
  return reinterpret_cast<ChunkMetadata *>(metadata_)->tag;
}

void LsanMetadata::set_tag(ChunkTag value) {
  reinterpret_cast<ChunkMetadata *>(metadata_)->tag = value;
}

uptr LsanMetadata::requested_size() const {
  return reinterpret_cast<ChunkMetadata *>(metadata_)->requested_size;
}

u32 LsanMetadata::stack_trace_id() const {
  return reinterpret_cast<ChunkMetadata *>(metadata_)->stack_trace_id;
}

void ForEachChunk(ForEachChunkCallback callback, void *arg) {
  allocator.ForEachChunk(callback, arg);
}

IgnoreObjectResult IgnoreObject(const void *p) {
  void *chunk = allocator.GetBlockBegin(p);
  if (!chunk || p < chunk) return kIgnoreObjectInvalid;
  ChunkMetadata *m = Metadata(chunk);
  CHECK(m);
  // Storing the tag is a read-modify-write over the chunk state byte (see
  // ChunkMetadata), so racing this against a free() of the same chunk can undo
  // that free's state change. Ignoring an object another thread is freeing is a
  // use-after-free to begin with, and the check does not make it safe: it can
  // miss or misreport the double free that follows.
  if (m->allocated && (uptr)p < (uptr)chunk + m->requested_size) {
    if (m->tag == kIgnored)
      return kIgnoreObjectAlreadyIgnored;
    m->tag = kIgnored;
    return kIgnoreObjectSuccess;
  } else {
    return kIgnoreObjectInvalid;
  }
}

} // namespace __lsan

using namespace __lsan;

extern "C" {
SANITIZER_INTERFACE_ATTRIBUTE
uptr __sanitizer_get_current_allocated_bytes() {
  uptr stats[AllocatorStatCount];
  allocator.GetStats(stats);
  return stats[AllocatorStatAllocated];
}

SANITIZER_INTERFACE_ATTRIBUTE
uptr __sanitizer_get_heap_size() {
  uptr stats[AllocatorStatCount];
  allocator.GetStats(stats);
  return stats[AllocatorStatMapped];
}

SANITIZER_INTERFACE_ATTRIBUTE
uptr __sanitizer_get_free_bytes() { return 1; }

SANITIZER_INTERFACE_ATTRIBUTE
uptr __sanitizer_get_unmapped_bytes() { return 0; }

SANITIZER_INTERFACE_ATTRIBUTE
uptr __sanitizer_get_estimated_allocated_size(uptr size) { return size; }

SANITIZER_INTERFACE_ATTRIBUTE
int __sanitizer_get_ownership(const void *p) {
  return GetMallocBegin(p) != nullptr;
}

SANITIZER_INTERFACE_ATTRIBUTE
const void * __sanitizer_get_allocated_begin(const void *p) {
  return GetMallocBegin(p);
}

SANITIZER_INTERFACE_ATTRIBUTE
uptr __sanitizer_get_allocated_size(const void *p) {
  return GetMallocUsableSize(p);
}

SANITIZER_INTERFACE_ATTRIBUTE
uptr __sanitizer_get_allocated_size_fast(const void *p) {
  DCHECK_EQ(p, __sanitizer_get_allocated_begin(p));
  uptr ret = GetMallocUsableSizeFast(p);
  DCHECK_EQ(ret, __sanitizer_get_allocated_size(p));
  return ret;
}

SANITIZER_INTERFACE_ATTRIBUTE
void __sanitizer_purge_allocator() { allocator.ForceReleaseToOS(); }

} // extern "C"
