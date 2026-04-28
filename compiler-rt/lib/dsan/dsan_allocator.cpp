//=-- dsan_allocator.cpp --------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file is a part of DoubleFreeSanitizer.
// See dsan_allocator.h for details.
//
//===----------------------------------------------------------------------===//

#include "dsan_allocator.h"

#include "sanitizer_common/sanitizer_allocator.h"
#include "sanitizer_common/sanitizer_allocator_checks.h"
#include "sanitizer_common/sanitizer_allocator_interface.h"
#include "sanitizer_common/sanitizer_allocator_report.h"
#include "sanitizer_common/sanitizer_atomic.h"
#include "sanitizer_common/sanitizer_errno.h"
#include "sanitizer_common/sanitizer_internal_defs.h"
#include "sanitizer_common/sanitizer_report_decorator.h"
#include "sanitizer_common/sanitizer_stackdepot.h"
#include "sanitizer_common/sanitizer_stacktrace.h"

extern "C" void* memset(void* ptr, int value, uptr num);

namespace __dsan {
#if defined(__i386__) || defined(__arm__)
static const uptr kMaxAllowedMallocSize = 1ULL << 30;
#elif defined(__mips64) || defined(__aarch64__)
static const uptr kMaxAllowedMallocSize = 4ULL << 30;
#else
static const uptr kMaxAllowedMallocSize = 1ULL << 40;
#endif

static Allocator allocator;

static uptr max_malloc_size;

struct SecondaryTombstone {
  void* ptr;
  u32 alloc_stack_id;
  u32 free_stack_id;
};

static constexpr uptr kSecondaryTombstoneLimit = 65536;
static Mutex secondary_tombstones_mutex;
static InternalMmapVector<SecondaryTombstone> secondary_tombstones;
static uptr next_secondary_tombstone;

static constexpr uptr kPrimaryQuarantineSize = 1024;
static Mutex primary_quarantine_mutex;
static InternalMmapVector<void*> primary_quarantine;
static uptr next_primary_quarantine;

static bool FindSecondaryTombstone(void* p, SecondaryTombstone* result) {
  Lock lock(&secondary_tombstones_mutex);
  for (uptr i = 0; i != secondary_tombstones.size(); ++i) {
    if (secondary_tombstones[i].ptr == p) {
      *result = secondary_tombstones[i];
      return true;
    }
  }
  return false;
}

static void AddSecondaryTombstone(void* p, const ChunkMetadata* m) {
  Lock lock(&secondary_tombstones_mutex);
  const SecondaryTombstone tombstone = {p, m->alloc_stack_id, m->free_stack_id};
  if (secondary_tombstones.size() < kSecondaryTombstoneLimit) {
    secondary_tombstones.push_back(tombstone);
    return;
  }
  secondary_tombstones[next_secondary_tombstone] = tombstone;
  next_secondary_tombstone =
      (next_secondary_tombstone + 1) % kSecondaryTombstoneLimit;
}

static void RemoveSecondaryTombstone(void* p) {
  Lock lock(&secondary_tombstones_mutex);
  for (uptr i = 0; i != secondary_tombstones.size(); ++i) {
    if (secondary_tombstones[i].ptr != p)
      continue;
    secondary_tombstones[i] = secondary_tombstones.back();
    secondary_tombstones.pop_back();
    return;
  }
}

static void QuarantinePrimary(void* p) {
  void* released = nullptr;
  {
    Lock lock(&primary_quarantine_mutex);
    if (primary_quarantine.size() < kPrimaryQuarantineSize) {
      primary_quarantine.push_back(p);
    } else {
      released = primary_quarantine[next_primary_quarantine];
      primary_quarantine[next_primary_quarantine] = p;
      next_primary_quarantine =
          (next_primary_quarantine + 1) % kPrimaryQuarantineSize;
    }
  }
  if (released)
    allocator.Deallocate(GetAllocatorCache(), released);
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
}

void AllocatorThreadStart() { allocator.InitCache(GetAllocatorCache()); }

void AllocatorThreadFinish() {
  allocator.SwallowCache(GetAllocatorCache());
  allocator.DestroyCache(GetAllocatorCache());
}

static ChunkMetadata* Metadata(const void* p) {
  return reinterpret_cast<ChunkMetadata*>(allocator.GetMetaData(p));
}

static void RegisterAllocation(const StackTrace& stack, void* p, uptr size) {
  if (!p)
    return;
  if (!allocator.FromPrimary(p))
    RemoveSecondaryTombstone(p);
  ChunkMetadata* m = Metadata(p);
  CHECK(m);
  m->alloc_stack_id = StackDepotPut(stack);
  m->free_stack_id = 0;
  m->requested_size = size;
  atomic_store(reinterpret_cast<atomic_uint8_t*>(m), kChunkAllocated,
               memory_order_release);
  RunMallocHooks(p, size);
}

// Report double-free and terminate.
static void NORETURN ReportDoubleFree(void* p, u32 alloc_stack_id,
                                      u32 first_free_stack_id,
                                      const StackTrace& free_stack) {
  class Decorator : public __sanitizer::SanitizerCommonDecorator {
   public:
    Decorator() : SanitizerCommonDecorator() {}
    const char* Error() { return Red(); }
    const char* Info() { return Blue(); }
  };

  Decorator d;
  Printf("\n");
  Printf("%s", d.Error());
  Report("ERROR: DoubleFreeSanitizer: double-free on address %p\n", p);
  Printf("%s", d.Default());

  // Print the second free (current) backtrace.
  Printf("\n");
  Printf("%s", d.Info());
  Printf("Second free (the invalid free) of address %p:\n", p);
  Printf("%s", d.Default());
  free_stack.Print();

  // Print the first free backtrace.
  if (first_free_stack_id) {
    Printf("\n");
    Printf("%s", d.Info());
    Printf("First free of address %p:\n", p);
    Printf("%s", d.Default());
    StackDepotGet(first_free_stack_id).Print();
  }

  // Print the original allocation backtrace.
  if (alloc_stack_id) {
    Printf("\n");
    Printf("%s", d.Info());
    Printf("Original allocation of address %p:\n", p);
    Printf("%s", d.Default());
    StackDepotGet(alloc_stack_id).Print();
  }

  Printf("\n");
  Printf("SUMMARY: DoubleFreeSanitizer: double-free on address %p\n", p);
  Die();
}

static void NORETURN ReportInvalidFree(void* p, const StackTrace& stack) {
  Report("ERROR: DoubleFreeSanitizer: invalid free on address %p\n", p);
  stack.Print();
  Die();
}

static ChunkMetadata* GetChunkMetadata(void* p, const StackTrace& stack) {
  if (!p)
    return nullptr;
  if (!allocator.PointerIsMine(p)) {
    SecondaryTombstone tombstone = {};
    if (FindSecondaryTombstone(p, &tombstone))
      ReportDoubleFree(p, tombstone.alloc_stack_id, tombstone.free_stack_id,
                       stack);
    ReportInvalidFree(p, stack);
  }
  if (allocator.GetBlockBegin(p) != p)
    ReportInvalidFree(p, stack);
  ChunkMetadata* m = Metadata(p);
  if (atomic_load(reinterpret_cast<atomic_uint8_t*>(m), memory_order_acquire) ==
      kChunkInvalid)
    ReportInvalidFree(p, stack);
  return m;
}

static void RegisterDeallocation(const StackTrace& stack, void* p) {
  ChunkMetadata* m = GetChunkMetadata(p, stack);
  if (!m)
    return;

  u8 expected = kChunkAllocated;
  if (!atomic_compare_exchange_strong(reinterpret_cast<atomic_uint8_t*>(m),
                                      &expected, kChunkFreeing,
                                      memory_order_acquire)) {
    while (expected == kChunkFreeing)
      expected = atomic_load(reinterpret_cast<atomic_uint8_t*>(m),
                             memory_order_acquire);
    if (expected == kChunkFreed)
      ReportDoubleFree(p, m->alloc_stack_id, m->free_stack_id, stack);
    ReportInvalidFree(p, stack);
  }

  m->free_stack_id = StackDepotPut(stack);
  atomic_store(reinterpret_cast<atomic_uint8_t*>(m), kChunkFreed,
               memory_order_release);
  RunFreeHooks(p);
}

static void* ReportAllocationSizeTooBig(uptr size, const StackTrace& stack) {
  if (AllocatorMayReturnNull()) {
    Report("WARNING: DoubleFreeSanitizer failed to allocate 0x%zx bytes\n",
           size);
    return nullptr;
  }
  ReportAllocationSizeTooBig(size, max_malloc_size, &stack);
}

void* Allocate(const StackTrace& stack, uptr size, uptr alignment,
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
  void* p = allocator.Allocate(GetAllocatorCache(), size, alignment);
  if (UNLIKELY(!p)) {
    SetAllocatorOutOfMemory();
    if (AllocatorMayReturnNull())
      return nullptr;
    ReportOutOfMemory(size, &stack);
  }
  if (cleared && allocator.FromPrimary(p))
    memset(p, 0, size);
  RegisterAllocation(stack, p, size);
  return p;
}

static void* Calloc(uptr nmemb, uptr size, const StackTrace& stack) {
  if (UNLIKELY(CheckForCallocOverflow(size, nmemb))) {
    if (AllocatorMayReturnNull())
      return nullptr;
    ReportCallocOverflow(nmemb, size, &stack);
  }
  size *= nmemb;
  return Allocate(stack, size, 1, true);
}

void Deallocate(const StackTrace& stack, void* p) {
  if (!p)
    return;
  RegisterDeallocation(stack, p);
  if (allocator.FromPrimary(p)) {
    QuarantinePrimary(p);
  } else {
    AddSecondaryTombstone(p, Metadata(p));
    allocator.Deallocate(GetAllocatorCache(), p);
  }
}

void* Reallocate(const StackTrace& stack, void* p, uptr new_size,
                 uptr alignment) {
  if (!p)
    return Allocate(stack, new_size, alignment, kAlwaysClearMemory);
  if (!new_size) {
    Deallocate(stack, p);
    return nullptr;
  }
  if (new_size > max_malloc_size) {
    ReportAllocationSizeTooBig(new_size, stack);
    return nullptr;
  }
  ChunkMetadata* m = GetChunkMetadata(p, stack);
  if (atomic_load(reinterpret_cast<atomic_uint8_t*>(m), memory_order_acquire) !=
      kChunkAllocated)
    ReportDoubleFree(p, m->alloc_stack_id, m->free_stack_id, stack);
  const uptr old_size = m->requested_size;
  void* new_p = Allocate(stack, new_size, alignment, kAlwaysClearMemory);
  if (!new_p)
    return nullptr;
  internal_memcpy(new_p, p, Min(old_size, new_size));
  Deallocate(stack, p);
  return new_p;
}

void GetAllocatorCacheRange(uptr* begin, uptr* end) {
  *begin = (uptr)GetAllocatorCache();
  *end = *begin + sizeof(AllocatorCache);
}

static const void* GetMallocBegin(const void* p) {
  if (!p)
    return nullptr;
  void* beg = allocator.GetBlockBegin(p);
  if (!beg)
    return nullptr;
  ChunkMetadata* m = Metadata(beg);
  if (!m)
    return nullptr;
  if (atomic_load(reinterpret_cast<atomic_uint8_t*>(m), memory_order_acquire) !=
      kChunkAllocated)
    return nullptr;
  if (m->requested_size == 0)
    return nullptr;
  return (const void*)beg;
}

uptr GetMallocUsableSize(const void* p) {
  if (!p)
    return 0;
  ChunkMetadata* m = Metadata(p);
  if (!m)
    return 0;
  return m->requested_size;
}

uptr GetMallocUsableSizeFast(const void* p) {
  return Metadata(p)->requested_size;
}

int dsan_posix_memalign(void** memptr, uptr alignment, uptr size,
                        const StackTrace& stack) {
  if (UNLIKELY(!CheckPosixMemalignAlignment(alignment))) {
    if (AllocatorMayReturnNull())
      return errno_EINVAL;
    ReportInvalidPosixMemalignAlignment(alignment, &stack);
  }
  void* ptr = Allocate(stack, size, alignment, kAlwaysClearMemory);
  if (UNLIKELY(!ptr))
    return errno_ENOMEM;
  CHECK(IsAligned((uptr)ptr, alignment));
  *memptr = ptr;
  return 0;
}

void* dsan_aligned_alloc(uptr alignment, uptr size, const StackTrace& stack) {
  if (UNLIKELY(!CheckAlignedAllocAlignmentAndSize(alignment, size))) {
    errno = errno_EINVAL;
    if (AllocatorMayReturnNull())
      return nullptr;
    ReportInvalidAlignedAllocAlignment(size, alignment, &stack);
  }
  return SetErrnoOnNull(Allocate(stack, size, alignment, kAlwaysClearMemory));
}

void* dsan_memalign(uptr alignment, uptr size, const StackTrace& stack) {
  if (UNLIKELY(!IsPowerOfTwo(alignment))) {
    errno = errno_EINVAL;
    if (AllocatorMayReturnNull())
      return nullptr;
    ReportInvalidAllocationAlignment(alignment, &stack);
  }
  return SetErrnoOnNull(Allocate(stack, size, alignment, kAlwaysClearMemory));
}

void* dsan_malloc(uptr size, const StackTrace& stack) {
  return SetErrnoOnNull(Allocate(stack, size, 1, kAlwaysClearMemory));
}

void dsan_free(void* p, const StackTrace& stack) { Deallocate(stack, p); }

void dsan_free_sized(void* p, uptr, const StackTrace& stack) {
  Deallocate(stack, p);
}

void dsan_free_aligned_sized(void* p, uptr, uptr, const StackTrace& stack) {
  Deallocate(stack, p);
}

void* dsan_realloc(void* p, uptr size, const StackTrace& stack) {
  return SetErrnoOnNull(Reallocate(stack, p, size, 1));
}

void* dsan_reallocarray(void* ptr, uptr nmemb, uptr size,
                        const StackTrace& stack) {
  if (UNLIKELY(CheckForCallocOverflow(size, nmemb))) {
    errno = errno_ENOMEM;
    if (AllocatorMayReturnNull())
      return nullptr;
    ReportReallocArrayOverflow(nmemb, size, &stack);
  }
  return dsan_realloc(ptr, nmemb * size, stack);
}

void* dsan_calloc(uptr nmemb, uptr size, const StackTrace& stack) {
  return SetErrnoOnNull(Calloc(nmemb, size, stack));
}

void* dsan_valloc(uptr size, const StackTrace& stack) {
  return SetErrnoOnNull(
      Allocate(stack, size, GetPageSizeCached(), kAlwaysClearMemory));
}

void* dsan_pvalloc(uptr size, const StackTrace& stack) {
  uptr PageSize = GetPageSizeCached();
  if (UNLIKELY(CheckForPvallocOverflow(size, PageSize))) {
    errno = errno_ENOMEM;
    if (AllocatorMayReturnNull())
      return nullptr;
    ReportPvallocOverflow(size, &stack);
  }
  size = size ? RoundUpTo(size, PageSize) : PageSize;
  return SetErrnoOnNull(Allocate(stack, size, PageSize, kAlwaysClearMemory));
}

uptr dsan_mz_size(const void* p) { return GetMallocUsableSize(p); }

void LockAllocator() { allocator.ForceLock(); }

void UnlockAllocator() { allocator.ForceUnlock(); }

}  // namespace __dsan

using namespace __dsan;

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
int __sanitizer_get_ownership(const void* p) {
  return GetMallocBegin(p) != nullptr;
}

SANITIZER_INTERFACE_ATTRIBUTE
const void* __sanitizer_get_allocated_begin(const void* p) {
  return GetMallocBegin(p);
}

SANITIZER_INTERFACE_ATTRIBUTE
uptr __sanitizer_get_allocated_size(const void* p) {
  return GetMallocUsableSize(p);
}

SANITIZER_INTERFACE_ATTRIBUTE
uptr __sanitizer_get_allocated_size_fast(const void* p) {
  DCHECK_EQ(p, __sanitizer_get_allocated_begin(p));
  uptr ret = GetMallocUsableSizeFast(p);
  DCHECK_EQ(ret, __sanitizer_get_allocated_size(p));
  return ret;
}

SANITIZER_INTERFACE_ATTRIBUTE
void __sanitizer_purge_allocator() { allocator.ForceReleaseToOS(); }

}  // extern "C"
