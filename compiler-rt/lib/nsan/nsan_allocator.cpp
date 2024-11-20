//===- nsan_allocator.cpp -------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// NumericalStabilitySanitizer allocator.
//
//===----------------------------------------------------------------------===//

#include "nsan_allocator.h"
#include "interception/interception.h"
#include "nsan.h"
#include "nsan_flags.h"
#include "nsan_platform.h"
#include "nsan_thread.h"
#include "sanitizer_common/sanitizer_allocator.h"
#include "sanitizer_common/sanitizer_allocator_checks.h"
#include "sanitizer_common/sanitizer_allocator_interface.h"
#include "sanitizer_common/sanitizer_allocator_report.h"
#include "sanitizer_common/sanitizer_common.h"
#include "sanitizer_common/sanitizer_errno.h"

using namespace __nsan;

DECLARE_REAL(void *, memcpy, void *dest, const void *src, uptr n)
DECLARE_REAL(void *, memset, void *dest, int c, uptr n)

namespace {
struct Metadata {
  uptr requested_size;
};

struct NsanMapUnmapCallback {
  void OnMap(uptr p, uptr size) const {}
  void OnMapSecondary(uptr p, uptr size, uptr user_begin,
                      uptr user_size) const {}
  void OnUnmap(uptr p, uptr size) const {}
};

const uptr kMaxAllowedMallocSize = 1ULL << 40;

// Allocator64 parameters. Deliberately using a short name.
struct AP64 {
  static const uptr kSpaceBeg = Mapping::kHeapMemBeg;
  static const uptr kSpaceSize = 0x40000000000; // 4T.
  static const uptr kMetadataSize = sizeof(Metadata);
  using SizeClassMap = DefaultSizeClassMap;
  using MapUnmapCallback = NsanMapUnmapCallback;
  static const uptr kFlags = 0;
  using AddressSpaceView = LocalAddressSpaceView;
};
} // namespace

using PrimaryAllocator = SizeClassAllocator64<AP64>;
using Allocator = CombinedAllocator<PrimaryAllocator>;
using AllocatorCache = Allocator::AllocatorCache;

static Allocator allocator;
static AllocatorCache fallback_allocator_cache;
static StaticSpinMutex fallback_mutex;

static uptr max_malloc_size;

void __nsan::NsanAllocatorInit() {
  SetAllocatorMayReturnNull(common_flags()->allocator_may_return_null);
  allocator.Init(common_flags()->allocator_release_to_os_interval_ms);
  if (common_flags()->max_allocation_size_mb)
    max_malloc_size = Min(common_flags()->max_allocation_size_mb << 20,
                          kMaxAllowedMallocSize);
  else
    max_malloc_size = kMaxAllowedMallocSize;
}

static AllocatorCache *GetAllocatorCache(NsanThreadLocalMallocStorage *ms) {
  CHECK_LE(sizeof(AllocatorCache), sizeof(ms->allocator_cache));
  return reinterpret_cast<AllocatorCache *>(ms->allocator_cache);
}

void NsanThreadLocalMallocStorage::Init() {
  allocator.InitCache(GetAllocatorCache(this));
}

void NsanThreadLocalMallocStorage::CommitBack() {
  allocator.SwallowCache(GetAllocatorCache(this));
  allocator.DestroyCache(GetAllocatorCache(this));
}

static void *NsanAllocate(uptr size, uptr alignment, bool zero) {
  if (UNLIKELY(size > max_malloc_size)) {
    if (AllocatorMayReturnNull()) {
      Report("WARNING: NumericalStabilitySanitizer failed to allocate 0x%zx "
             "bytes\n",
             size);
      return nullptr;
    }
    BufferedStackTrace stack;
    GET_FATAL_STACK_TRACE_IF_EMPTY(&stack);
    ReportAllocationSizeTooBig(size, max_malloc_size, &stack);
  }
  if (UNLIKELY(IsRssLimitExceeded())) {
    if (AllocatorMayReturnNull())
      return nullptr;
    BufferedStackTrace stack;
    GET_FATAL_STACK_TRACE_IF_EMPTY(&stack);
    ReportRssLimitExceeded(&stack);
  }

  void *allocated;
  if (NsanThread *t = GetCurrentThread()) {
    AllocatorCache *cache = GetAllocatorCache(&t->malloc_storage());
    allocated = allocator.Allocate(cache, size, alignment);
  } else {
    SpinMutexLock l(&fallback_mutex);
    AllocatorCache *cache = &fallback_allocator_cache;
    allocated = allocator.Allocate(cache, size, alignment);
  }
  if (UNLIKELY(!allocated)) {
    SetAllocatorOutOfMemory();
    if (AllocatorMayReturnNull())
      return nullptr;
    BufferedStackTrace stack;
    GET_FATAL_STACK_TRACE_IF_EMPTY(&stack);
    ReportOutOfMemory(size, &stack);
  }
  auto *meta = reinterpret_cast<Metadata *>(allocator.GetMetaData(allocated));
  meta->requested_size = size;
  if (zero && allocator.FromPrimary(allocated))
    REAL(memset)(allocated, 0, size);
  __nsan_set_value_unknown(allocated, size);
  RunMallocHooks(allocated, size);
  return allocated;
}

void __nsan::NsanDeallocate(void *p) {
  DCHECK(p);
  RunFreeHooks(p);
  auto *meta = reinterpret_cast<Metadata *>(allocator.GetMetaData(p));
  uptr size = meta->requested_size;
  meta->requested_size = 0;
  if (flags().poison_in_free)
    __nsan_set_value_unknown(p, size);
  if (NsanThread *t = GetCurrentThread()) {
    AllocatorCache *cache = GetAllocatorCache(&t->malloc_storage());
    allocator.Deallocate(cache, p);
  } else {
    // In a just created thread, glibc's _dl_deallocate_tls might reach here
    // before nsan_current_thread is set.
    SpinMutexLock l(&fallback_mutex);
    AllocatorCache *cache = &fallback_allocator_cache;
    allocator.Deallocate(cache, p);
  }
}

static void *NsanReallocate(void *ptr, uptr new_size, uptr alignment) {
  Metadata *meta = reinterpret_cast<Metadata *>(allocator.GetMetaData(ptr));
  uptr old_size = meta->requested_size;
  uptr actually_allocated_size = allocator.GetActuallyAllocatedSize(ptr);
  if (new_size <= actually_allocated_size) {
    // We are not reallocating here.
    meta->requested_size = new_size;
    if (new_size > old_size)
      __nsan_set_value_unknown((u8 *)ptr + old_size, new_size - old_size);
    return ptr;
  }
  void *new_p = NsanAllocate(new_size, alignment, false);
  if (new_p) {
    uptr memcpy_size = Min(new_size, old_size);
    REAL(memcpy)(new_p, ptr, memcpy_size);
    __nsan_copy_values(new_p, ptr, memcpy_size);
    NsanDeallocate(ptr);
  }
  return new_p;
}

static void *NsanCalloc(uptr nmemb, uptr size) {
  if (UNLIKELY(CheckForCallocOverflow(size, nmemb))) {
    if (AllocatorMayReturnNull())
      return nullptr;
    BufferedStackTrace stack;
    GET_FATAL_STACK_TRACE_IF_EMPTY(&stack);
    ReportCallocOverflow(nmemb, size, &stack);
  }
  return NsanAllocate(nmemb * size, sizeof(u64), true);
}

static const void *AllocationBegin(const void *p) {
  if (!p)
    return nullptr;
  void *beg = allocator.GetBlockBegin(p);
  if (!beg)
    return nullptr;
  auto *b = reinterpret_cast<Metadata *>(allocator.GetMetaData(beg));
  if (!b)
    return nullptr;
  if (b->requested_size == 0)
    return nullptr;

  return beg;
}

static uptr AllocationSizeFast(const void *p) {
  return reinterpret_cast<Metadata *>(allocator.GetMetaData(p))->requested_size;
}

static uptr AllocationSize(const void *p) {
  if (!p)
    return 0;
  if (allocator.GetBlockBegin(p) != p)
    return 0;
  return AllocationSizeFast(p);
}

void *__nsan::nsan_malloc(uptr size) {
  return SetErrnoOnNull(NsanAllocate(size, sizeof(u64), false));
}

void *__nsan::nsan_calloc(uptr nmemb, uptr size) {
  return SetErrnoOnNull(NsanCalloc(nmemb, size));
}

void *__nsan::nsan_realloc(void *ptr, uptr size) {
  if (!ptr)
    return SetErrnoOnNull(NsanAllocate(size, sizeof(u64), false));
  if (size == 0) {
    NsanDeallocate(ptr);
    return nullptr;
  }
  return SetErrnoOnNull(NsanReallocate(ptr, size, sizeof(u64)));
}

void *__nsan::nsan_reallocarray(void *ptr, uptr nmemb, uptr size) {
  if (UNLIKELY(CheckForCallocOverflow(size, nmemb))) {
    errno = errno_ENOMEM;
    if (AllocatorMayReturnNull())
      return nullptr;
    BufferedStackTrace stack;
    GET_FATAL_STACK_TRACE_IF_EMPTY(&stack);
    ReportReallocArrayOverflow(nmemb, size, &stack);
  }
  return nsan_realloc(ptr, nmemb * size);
}

void *__nsan::nsan_valloc(uptr size) {
  return SetErrnoOnNull(NsanAllocate(size, GetPageSizeCached(), false));
}

void *__nsan::nsan_pvalloc(uptr size) {
  uptr PageSize = GetPageSizeCached();
  if (UNLIKELY(CheckForPvallocOverflow(size, PageSize))) {
    errno = errno_ENOMEM;
    if (AllocatorMayReturnNull())
      return nullptr;
    BufferedStackTrace stack;
    GET_FATAL_STACK_TRACE_IF_EMPTY(&stack);
    ReportPvallocOverflow(size, &stack);
  }
  // pvalloc(0) should allocate one page.
  size = size ? RoundUpTo(size, PageSize) : PageSize;
  return SetErrnoOnNull(NsanAllocate(size, PageSize, false));
}

void *__nsan::nsan_aligned_alloc(uptr alignment, uptr size) {
  if (UNLIKELY(!CheckAlignedAllocAlignmentAndSize(alignment, size))) {
    errno = errno_EINVAL;
    if (AllocatorMayReturnNull())
      return nullptr;
    BufferedStackTrace stack;
    GET_FATAL_STACK_TRACE_IF_EMPTY(&stack);
    ReportInvalidAlignedAllocAlignment(size, alignment, &stack);
  }
  return SetErrnoOnNull(NsanAllocate(size, alignment, false));
}

void *__nsan::nsan_memalign(uptr alignment, uptr size) {
  if (UNLIKELY(!IsPowerOfTwo(alignment))) {
    errno = errno_EINVAL;
    if (AllocatorMayReturnNull())
      return nullptr;
    BufferedStackTrace stack;
    GET_FATAL_STACK_TRACE_IF_EMPTY(&stack);
    ReportInvalidAllocationAlignment(alignment, &stack);
  }
  return SetErrnoOnNull(NsanAllocate(size, alignment, false));
}

int __nsan::nsan_posix_memalign(void **memptr, uptr alignment, uptr size) {
  if (UNLIKELY(!CheckPosixMemalignAlignment(alignment))) {
    if (AllocatorMayReturnNull())
      return errno_EINVAL;
    BufferedStackTrace stack;
    ReportInvalidPosixMemalignAlignment(alignment, &stack);
  }
  void *ptr = NsanAllocate(size, alignment, false);
  if (UNLIKELY(!ptr))
    // OOM error is already taken care of by NsanAllocate.
    return errno_ENOMEM;
  DCHECK(IsAligned((uptr)ptr, alignment));
  *memptr = ptr;
  return 0;
}

extern "C" {
uptr __sanitizer_get_current_allocated_bytes() {
  uptr stats[AllocatorStatCount];
  allocator.GetStats(stats);
  return stats[AllocatorStatAllocated];
}

uptr __sanitizer_get_heap_size() {
  uptr stats[AllocatorStatCount];
  allocator.GetStats(stats);
  return stats[AllocatorStatMapped];
}

uptr __sanitizer_get_free_bytes() { return 1; }

uptr __sanitizer_get_unmapped_bytes() { return 1; }

uptr __sanitizer_get_estimated_allocated_size(uptr size) { return size; }

int __sanitizer_get_ownership(const void *p) { return AllocationSize(p) != 0; }

const void *__sanitizer_get_allocated_begin(const void *p) {
  return AllocationBegin(p);
}

uptr __sanitizer_get_allocated_size(const void *p) { return AllocationSize(p); }

uptr __sanitizer_get_allocated_size_fast(const void *p) {
  DCHECK_EQ(p, __sanitizer_get_allocated_begin(p));
  uptr ret = AllocationSizeFast(p);
  DCHECK_EQ(ret, __sanitizer_get_allocated_size(p));
  return ret;
}

void __sanitizer_purge_allocator() { allocator.ForceReleaseToOS(); }
}
