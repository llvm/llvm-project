//===-- copyprof_allocator.cpp -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// This file implements the standalone memory allocator for CopyProf, managing
/// internal dynamic allocations required during profiling.
///
//===----------------------------------------------------------------------===//

#include "copyprof_allocator.h"

#include <pthread.h>

#include "copyprof_interceptors.h"
#include "copyprof_interface_internal.h"
#include "copyprof_per_object_state.h"
#include "copyprof_shadow.h"
#include "copyprof_state.h"
#include "sanitizer_common/sanitizer_allocator.h"
#include "sanitizer_common/sanitizer_allocator_checks.h"
#include "sanitizer_common/sanitizer_allocator_interface.h"
#include "sanitizer_common/sanitizer_allocator_report.h"
#include "sanitizer_common/sanitizer_common.h"
#include "sanitizer_common/sanitizer_errno.h"
#include "sanitizer_common/sanitizer_flags.h"
#include "sanitizer_common/sanitizer_internal_defs.h"
#include "sanitizer_common/sanitizer_platform_limits_posix.h"
#include "sanitizer_common/sanitizer_stacktrace.h"

namespace __copyprof {
namespace {

using namespace __sanitizer;

struct CopyProfMapUnmapCallback {
  void OnMap(uptr p, uptr size) const {}
  void OnMapSecondary(uptr p, uptr size, uptr user_begin,
                      uptr user_size) const {}
  void OnUnmap(uptr p, uptr size) const {}
};

struct CopyProfMetadata {
  // The requested allocation size in bytes.
  uptr requested_size;
};

struct AllocatorParameters {
  static constexpr uptr kSpaceBeg = ~(uptr)0;
  static constexpr uptr kSpaceSize = 0x40000000000ULL;
  static constexpr uptr kMetadataSize = sizeof(CopyProfMetadata);
  static constexpr uptr kFlags = 0;
  using SizeClassMap = DefaultSizeClassMap;
  using MapUnmapCallback = CopyProfMapUnmapCallback;
  using AddressSpaceView = LocalAddressSpaceView;
};

using PrimaryAllocator = SizeClassAllocator64<AllocatorParameters>;
using Allocator = CombinedAllocator<PrimaryAllocator>;
using AllocatorCache = Allocator::AllocatorCache;

// Per-thread allocator cache. It is mmap()ed, and thus zero-initialized as
// `AllocatorCache` requires, and committed back to `allocator` when its thread
// exits. As CopyProf does not intercept thread creation, the cache is created
// by the thread's first (de)allocation, which also arms the TSD destructor that
// commits it back.
pthread_key_t thread_cache_key;
__attribute__((
    tls_model("initial-exec"))) THREADLOCAL AllocatorCache* thread_cache;
// Set once the exiting thread's cache has been committed back. Later TSD
// destructors may still (de)allocate on this thread so they use the fallback
// cache, as a re-created cache could leak.
__attribute__((
    tls_model("initial-exec"))) THREADLOCAL bool thread_cache_committed_back;

uptr ThreadCacheSize() {
  return RoundUpTo(sizeof(AllocatorCache), GetPageSizeCached());
}

// Returns the calling thread's allocator cache, or `nullptr` if it has already
// been committed back.
AllocatorCache* GetThreadCache() {
  if (LIKELY(thread_cache))
    return thread_cache;
  if (UNLIKELY(thread_cache_committed_back))
    return nullptr;
  thread_cache =
      static_cast<AllocatorCache*>(MmapOrDie(ThreadCacheSize(), __func__));
  CHECK_EQ(0, pthread_setspecific(thread_cache_key, thread_cache));
  return thread_cache;
}

// Custom allocator that updates shadow memory accordingly.
struct CopyProfAllocator {
  // Heap allocations must be at least 8-byte aligned so independent
  // allocations never share a shadow byte (avoiding data races).
  static_assert(kMinAlignment == 1 << kShadowScale);

  Allocator allocator;
  // Fallback allocator cache and mutex used in the slow path when no per-thread
  // allocator cache is available.
  StaticSpinMutex fallback_mutex;
  AllocatorCache fallback_allocator_cache;

  void Init() {
    SetAllocatorMayReturnNull(common_flags()->allocator_may_return_null);
    allocator.Init(common_flags()->allocator_release_to_os_interval_ms);
  }

  CopyProfMetadata* GetMetadata(const void* ptr) {
    return reinterpret_cast<CopyProfMetadata*>(allocator.GetMetaData(ptr));
  }

  // `align` must be a non-zero power of two; callers are responsible for
  // validating it.
  void* Allocate(uptr size, uptr align = kMinAlignment) {
    if (UNLIKELY(!copyprof_is_initialized))
      __copyprof_init_once();
    align = (align < kMinAlignment) ? kMinAlignment : align;
    CHECK(IsPowerOfTwo(align));
    // C++ standard requires zero-sized allocations to return a non-null
    // pointer.
    if (size == 0)
      size = 1;
    void* allocated;
    if (AllocatorCache* cache = GetThreadCache()) {
      allocated = allocator.Allocate(cache, size, align);
    } else {
      SpinMutexLock lock(&fallback_mutex);
      allocated = allocator.Allocate(&fallback_allocator_cache, size, align);
    }

    if (UNLIKELY(!allocated)) {
      SetAllocatorOutOfMemory();
      if (AllocatorMayReturnNull())
        return nullptr;
      UNINITIALIZED BufferedStackTrace stack;
      stack.Unwind(StackTrace::GetCurrentPc(), GET_CURRENT_FRAME(),
                   /*context=*/nullptr, common_flags()->fast_unwind_on_fatal);
      ReportOutOfMemory(size, &stack);
    }
    GetMetadata(allocated)->requested_size = size;
    SmfContext ctx = __copyprof_state.smf_context;
    MarkApplicationMemory(allocated, size, ShouldMarkNewMemoryAsCopy(ctx));
    if (ctx == SmfContext::COPY)
      RecordPerObjectAllocation(__copyprof_state.current_this_ptr, size);
    return allocated;
  }

  void Deallocate(void* ptr) {
    if (UNLIKELY(!ptr))
      return;
    if (__copyprof_state.smf_context == SmfContext::DTOR) {
      __copyprof_state.is_transitive_copy &=
          IsMarkedAsCopy(ptr, GetMetadata(ptr)->requested_size);
    }
    if (AllocatorCache* cache = GetThreadCache()) {
      allocator.Deallocate(cache, ptr);
    } else {
      SpinMutexLock lock(&fallback_mutex);
      allocator.Deallocate(&fallback_allocator_cache, ptr);
    }
  }

  void* Calloc(uptr num, uptr size) {
    if (UNLIKELY(CheckForCallocOverflow(size, num))) {
      if (AllocatorMayReturnNull())
        return nullptr;
      UNINITIALIZED BufferedStackTrace stack;
      stack.Unwind(StackTrace::GetCurrentPc(), GET_CURRENT_FRAME(),
                   /*context=*/nullptr, common_flags()->fast_unwind_on_fatal);
      ReportCallocOverflow(num, size, &stack);
    }
    void* ptr = Allocate(num * size);
    // Memory from the secondary allocator is freshly mmap()ed, and thus already
    // zeroed.
    if (ptr && allocator.FromPrimary(ptr))
      REAL(memset)(ptr, 0, num * size);
    return ptr;
  }

  // Implemented via `Allocate` and `Deallocate` rather than
  // `CombinedAllocator::Reallocate`, s.t. shadow memory, metadata and
  // per-object state are updated as for any other (de)allocation.
  // Note that the shadow of the old memory is not carried over.
  // TODO: figure out whether this assumption is a problem in practice.
  void* Reallocate(void* old_ptr, uptr new_size) {
    CHECK(old_ptr && new_size);
    void* new_ptr = Allocate(new_size);
    if (new_ptr) {
      CHECK_NE(REAL(memcpy), nullptr);
      uptr memcpy_size = Min(new_size, GetMetadata(old_ptr)->requested_size);
      REAL(memcpy)(new_ptr, old_ptr, memcpy_size);
      Deallocate(old_ptr);
    }
    return new_ptr;
  }
};

CopyProfAllocator instance;

// TSD destructor that commits the exiting thread's cache back to `allocator`.
void CommitBackThreadCache(void* cache) {
  thread_cache = nullptr;
  thread_cache_committed_back = true;
  instance.allocator.SwallowCache(static_cast<AllocatorCache*>(cache));
  UnmapOrDie(cache, ThreadCacheSize());
}

}  // namespace

void InitializeCopyProfAllocator() {
  instance.Init();
  CHECK_EQ(0, pthread_key_create(&thread_cache_key, CommitBackThreadCache));
}

void* CopyProfAllocate(uptr size, uptr align) {
  if (UNLIKELY(align == 0 || !IsPowerOfTwo(align))) {
    errno = errno_EINVAL;
    if (AllocatorMayReturnNull())
      return nullptr;
    UNINITIALIZED BufferedStackTrace stack;
    stack.Unwind(StackTrace::GetCurrentPc(), GET_CURRENT_FRAME(),
                 /*context=*/nullptr, common_flags()->fast_unwind_on_fatal);
    ReportInvalidAllocationAlignment(align, &stack);
  }
  return SetErrnoOnNull(instance.Allocate(size, align));
}

void CopyProfDeallocate(void* ptr, uptr size, uptr align) {
  instance.Deallocate(ptr);
}

void* CopyProfMalloc(uptr size) {
  return SetErrnoOnNull(instance.Allocate(size));
}

void CopyProfFree(void* ptr) { instance.Deallocate(ptr); }

void* CopyProfCalloc(uptr num, uptr size) {
  return SetErrnoOnNull(instance.Calloc(num, size));
}

void* CopyProfRealloc(void* ptr, uptr size) {
  if (!ptr)
    return SetErrnoOnNull(instance.Allocate(size));
  if (size == 0) {
    instance.Deallocate(ptr);
    return nullptr;
  }
  return SetErrnoOnNull(instance.Reallocate(ptr, size));
}

void* CopyProfReallocarray(void* ptr, uptr num, uptr size) {
  if (UNLIKELY(CheckForCallocOverflow(size, num))) {
    errno = errno_ENOMEM;
    if (AllocatorMayReturnNull())
      return nullptr;
    UNINITIALIZED BufferedStackTrace stack;
    stack.Unwind(StackTrace::GetCurrentPc(), GET_CURRENT_FRAME(),
                 /*context=*/nullptr, common_flags()->fast_unwind_on_fatal);
    ReportReallocArrayOverflow(num, size, &stack);
  }
  return CopyProfRealloc(ptr, num * size);
}

void* CopyProfValloc(uptr size) {
  return SetErrnoOnNull(instance.Allocate(size, GetPageSizeCached()));
}

void* CopyProfPvalloc(uptr size) {
  uptr page_size = GetPageSizeCached();
  if (UNLIKELY(CheckForPvallocOverflow(size, page_size))) {
    errno = errno_ENOMEM;
    if (AllocatorMayReturnNull())
      return nullptr;
    UNINITIALIZED BufferedStackTrace stack;
    stack.Unwind(StackTrace::GetCurrentPc(), GET_CURRENT_FRAME(),
                 /*context=*/nullptr, common_flags()->fast_unwind_on_fatal);
    ReportPvallocOverflow(size, &stack);
  }
  // `pvalloc(0)` allocates one page.
  size = size ? RoundUpTo(size, page_size) : page_size;
  return SetErrnoOnNull(instance.Allocate(size, page_size));
}

int CopyProfPosixMemalign(void** memptr, uptr alignment, uptr size) {
  if (UNLIKELY(!CheckPosixMemalignAlignment(alignment))) {
    if (AllocatorMayReturnNull())
      return errno_EINVAL;
    UNINITIALIZED BufferedStackTrace stack;
    stack.Unwind(StackTrace::GetCurrentPc(), GET_CURRENT_FRAME(),
                 /*context=*/nullptr, common_flags()->fast_unwind_on_fatal);
    ReportInvalidPosixMemalignAlignment(alignment, &stack);
  }
  void* ptr = instance.Allocate(size, alignment);
  // Running out of memory has already been handled by `Allocate`.
  if (UNLIKELY(!ptr))
    return errno_ENOMEM;
  *memptr = ptr;
  return 0;
}

uptr CopyProfMallocUsableSize(const void* ptr) {
  // Return the requested size, and 0 for memory that was not allocated by
  // CopyProf.
  if (!ptr || instance.allocator.GetBlockBegin(ptr) != ptr)
    return 0;
  return instance.GetMetadata(ptr)->requested_size;
}

}  // namespace __copyprof
