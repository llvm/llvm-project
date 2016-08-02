//===-- sanitizer_allocator.cc --------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is shared between AddressSanitizer and ThreadSanitizer
// run-time libraries.
// This allocator is used inside run-times.
//===----------------------------------------------------------------------===//

#include "sanitizer_allocator.h"
#include "sanitizer_allocator_internal.h"
#include "sanitizer_common.h"

namespace __sanitizer {

// ThreadSanitizer for Go uses libc malloc/free.
#if defined(SANITIZER_GO) || defined(SANITIZER_USE_MALLOC)
# if SANITIZER_LINUX && !SANITIZER_ANDROID
extern "C" void *__libc_malloc(uptr size);
extern "C" void __libc_free(void *ptr);
#  define LIBC_MALLOC __libc_malloc
#  define LIBC_FREE __libc_free
# else
#  include <stdlib.h>
#  define LIBC_MALLOC malloc
#  define LIBC_FREE free
# endif

static void *RawInternalAlloc(uptr size, InternalAllocatorCache *cache) {
  (void)cache;
  return LIBC_MALLOC(size);
}

static void RawInternalFree(void *ptr, InternalAllocatorCache *cache) {
  (void)cache;
  LIBC_FREE(ptr);
}

InternalAllocator *internal_allocator() {
  return 0;
}

#else // SANITIZER_GO

static ALIGNED(64) char internal_alloc_placeholder[sizeof(InternalAllocator)];
static atomic_uint8_t internal_allocator_initialized;
static StaticSpinMutex internal_alloc_init_mu;

static InternalAllocatorCache internal_allocator_cache;
static StaticSpinMutex internal_allocator_cache_mu;

InternalAllocator *internal_allocator() {
  InternalAllocator *internal_allocator_instance =
      reinterpret_cast<InternalAllocator *>(&internal_alloc_placeholder);
  if (atomic_load(&internal_allocator_initialized, memory_order_acquire) == 0) {
    SpinMutexLock l(&internal_alloc_init_mu);
    if (atomic_load(&internal_allocator_initialized, memory_order_relaxed) ==
        0) {
      internal_allocator_instance->Init(/* may_return_null*/ false);
      atomic_store(&internal_allocator_initialized, 1, memory_order_release);
    }
  }
  return internal_allocator_instance;
}

static void *RawInternalAlloc(uptr size, InternalAllocatorCache *cache) {
  if (cache == 0) {
    SpinMutexLock l(&internal_allocator_cache_mu);
    return internal_allocator()->Allocate(&internal_allocator_cache, size, 8,
                                          false);
  }
  return internal_allocator()->Allocate(cache, size, 8, false);
}

static void RawInternalFree(void *ptr, InternalAllocatorCache *cache) {
  if (!cache) {
    SpinMutexLock l(&internal_allocator_cache_mu);
    return internal_allocator()->Deallocate(&internal_allocator_cache, ptr);
  }
  internal_allocator()->Deallocate(cache, ptr);
}

#endif // SANITIZER_GO

const u64 kBlockMagic = 0x6A6CB03ABCEBC041ull;

void *InternalAlloc(uptr size, InternalAllocatorCache *cache) {
  if (size + sizeof(u64) < size)
    return nullptr;
  void *p = RawInternalAlloc(size + sizeof(u64), cache);
  if (!p)
    return nullptr;
  ((u64*)p)[0] = kBlockMagic;
  return (char*)p + sizeof(u64);
}

void InternalFree(void *addr, InternalAllocatorCache *cache) {
  if (!addr)
    return;
  addr = (char*)addr - sizeof(u64);
  CHECK_EQ(kBlockMagic, ((u64*)addr)[0]);
  ((u64*)addr)[0] = 0;
  RawInternalFree(addr, cache);
}

// LowLevelAllocator
static LowLevelAllocateCallback low_level_alloc_callback;

void *LowLevelAllocator::Allocate(uptr size) {
  // Align allocation size.
  size = RoundUpTo(size, 8);
  if (allocated_end_ - allocated_current_ < (sptr)size) {
    uptr size_to_allocate = Max(size, GetPageSizeCached());
    allocated_current_ =
        (char*)MmapOrDie(size_to_allocate, __func__);
    allocated_end_ = allocated_current_ + size_to_allocate;
    if (low_level_alloc_callback) {
      low_level_alloc_callback((uptr)allocated_current_,
                               size_to_allocate);
    }
  }
  CHECK(allocated_end_ - allocated_current_ >= (sptr)size);
  void *res = allocated_current_;
  allocated_current_ += size;
  return res;
}

void SetLowLevelAllocateCallback(LowLevelAllocateCallback callback) {
  low_level_alloc_callback = callback;
}

bool CallocShouldReturnNullDueToOverflow(uptr size, uptr n) {
  if (!size) return false;
  uptr max = (uptr)-1L;
  return (max / size) < n;
}

void NORETURN ReportAllocatorCannotReturnNull() {
  Report("%s's allocator is terminating the process instead of returning 0\n",
         SanitizerToolName);
  Report("If you don't like this behavior set allocator_may_return_null=1\n");
  CHECK(0);
  Die();
}

} // namespace __sanitizer
