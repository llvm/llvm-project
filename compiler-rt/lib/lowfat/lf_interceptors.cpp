//===-- lf_interceptors.cpp - LowFat Malloc/Free Interceptors -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Interceptors for malloc/free/calloc/realloc that route heap allocations
// through the LowFat allocator so all heap memory gets bounds-checked.
//
//===----------------------------------------------------------------------===//

#include "interception/interception.h"
#include "lf_allocator.h"
#include "lf_config.h"
#include "lf_interface.h"
#include "sanitizer_common/sanitizer_allocator_dlsym.h"

using namespace __sanitizer;

namespace __lowfat {
extern bool lowfat_inited;
extern bool lowfat_recover;
extern bool lowfat_right_align;
} // namespace __lowfat

// DlsymAlloc handles allocations that happen before our runtime is initialized
// (e.g., during dynamic linker symbol resolution). Uses a small static buffer.
namespace {
struct DlsymAlloc : public DlSymAllocator<DlsymAlloc> {
  static bool UseImpl() { return !__lowfat::lowfat_inited; }
};
} // namespace

// Helper: should this allocation go through LowFat?
// Allocations larger than our max size class fall back to system malloc.
static inline bool ShouldUseLowFat(uptr size) {
  return __lowfat::lowfat_inited && size > 0 && size <= __lowfat::kMaxSize;
}

//===----------------------------------------------------------------------===//
// Interceptors
//===----------------------------------------------------------------------===//

INTERCEPTOR(void *, malloc, uptr size) {
  if (DlsymAlloc::Use())
    return DlsymAlloc::Allocate(size);
  if (ShouldUseLowFat(size))
    return __lowfat::Allocate(size);
  return REAL(malloc)(size);
}

INTERCEPTOR(void, free, void *ptr) {
  if (!ptr)
    return;
  if (DlsymAlloc::PointerIsMine(ptr))
    return DlsymAlloc::Free(ptr);
  if (__lowfat::IsLowFatPointer((uptr)ptr)) {
    __lowfat::Deallocate(ptr);
    return;
  }
  REAL(free)(ptr);
}

INTERCEPTOR(void *, calloc, uptr nmemb, uptr size) {
  if (DlsymAlloc::Use())
    return DlsymAlloc::Callocate(nmemb, size);
  uptr total = nmemb * size;
  if (ShouldUseLowFat(total)) {
    void *ptr = __lowfat::Allocate(total);
    if (ptr)
      internal_memset(ptr, 0, total);
    return ptr;
  }
  return REAL(calloc)(nmemb, size);
}

INTERCEPTOR(void *, realloc, void *ptr, uptr size) {
  if (DlsymAlloc::Use() || DlsymAlloc::PointerIsMine(ptr))
    return DlsymAlloc::Realloc(ptr, size);

  // realloc(nullptr, size) == malloc(size)
  if (!ptr) {
    if (ShouldUseLowFat(size))
      return __lowfat::Allocate(size);
    return REAL(malloc)(size);
  }

  // realloc(ptr, 0) == free(ptr)
  if (size == 0) {
    if (__lowfat::IsLowFatPointer((uptr)ptr))
      __lowfat::Deallocate(ptr);
    else
      REAL(free)(ptr);
    return nullptr;
  }

  bool old_is_lowfat = __lowfat::IsLowFatPointer((uptr)ptr);

  if (ShouldUseLowFat(size)) {
    void *new_ptr = __lowfat::Allocate(size);
    if (!new_ptr)
      return nullptr;
    // Copy old data. For LowFat pointers, cap the copy to the bytes reachable
    // from the returned pointer to the end of the slot. In right-align mode
    // the user pointer may be shifted within the slot, so copying from the
    // slot base would corrupt the preserved contents.
    uptr copy_size = size;
    if (old_is_lowfat) {
      uptr old_class_size = __lowfat::GetSize((uptr)ptr);
      uptr old_base = __lowfat::GetBase((uptr)ptr);
      uptr old_offset = (uptr)ptr - old_base;
      uptr old_usable = old_class_size - old_offset;
      if (old_usable < copy_size)
        copy_size = old_usable;
    }
    internal_memcpy(new_ptr, ptr, copy_size);
    // Free old
    if (old_is_lowfat)
      __lowfat::Deallocate(ptr);
    else
      REAL(free)(ptr);
    return new_ptr;
  }

  // New size exceeds LowFat max — use system realloc
  if (old_is_lowfat) {
    // Must migrate from LowFat to system
    void *new_ptr = REAL(malloc)(size);
    if (!new_ptr)
      return nullptr;
    uptr old_size = __lowfat::GetSize((uptr)ptr);
    uptr copy_size = old_size < size ? old_size : size;
    internal_memcpy(new_ptr, ptr, copy_size);
    __lowfat::Deallocate(ptr);
    return new_ptr;
  }

  return REAL(realloc)(ptr, size);
}

INTERCEPTOR(void *, valloc, uptr size) {
  // valloc requires page-aligned allocation; fall back to system
  return REAL(valloc)(size);
}

INTERCEPTOR(int, posix_memalign, void **memptr, uptr alignment, uptr size) {
  // LowFat allocations are aligned to their size class, which may satisfy
  // the alignment requirement. For now, fall back to system.
  return REAL(posix_memalign)(memptr, alignment, size);
}

static inline void check_bounds(const void *ptr, uptr access_size, int is_write) {
  if (!ptr || access_size == 0) return;
  if (!__lowfat::CheckBounds((uptr)ptr, access_size)) {
    uptr start = (uptr)ptr;
    uptr size = __lowfat::GetSize(start);
    uptr base = __lowfat::GetBase(start);
    if (__lowfat::lowfat_recover)
      __lf_warn_oob(start + access_size, base, size, is_write);
    else
      __lf_report_oob(start + access_size, base, size, is_write);
  }
}

// The compiler pass instruments direct memory accesses inline, but cannot 
// instrument external libc calls. We intercept them here to check bounds.
INTERCEPTOR(void *, memset, void *dst, int v, uptr size) {
  check_bounds(dst, size, 1 /* write */);
  return REAL(memset)(dst, v, size);
}

INTERCEPTOR(void *, memcpy, void *dst, const void *src, uptr size) {
  check_bounds(dst, size, 1 /* write */);
  check_bounds(src, size, 0 /* read */);
  return REAL(memcpy)(dst, src, size);
}

INTERCEPTOR(void *, memmove, void *dst, const void *src, uptr size) {
  check_bounds(dst, size, 1 /* write */);
  check_bounds(src, size, 0 /* read */);
  return REAL(memmove)(dst, src, size);
}

#if SANITIZER_APPLE
INTERCEPTOR(uptr, malloc_size, void *ptr) {
  if (DlsymAlloc::PointerIsMine(ptr))
    return DlsymAlloc::GetSize(ptr);
  if (__lowfat::IsLowFatPointer((uptr)ptr))
    return __lowfat::GetSize((uptr)ptr);
  return REAL(malloc_size)(ptr);
}
#endif

namespace __lowfat {
void InitializeInterceptors() {
  static int inited = 0;
  CHECK_EQ(inited, 0);

  INTERCEPT_FUNCTION(malloc);
  INTERCEPT_FUNCTION(free);
  INTERCEPT_FUNCTION(calloc);
  INTERCEPT_FUNCTION(realloc);
  INTERCEPT_FUNCTION(valloc);
  INTERCEPT_FUNCTION(posix_memalign);
  INTERCEPT_FUNCTION(memset);
  INTERCEPT_FUNCTION(memcpy);
  INTERCEPT_FUNCTION(memmove);
#if SANITIZER_APPLE
  INTERCEPT_FUNCTION(malloc_size);
#endif
  inited = 1;
}
} // namespace __lowfat
