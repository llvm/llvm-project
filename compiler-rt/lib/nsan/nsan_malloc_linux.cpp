//===- nsan_malloc_linux.cpp ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Interceptors for memory allocation functions on ELF OSes.
//
//===----------------------------------------------------------------------===//

#include "interception/interception.h"
#include "nsan.h"
#include "nsan_allocator.h"
#include "sanitizer_common/sanitizer_allocator_dlsym.h"
#include "sanitizer_common/sanitizer_common.h"
#include "sanitizer_common/sanitizer_platform.h"
#include "sanitizer_common/sanitizer_platform_interceptors.h"
#include "sanitizer_common/sanitizer_stacktrace.h"

#if !SANITIZER_APPLE && !SANITIZER_WINDOWS
using namespace __sanitizer;
using namespace __nsan;

namespace {
struct DlsymAlloc : public DlSymAllocator<DlsymAlloc> {
  static bool UseImpl() { return !nsan_initialized; }
};
} // namespace

INTERCEPTOR(void *, aligned_alloc, uptr align, uptr size) {
  return nsan_aligned_alloc(align, size);
}

INTERCEPTOR(void *, calloc, uptr nmemb, uptr size) {
  if (DlsymAlloc::Use())
    return DlsymAlloc::Callocate(nmemb, size);
  return nsan_calloc(nmemb, size);
}

INTERCEPTOR(void, free, void *ptr) {
  if (UNLIKELY(!ptr))
    return;
  if (DlsymAlloc::PointerIsMine(ptr))
    return DlsymAlloc::Free(ptr);
  NsanDeallocate(ptr);
}

INTERCEPTOR(void *, malloc, uptr size) {
  if (DlsymAlloc::Use())
    return DlsymAlloc::Allocate(size);
  return nsan_malloc(size);
}

INTERCEPTOR(void *, realloc, void *ptr, uptr size) {
  if (DlsymAlloc::Use() || DlsymAlloc::PointerIsMine(ptr))
    return DlsymAlloc::Realloc(ptr, size);
  return nsan_realloc(ptr, size);
}

#if SANITIZER_INTERCEPT_REALLOCARRAY
INTERCEPTOR(void *, reallocarray, void *ptr, uptr nmemb, uptr size) {
  return nsan_reallocarray(ptr, nmemb, size);
}
#endif // SANITIZER_INTERCEPT_REALLOCARRAY

INTERCEPTOR(int, posix_memalign, void **memptr, uptr align, uptr size) {
  return nsan_posix_memalign(memptr, align, size);
}

// Deprecated allocation functions (memalign, etc).
#if SANITIZER_INTERCEPT_MEMALIGN
INTERCEPTOR(void *, memalign, uptr align, uptr size) {
  return nsan_memalign(align, size);
}

INTERCEPTOR(void *, __libc_memalign, uptr align, uptr size) {
  return nsan_memalign(align, size);
}
#endif

void __nsan::InitializeMallocInterceptors() {
  INTERCEPT_FUNCTION(aligned_alloc);
  INTERCEPT_FUNCTION(calloc);
  INTERCEPT_FUNCTION(free);
  INTERCEPT_FUNCTION(malloc);
  INTERCEPT_FUNCTION(posix_memalign);
  INTERCEPT_FUNCTION(realloc);
#if SANITIZER_INTERCEPT_REALLOCARRAY
  INTERCEPT_FUNCTION(reallocarray);
#endif

#if SANITIZER_INTERCEPT_MEMALIGN
  INTERCEPT_FUNCTION(memalign);
  INTERCEPT_FUNCTION(__libc_memalign);
#endif
}

#endif
