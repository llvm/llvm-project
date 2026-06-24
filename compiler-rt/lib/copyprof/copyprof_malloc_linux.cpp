//===-- copyprof_malloc_linux.cpp ----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// This file implements Linux-specific malloc and memory allocation
/// interceptors for CopyProf, wrapping application memory management calls.
///
//===----------------------------------------------------------------------===//

#include "copyprof/copyprof_interface_internal.h"
#include "copyprof/copyprof_internal.h"
#include "copyprof_allocator.h"
#include "copyprof_interceptors.h"
#include "interception/interception.h"
#include "sanitizer_common/sanitizer_allocator_dlsym.h"
#include "sanitizer_common/sanitizer_platform_interceptors.h"

using namespace __copyprof;
using namespace __sanitizer;

namespace __copyprof {

// This is needed to make the heap interceptors re-entrancy safe during the
// startup phase as `dlsym` itself might allocate.
struct DlsymAlloc : public DlSymAllocator<DlsymAlloc> {
  static bool UseImpl() { return copyprof_init_is_running; }
};

}  // namespace __copyprof

#define ENSURE_COPYPROF_INITIALIZED()         \
  do {                                        \
    if (UNLIKELY(!copyprof_is_initialized)) { \
      __copyprof_init_once();                 \
    }                                         \
  } while (0)

INTERCEPTOR(void*, malloc, SIZE_T size) {
  if (DlsymAlloc::Use())
    return DlsymAlloc::Allocate(size);
  ENSURE_COPYPROF_INITIALIZED();
  return CopyProfMalloc(size);
}

INTERCEPTOR(void*, calloc, SIZE_T nmemb, SIZE_T size) {
  if (DlsymAlloc::Use())
    return DlsymAlloc::Callocate(nmemb, size);
  ENSURE_COPYPROF_INITIALIZED();
  return CopyProfCalloc(nmemb, size);
}

INTERCEPTOR(void*, realloc, void* ptr, SIZE_T size) {
  if (DlsymAlloc::Use() || DlsymAlloc::PointerIsMine(ptr))
    return DlsymAlloc::Realloc(ptr, size);
  ENSURE_COPYPROF_INITIALIZED();
  return CopyProfRealloc(ptr, size);
}

INTERCEPTOR(void, free, void* ptr) {
  if (UNLIKELY(!ptr))
    return;
  if (DlsymAlloc::PointerIsMine(ptr))
    return DlsymAlloc::Free(ptr);
  ENSURE_COPYPROF_INITIALIZED();
  CopyProfFree(ptr);
}

// `free` and `realloc` only accept memory from the CopyProf allocator (or
// `DlsymAlloc`), so all other libc heap functions are replaced as well.

#if SANITIZER_INTERCEPT_CFREE
INTERCEPTOR(void, cfree, void* ptr) { WRAP(free)(ptr); }
#endif

#if SANITIZER_INTERCEPT_FREE_SIZED
INTERCEPTOR(void, free_sized, void* ptr, SIZE_T size) { WRAP(free)(ptr); }
#endif

#if SANITIZER_INTERCEPT_FREE_ALIGNED_SIZED
INTERCEPTOR(void, free_aligned_sized, void* ptr, SIZE_T alignment,
            SIZE_T size) {
  WRAP(free)(ptr);
}
#endif

#if SANITIZER_INTERCEPT_REALLOCARRAY
INTERCEPTOR(void*, reallocarray, void* ptr, SIZE_T nmemb, SIZE_T size) {
  ENSURE_COPYPROF_INITIALIZED();
  return CopyProfReallocarray(ptr, nmemb, size);
}
#endif

#if SANITIZER_INTERCEPT_MEMALIGN
INTERCEPTOR(void*, memalign, SIZE_T alignment, SIZE_T size) {
  ENSURE_COPYPROF_INITIALIZED();
  return CopyProfAllocate(size, alignment);
}
#endif

#if SANITIZER_INTERCEPT___LIBC_MEMALIGN
INTERCEPTOR(void*, __libc_memalign, SIZE_T alignment, SIZE_T size) {
  ENSURE_COPYPROF_INITIALIZED();
  return CopyProfAllocate(size, alignment);
}
#endif

#if SANITIZER_INTERCEPT_ALIGNED_ALLOC
// Unlike other sanitizers, `size` need not be a multiple of `alignment`: C17
// dropped that requirement, and glibc does not enforce it either.
INTERCEPTOR(void*, aligned_alloc, SIZE_T alignment, SIZE_T size) {
  ENSURE_COPYPROF_INITIALIZED();
  return CopyProfAllocate(size, alignment);
}
#endif

INTERCEPTOR(int, posix_memalign, void** memptr, SIZE_T alignment, SIZE_T size) {
  ENSURE_COPYPROF_INITIALIZED();
  return CopyProfPosixMemalign(memptr, alignment, size);
}

INTERCEPTOR(void*, valloc, SIZE_T size) {
  ENSURE_COPYPROF_INITIALIZED();
  return CopyProfValloc(size);
}

#if SANITIZER_INTERCEPT_PVALLOC
INTERCEPTOR(void*, pvalloc, SIZE_T size) {
  ENSURE_COPYPROF_INITIALIZED();
  return CopyProfPvalloc(size);
}
#endif

#if SANITIZER_INTERCEPT_MALLOC_USABLE_SIZE
INTERCEPTOR(SIZE_T, malloc_usable_size, void* ptr) {
  ENSURE_COPYPROF_INITIALIZED();
  return CopyProfMallocUsableSize(ptr);
}
#endif

namespace __copyprof {

void InitializeHeapInterceptors() {
  COPYPROF_INTERCEPT_FUNC(malloc);
  COPYPROF_INTERCEPT_FUNC(calloc);
  COPYPROF_INTERCEPT_FUNC(realloc);
  COPYPROF_INTERCEPT_FUNC(free);
#if SANITIZER_INTERCEPT_CFREE
  COPYPROF_INTERCEPT_FUNC(cfree);
#endif
#if SANITIZER_INTERCEPT_FREE_SIZED
  COPYPROF_INTERCEPT_FUNC(free_sized);
#endif
#if SANITIZER_INTERCEPT_FREE_ALIGNED_SIZED
  COPYPROF_INTERCEPT_FUNC(free_aligned_sized);
#endif
#if SANITIZER_INTERCEPT_REALLOCARRAY
  COPYPROF_INTERCEPT_FUNC(reallocarray);
#endif
#if SANITIZER_INTERCEPT_MEMALIGN
  COPYPROF_INTERCEPT_FUNC(memalign);
#endif
#if SANITIZER_INTERCEPT___LIBC_MEMALIGN
  COPYPROF_INTERCEPT_FUNC(__libc_memalign);
#endif
#if SANITIZER_INTERCEPT_ALIGNED_ALLOC
  COPYPROF_INTERCEPT_FUNC(aligned_alloc);
#endif
  COPYPROF_INTERCEPT_FUNC(posix_memalign);
  COPYPROF_INTERCEPT_FUNC(valloc);
#if SANITIZER_INTERCEPT_PVALLOC
  COPYPROF_INTERCEPT_FUNC(pvalloc);
#endif
#if SANITIZER_INTERCEPT_MALLOC_USABLE_SIZE
  COPYPROF_INTERCEPT_FUNC(malloc_usable_size);
#endif
}

}  // namespace __copyprof
