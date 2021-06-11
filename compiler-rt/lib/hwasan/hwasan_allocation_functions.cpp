//===-- hwasan_allocation_functions.cpp -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file is a part of HWAddressSanitizer.
//
// Definitions for __sanitizer allocation functions.
//
//===----------------------------------------------------------------------===//

#include "hwasan.h"
#include "interception/interception.h"
#include "sanitizer_common/sanitizer_allocator_interface.h"
#include "sanitizer_common/sanitizer_tls_get_addr.h"

using namespace __hwasan;

static uptr allocated_for_dlsym;
static const uptr kDlsymAllocPoolSize = 1024;
static uptr alloc_memory_for_dlsym[kDlsymAllocPoolSize];

static bool IsInDlsymAllocPool(const void *ptr) {
  uptr off = (uptr)ptr - (uptr)alloc_memory_for_dlsym;
  return off < sizeof(alloc_memory_for_dlsym);
}

static void *AllocateFromLocalPool(uptr size_in_bytes) {
  uptr size_in_words = RoundUpTo(size_in_bytes, kWordSize) / kWordSize;
  void *mem = (void *)&alloc_memory_for_dlsym[allocated_for_dlsym];
  allocated_for_dlsym += size_in_words;
  CHECK_LT(allocated_for_dlsym, kDlsymAllocPoolSize);
  return mem;
}

int __sanitizer_posix_memalign(void **memptr, uptr alignment, uptr size) {
  GET_MALLOC_STACK_TRACE;
  CHECK_NE(memptr, 0);
  int res = hwasan_posix_memalign(memptr, alignment, size, &stack);
  return res;
}

void *__sanitizer_memalign(uptr alignment, uptr size) {
  GET_MALLOC_STACK_TRACE;
  return hwasan_memalign(alignment, size, &stack);
}

void *__sanitizer_aligned_alloc(uptr alignment, uptr size) {
  GET_MALLOC_STACK_TRACE;
  return hwasan_aligned_alloc(alignment, size, &stack);
}

void *__sanitizer___libc_memalign(uptr alignment, uptr size) {
  GET_MALLOC_STACK_TRACE;
  void *ptr = hwasan_memalign(alignment, size, &stack);
  if (ptr)
    DTLS_on_libc_memalign(ptr, size);
  return ptr;
}

void *__sanitizer_valloc(uptr size) {
  GET_MALLOC_STACK_TRACE;
  return hwasan_valloc(size, &stack);
}

void *__sanitizer_pvalloc(uptr size) {
  GET_MALLOC_STACK_TRACE;
  return hwasan_pvalloc(size, &stack);
}

void __sanitizer_free(void *ptr) {
  GET_MALLOC_STACK_TRACE;
  if (!ptr || UNLIKELY(IsInDlsymAllocPool(ptr)))
    return;
  hwasan_free(ptr, &stack);
}

void __sanitizer_cfree(void *ptr) {
  GET_MALLOC_STACK_TRACE;
  if (!ptr || UNLIKELY(IsInDlsymAllocPool(ptr)))
    return;
  hwasan_free(ptr, &stack);
}

uptr __sanitizer_malloc_usable_size(const void *ptr) {
  return __sanitizer_get_allocated_size(ptr);
}

struct __sanitizer_struct_mallinfo __sanitizer_mallinfo() {
  __sanitizer_struct_mallinfo sret;
  internal_memset(&sret, 0, sizeof(sret));
  return sret;
}

int __sanitizer_mallopt(int cmd, int value) { return 0; }

void __sanitizer_malloc_stats(void) {
  // FIXME: implement, but don't call REAL(malloc_stats)!
}

void *__sanitizer_calloc(uptr nmemb, uptr size) {
  GET_MALLOC_STACK_TRACE;
  if (UNLIKELY(!hwasan_inited))
    // Hack: dlsym calls calloc before REAL(calloc) is retrieved from dlsym.
    return AllocateFromLocalPool(nmemb * size);
  return hwasan_calloc(nmemb, size, &stack);
}

void *__sanitizer_realloc(void *ptr, uptr size) {
  GET_MALLOC_STACK_TRACE;
  if (UNLIKELY(IsInDlsymAllocPool(ptr))) {
    uptr offset = (uptr)ptr - (uptr)alloc_memory_for_dlsym;
    uptr copy_size = Min(size, kDlsymAllocPoolSize - offset);
    void *new_ptr;
    if (UNLIKELY(!hwasan_inited)) {
      new_ptr = AllocateFromLocalPool(copy_size);
    } else {
      copy_size = size;
      new_ptr = hwasan_malloc(copy_size, &stack);
    }
    internal_memcpy(new_ptr, ptr, copy_size);
    return new_ptr;
  }
  return hwasan_realloc(ptr, size, &stack);
}

void *__sanitizer_reallocarray(void *ptr, uptr nmemb, uptr size) {
  GET_MALLOC_STACK_TRACE;
  return hwasan_reallocarray(ptr, nmemb, size, &stack);
}

void *__sanitizer_malloc(uptr size) {
  GET_MALLOC_STACK_TRACE;
  if (UNLIKELY(!hwasan_init_is_running))
    ENSURE_HWASAN_INITED();
  if (UNLIKELY(!hwasan_inited))
    // Hack: dlsym calls malloc before REAL(malloc) is retrieved from dlsym.
    return AllocateFromLocalPool(size);
  return hwasan_malloc(size, &stack);
}

#if HWASAN_WITH_INTERCEPTORS
#  define INTERCEPTOR_ALIAS(RET, FN, ARGS...)                                 \
    extern "C" SANITIZER_INTERFACE_ATTRIBUTE RET WRAP(FN)(ARGS)               \
        ALIAS("__sanitizer_" #FN);                                            \
    extern "C" SANITIZER_INTERFACE_ATTRIBUTE SANITIZER_WEAK_ATTRIBUTE RET FN( \
        ARGS) ALIAS("__sanitizer_" #FN)

INTERCEPTOR_ALIAS(int, posix_memalign, void **memptr, SIZE_T alignment,
                  SIZE_T size);
INTERCEPTOR_ALIAS(void *, aligned_alloc, SIZE_T alignment, SIZE_T size);
INTERCEPTOR_ALIAS(void *, __libc_memalign, SIZE_T alignment, SIZE_T size);
INTERCEPTOR_ALIAS(void *, valloc, SIZE_T size);
INTERCEPTOR_ALIAS(void, free, void *ptr);
INTERCEPTOR_ALIAS(uptr, malloc_usable_size, const void *ptr);
INTERCEPTOR_ALIAS(void *, calloc, SIZE_T nmemb, SIZE_T size);
INTERCEPTOR_ALIAS(void *, realloc, void *ptr, SIZE_T size);
INTERCEPTOR_ALIAS(void *, reallocarray, void *ptr, SIZE_T nmemb, SIZE_T size);
INTERCEPTOR_ALIAS(void *, malloc, SIZE_T size);

#  if !SANITIZER_FREEBSD && !SANITIZER_NETBSD
INTERCEPTOR_ALIAS(void *, memalign, SIZE_T alignment, SIZE_T size);
INTERCEPTOR_ALIAS(void *, pvalloc, SIZE_T size);
INTERCEPTOR_ALIAS(void, cfree, void *ptr);
INTERCEPTOR_ALIAS(__sanitizer_struct_mallinfo, mallinfo);
INTERCEPTOR_ALIAS(int, mallopt, int cmd, int value);
INTERCEPTOR_ALIAS(void, malloc_stats, void);
#  endif
#endif  // #if HWASAN_WITH_INTERCEPTORS
