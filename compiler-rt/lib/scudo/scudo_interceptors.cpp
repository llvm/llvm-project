//===-- scudo_interceptors.cpp ----------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// Linux specific malloc interception functions.
///
//===----------------------------------------------------------------------===//

#include "sanitizer_common/sanitizer_platform.h"
#if SANITIZER_LINUX

#include "scudo_allocator.h"

#include "interception/interception.h"

using namespace __scudo;

INTERCEPTOR(void, free, void *ptr) {
  scudoFree(ptr, FromMalloc);
}

INTERCEPTOR(void, cfree, void *ptr) {
  scudoFree(ptr, FromMalloc);
}

INTERCEPTOR(void*, malloc, uptr size) {
  return scudoMalloc(size, FromMalloc);
}

INTERCEPTOR(void*, realloc, void *ptr, uptr size) {
  return scudoRealloc(ptr, size);
}

INTERCEPTOR(void*, calloc, uptr nmemb, uptr size) {
  return scudoCalloc(nmemb, size);
}

INTERCEPTOR(void*, valloc, uptr size) {
  return scudoValloc(size);
}

INTERCEPTOR(void*, memalign, uptr alignment, uptr size) {
  return scudoMemalign(alignment, size);
}

INTERCEPTOR(void*, __libc_memalign, uptr alignment, uptr size) {
  return scudoMemalign(alignment, size);
}

INTERCEPTOR(void*, pvalloc, uptr size) {
  return scudoPvalloc(size);
}

INTERCEPTOR(void*, aligned_alloc, uptr alignment, uptr size) {
  return scudoAlignedAlloc(alignment, size);
}

INTERCEPTOR(int, posix_memalign, void **memptr, uptr alignment, uptr size) {
  return scudoPosixMemalign(memptr, alignment, size);
}

INTERCEPTOR(uptr, malloc_usable_size, void *ptr) {
  return scudoMallocUsableSize(ptr);
}

INTERCEPTOR(int, mallopt, int cmd, int value) {
  return -1;
}

#endif  // SANITIZER_LINUX
