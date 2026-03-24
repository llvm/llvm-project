//===-- Unit test allocation and string wrappers -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/macros/config.h"

#ifdef LIBC_TESTS_USE_INTERNAL_SCUDO_ALLOCATOR

#include "src/string/strncmp.h"
#include "src/string/strlen.h"
#include "src/stdlib/aligned_alloc.h"
#include "src/stdlib/calloc.h"
#include "src/stdlib/free.h"
#include "src/stdlib/malloc.h"
#include "src/stdlib/realloc.h"

#include <stddef.h>

extern "C" {

void *aligned_alloc(size_t alignment, size_t size) {
  return LIBC_NAMESPACE::aligned_alloc(alignment, size);
}

void *calloc(size_t num, size_t size) {
  return LIBC_NAMESPACE::calloc(num, size);
}

void free(void *ptr) { LIBC_NAMESPACE::free(ptr); }

void *malloc(size_t size) noexcept { return LIBC_NAMESPACE::malloc(size); }

void *realloc(void *ptr, size_t size) {
  return LIBC_NAMESPACE::realloc(ptr, size);
}

int strncmp(const char *lhs, const char *rhs, size_t count) {
  return LIBC_NAMESPACE::strncmp(lhs, rhs, count);
}

size_t strlen(const char *src) { return LIBC_NAMESPACE::strlen(src); }

} // extern "C"

#endif
