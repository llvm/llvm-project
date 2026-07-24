//===-- Implementation of hermetic heap functions -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/common.h"
#include "src/__support/macros/config.h"
#include "test/UnitTest/sbrk_heap.h"
#include <stddef.h>

namespace LIBC_NAMESPACE_DECL {
static SbrkHeap hermetic_heap(4096);
} // namespace LIBC_NAMESPACE_DECL

extern "C" {

void *malloc(size_t s) noexcept {
  return LIBC_NAMESPACE::hermetic_heap.allocate(s);
}

void free(void *ptr) noexcept { LIBC_NAMESPACE::hermetic_heap.free(ptr); }

void *realloc(void *mem, size_t s) noexcept {
  return LIBC_NAMESPACE::hermetic_heap.realloc(mem, s);
}

} // extern "C"
