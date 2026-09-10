//===-- Implementation of coverage test utilities -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/common.h"
#include "src/__support/macros/config.h"
#include <stddef.h>

namespace LIBC_NAMESPACE_DECL {
void *memset(void *ptr, int value, size_t count);
} // namespace LIBC_NAMESPACE_DECL

extern "C" {

void *malloc(size_t);

void *calloc(size_t num, size_t size) {
  size_t total;
  if (__builtin_mul_overflow(num, size, &total))
    return nullptr;
  void *mem = malloc(total);
  if (mem != nullptr)
    LIBC_NAMESPACE::memset(mem, 0, total);
  return mem;
}

int *__llvm_libc_errno() noexcept;
int *__errno_location() { return __llvm_libc_errno(); }

} // extern "C"
