//===-- Implementation of mempcpy ----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/string/mempcpy.h"
#include "src/string/memory_utils/memcpy_implementations.h"

#include "src/__support/common.h"
#include <stddef.h> // For size_t.

namespace __llvm_libc {

LLVM_LIBC_FUNCTION(void *, mempcpy,
                   (void *__restrict dest, const void *__restrict src,
                    size_t count)) {
  char *result = reinterpret_cast<char *>(dest);
  inline_memcpy(result, reinterpret_cast<const char *>(src), count);
  return result + count;
}

} // namespace __llvm_libc
