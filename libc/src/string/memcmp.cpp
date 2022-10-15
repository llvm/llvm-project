//===-- Implementation of memcmp ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/string/memcmp.h"
#include "src/string/memory_utils/memcmp_implementations.h"

#include <stddef.h> // size_t

namespace __llvm_libc {

LLVM_LIBC_FUNCTION(int, memcmp,
                   (const void *lhs, const void *rhs, size_t count)) {
  return static_cast<int>(inline_memcmp(static_cast<const char *>(lhs),
                                        static_cast<const char *>(rhs), count));
}

} // namespace __llvm_libc
