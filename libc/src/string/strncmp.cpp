//===-- Implementation of strncmp -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/string/strncmp.h"

#include "src/__support/common.h"
#include <stddef.h>

namespace __llvm_libc {

// TODO: Look at benefits for comparing words at a time.
LLVM_LIBC_FUNCTION(int, strncmp,
                   (const char *left, const char *right, size_t n)) {

  if (n == 0)
    return 0;

  for (; n > 1; --n, ++left, ++right) {
    char lc = *left;
    if (lc == '\0' || lc != *right)
      break;
  }
  return *reinterpret_cast<const unsigned char *>(left) -
         *reinterpret_cast<const unsigned char *>(right);
}

} // namespace __llvm_libc
