//===-- Implementation of stpncpy -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/string/stpncpy.h"
#include "src/string/memory_utils/bzero_implementations.h"

#include "src/__support/common.h"

namespace __llvm_libc {

LLVM_LIBC_FUNCTION(char *, stpncpy,
                   (char *__restrict dest, const char *__restrict src,
                    size_t n)) {
  size_t i;
  // Copy up until \0 is found.
  for (i = 0; i < n && src[i] != '\0'; ++i)
    dest[i] = src[i];
  // When n>strlen(src), n-strlen(src) \0 are appended.
  if (n > i)
    inline_bzero(dest + i, n - i);
  return dest + i;
}

} // namespace __llvm_libc
