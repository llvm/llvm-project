//===-- Implementation of strncat -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/string/strncat.h"
#include "src/string/string_utils.h"
#include "src/string/strncpy.h"

#include "src/__support/common.h"

namespace __llvm_libc {

LLVM_LIBC_FUNCTION(char *, strncat,
                   (char *__restrict dest, const char *__restrict src,
                    size_t count)) {
  size_t srcLength = internal::string_length(src);
  size_t copyAmount = srcLength > count ? count : srcLength;
  size_t destLength = internal::string_length(dest);
  __llvm_libc::strncpy(dest + destLength, src, copyAmount);
  dest[destLength + copyAmount] = '\0';
  return dest;
}

} // namespace __llvm_libc
