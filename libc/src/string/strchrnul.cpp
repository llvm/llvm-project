//===-- Implementation of strchrnul --------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/string/strchrnul.h"
#include "src/string/string_utils.h"

#include "src/__support/common.h"

namespace __llvm_libc {

LLVM_LIBC_FUNCTION(char *, strchrnul, (const char *src, int c)) {
  const char ch = static_cast<char>(c);
  for (; *src && *src != ch; ++src)
    ;
  return const_cast<char *>(src);
}

} // namespace __llvm_libc
