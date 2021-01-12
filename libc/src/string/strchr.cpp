//===-- Implementation of strchr ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/string/strchr.h"

#include "src/__support/common.h"

namespace __llvm_libc {

// TODO: Look at performance benefits of comparing words.
LLVM_LIBC_FUNCTION(char *, strchr, (const char *src, int c)) {
  unsigned char *str =
      const_cast<unsigned char *>(reinterpret_cast<const unsigned char *>(src));
  const unsigned char ch = c;
  for (; *str && *str != ch; ++str)
    ;
  return *str == ch ? reinterpret_cast<char *>(str) : nullptr;
}

} // namespace __llvm_libc
