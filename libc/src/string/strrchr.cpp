//===-- Implementation of strrchr------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/string/strrchr.h"

#include "src/__support/common.h"

namespace __llvm_libc {

char *LLVM_LIBC_ENTRYPOINT(strrchr)(const char *src, int c) {
  unsigned char *str =
      const_cast<unsigned char *>(reinterpret_cast<const unsigned char *>(src));
  const unsigned char ch = c;

  unsigned char *last_occurrence = nullptr;
  do {
    if (*str == ch)
      last_occurrence = str;
  } while (*str++);
  return reinterpret_cast<char *>(last_occurrence);
}

} // namespace __llvm_libc
