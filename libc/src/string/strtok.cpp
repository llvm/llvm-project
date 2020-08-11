//===-- Implementation of strtok ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/string/strtok.h"

#include "src/__support/common.h"
#include "utils/CPP/Bitset.h"

namespace __llvm_libc {

static char *strtok_str = nullptr;

char *LLVM_LIBC_ENTRYPOINT(strtok)(char *src, const char *delimiter_string) {
  cpp::Bitset<256> delimiter_set;
  for (; *delimiter_string; ++delimiter_string)
    delimiter_set.set(*delimiter_string);

  src = src ? src : strtok_str;
  for (; *src && delimiter_set.test(*src); ++src)
    ;
  if (!*src) {
    strtok_str = src;
    return nullptr;
  }
  char *token = src;
  for (; *src && !delimiter_set.test(*src); ++src)
    ;

  strtok_str = src;
  if (*strtok_str) {
    *strtok_str = '\0';
    ++strtok_str;
  }
  return token;
}

} // namespace __llvm_libc
