//===-- Implementation of strstr ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/string/strstr.h"

#include "src/__support/common.h"
#include <stddef.h>

namespace __llvm_libc {

// TODO: This is a simple brute force implementation. This can be
// improved upon using well known string matching algorithms.
char *LLVM_LIBC_ENTRYPOINT(strstr)(const char *haystack, const char *needle) {
  for (size_t i = 0; haystack[i]; ++i) {
    size_t j;
    for (j = 0; haystack[i + j] && haystack[i + j] == needle[j]; ++j)
      ;
    if (!needle[j])
      return const_cast<char *>(haystack + i);
  }
  return nullptr;
}

} // namespace __llvm_libc
