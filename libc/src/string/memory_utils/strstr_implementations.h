//===-- str{,case}str implementation ----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_STRING_MEMORY_UTILS_STRSTR_IMPLEMENTATIONS_H
#define LLVM_LIBC_SRC_STRING_MEMORY_UTILS_STRSTR_IMPLEMENTATIONS_H

#include <stddef.h>

namespace __llvm_libc {

template <typename Comp>
constexpr static char *strstr_implementation(const char *haystack,
                                             const char *needle, Comp &&comp) {
  // TODO: This is a simple brute force implementation. This can be
  // improved upon using well known string matching algorithms.
  for (size_t i = 0; comp(haystack[i], 0); ++i) {
    size_t j = 0;
    for (; comp(haystack[i + j], 0) && !comp(haystack[i + j], needle[j]); ++j)
      ;
    if (!comp(needle[j], 0))
      return const_cast<char *>(haystack + i);
  }
  return nullptr;
}

} // namespace __llvm_libc

#endif // LLVM_LIBC_SRC_STRING_MEMORY_UTILS_STRSTR_IMPLEMENTATIONS_H
