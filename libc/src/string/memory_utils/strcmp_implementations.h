//===-- str{,case}cmp implementation ----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_STRING_MEMORY_UTILS_STRCMP_IMPLEMENTATIONS_H
#define LLVM_LIBC_SRC_STRING_MEMORY_UTILS_STRCMP_IMPLEMENTATIONS_H

#include <stddef.h>

namespace __llvm_libc {

template <typename Comp>
constexpr static int strcmp_implementation(const char *left, const char *right,
                                           Comp &&comp) {
  // TODO: Look at benefits for comparing words at a time.
  for (; *left && !comp(*left, *right); ++left, ++right)
    ;
  return comp(*reinterpret_cast<const unsigned char *>(left),
              *reinterpret_cast<const unsigned char *>(right));
}

template <typename Comp>
constexpr static int strncmp_implementation(const char *left, const char *right,
                                            size_t n, Comp &&comp) {
  if (n == 0)
    return 0;

  // TODO: Look at benefits for comparing words at a time.
  for (; n > 1; --n, ++left, ++right) {
    char lc = *left;
    if (!comp(lc, '\0') || comp(lc, *right))
      break;
  }
  return comp(*reinterpret_cast<const unsigned char *>(left),
              *reinterpret_cast<const unsigned char *>(right));
}

} // namespace __llvm_libc

#endif // LLVM_LIBC_SRC_STRING_MEMORY_UTILS_STRCMP_IMPLEMENTATIONS_H
