//===-- Collection of utils for implementing ctype functions-------*-C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_CTYPE_CTYPE_UTILS_H
#define LLVM_LIBC_SRC_CTYPE_CTYPE_UTILS_H

namespace __llvm_libc {
namespace internal {

// ------------------------------------------------------
// Rationale: Since these classification functions are
// called in other functions, we will avoid the overhead
// of a function call by inlining them.
// ------------------------------------------------------

static inline int isdigit(int c) {
  const unsigned ch = c;
  return (ch - '0') < 10;
}

static inline int isalpha(int c) {
  const unsigned ch = c;
  return (ch | 32) - 'a' < 26;
}

} // namespace internal
} // namespace __llvm_libc

#endif //  LLVM_LIBC_SRC_CTYPE_CTYPE_UTILS_H
