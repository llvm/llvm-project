//===-- Collection of utils for implementing ctype functions-------*-C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_CTYPE_UTILS_H
#define LLVM_LIBC_SRC___SUPPORT_CTYPE_UTILS_H

#include "src/__support/macros/attributes.h"

namespace LIBC_NAMESPACE {
namespace internal {

// ------------------------------------------------------
// Rationale: Since these classification functions are
// called in other functions, we will avoid the overhead
// of a function call by inlining them.
// ------------------------------------------------------

LIBC_INLINE static constexpr bool isalpha(unsigned ch) {
  return (ch | 32) - 'a' < 26;
}

LIBC_INLINE static constexpr bool isdigit(unsigned ch) {
  return (ch - '0') < 10;
}

LIBC_INLINE static constexpr bool isalnum(unsigned ch) {
  return isalpha(ch) || isdigit(ch);
}

LIBC_INLINE static constexpr bool isgraph(unsigned ch) {
  return 0x20 < ch && ch < 0x7f;
}

LIBC_INLINE static constexpr bool islower(unsigned ch) {
  return (ch - 'a') < 26;
}

LIBC_INLINE static constexpr bool isupper(unsigned ch) {
  return (ch - 'A') < 26;
}

LIBC_INLINE static constexpr bool isspace(unsigned ch) {
  return ch == ' ' || (ch - '\t') < 5;
}

LIBC_INLINE static constexpr int tolower(int ch) {
  if (isupper(ch))
    return ch + ('a' - 'A');
  return ch;
}

} // namespace internal
} // namespace LIBC_NAMESPACE

#endif //  LLVM_LIBC_SRC___SUPPORT_CTYPE_UTILS_H
