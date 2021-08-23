//===-- Collection of utils for implementing ctype functions-------*-C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_SUPPORT_CTYPE_UTILS_H
#define LLVM_LIBC_SRC_SUPPORT_CTYPE_UTILS_H

namespace __llvm_libc {
namespace internal {

// ------------------------------------------------------
// Rationale: Since these classification functions are
// called in other functions, we will avoid the overhead
// of a function call by inlining them.
// ------------------------------------------------------

static constexpr bool isalpha(unsigned ch) { return (ch | 32) - 'a' < 26; }

static constexpr bool isdigit(unsigned ch) { return (ch - '0') < 10; }

static constexpr bool isalnum(unsigned ch) {
  return isalpha(ch) || isdigit(ch);
}

static constexpr bool isgraph(unsigned ch) { return 0x20 < ch && ch < 0x7f; }

static constexpr bool islower(unsigned ch) { return (ch - 'a') < 26; }

static constexpr bool isupper(unsigned ch) { return (ch - 'A') < 26; }

static constexpr bool isspace(unsigned ch) {
  return ch == ' ' || (ch - '\t') < 5;
}

} // namespace internal
} // namespace __llvm_libc

#endif //  LLVM_LIBC_SRC_SUPPORT_CTYPE_UTILS_H
