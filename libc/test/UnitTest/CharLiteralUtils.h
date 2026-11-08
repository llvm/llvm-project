//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Helper utilities for character literals in tests.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_TEST_UNITTEST_CHARLITERALUTILS_H
#define LLVM_LIBC_TEST_UNITTEST_CHARLITERALUTILS_H

#include "src/__support/CPP/type_traits.h"
#include "src/__support/macros/config.h"

// Returns the ordinary character or multicharacter `Literal` with the encoding
// prefix associated with `CharType`, if any.
//
// E.g.
//   ENCODED(char, "hello")    -> "hello"
//   ENCODED(wchar_t, "hello") -> L"hello"
#define ENCODED(CharType, Literal)                                             \
  ::LIBC_NAMESPACE::testing::choose_literal<CharType>(Literal, L##Literal)

namespace LIBC_NAMESPACE_DECL {
namespace testing {

template <typename CharT>
const CharT *choose_literal(const char *CharStr, const wchar_t *WCharStr) {
  if constexpr (LIBC_NAMESPACE::cpp::is_same_v<CharT, char>)
    return CharStr;
  else {
    static_assert(LIBC_NAMESPACE::cpp::is_same_v<CharT, wchar_t>);
    return WCharStr;
  }
}

template <typename CharT>
CharT choose_literal(char CharValue, wchar_t WCharValue) {
  if constexpr (LIBC_NAMESPACE::cpp::is_same_v<CharT, char>)
    return CharValue;
  else {
    static_assert(LIBC_NAMESPACE::cpp::is_same_v<CharT, wchar_t>);
    return WCharValue;
  }
}

} // namespace testing
} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_TEST_UNITTEST_CHARLITERALUTILS_H
