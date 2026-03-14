//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// <format>

// template<class T, class charT>
// concept formattable = ...

#include <concepts>
#include <format>

#include "test_macros.h"

template <class T, class CharT>
void assert_is_not_formattable() {
  static_assert(!std::formattable<T, CharT>);
}

template <class T, class CharT>
void assert_is_formattable() {
  // Only formatters for CharT == char || CharT == wchar_t are enabled for the
  // standard formatters. When CharT is a different type the formatter should
  // be disabled.
  if constexpr (std::same_as<CharT, char>
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
                || std::same_as<CharT, wchar_t>
#endif
  )
    static_assert(std::formattable<T, CharT>);
  else
    assert_is_not_formattable<T, CharT>();
}

template <class CharT>
void test() {
  assert_is_formattable<float, CharT>();
  assert_is_formattable<double, CharT>();
  assert_is_formattable<long double, CharT>();
}

void test() {
  test<char>();
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  test<wchar_t>();
#endif
  test<char8_t>();
  test<char16_t>();
  test<char32_t>();

  test<int>();
}
