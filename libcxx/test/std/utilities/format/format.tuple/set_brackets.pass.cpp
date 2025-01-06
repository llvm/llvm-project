//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// UNSUPPORTED: GCC-ALWAYS_INLINE-FIXME

// <format>

// template<class charT, formattable<charT>... Ts>
//   struct formatter<pair-or-tuple<Ts...>, charT>

// constexpr void constexpr void set_brackets(basic_string_view<charT> opening,
//                                            basic_string_view<charT> closing) noexcept;

// Note this tests the basics of this function. It's tested in more detail in
// the format functions tests.

#include <format>
#include <tuple>
#include <utility>

#include "make_string.h"

#define SV(S) MAKE_STRING_VIEW(CharT, S)

template <class CharT, class Arg>
constexpr void test() {
  std::formatter<Arg, CharT> formatter;
  formatter.set_brackets(SV("open"), SV("close"));
  // Note the SV macro may throw, so can't use it.
  static_assert(noexcept(formatter.set_brackets(std::basic_string_view<CharT>{}, std::basic_string_view<CharT>{})));

  // Note there is no direct way to validate this function modified the object.
}

template <class CharT>
constexpr void test() {
  test<CharT, std::tuple<int>>();
  test<CharT, std::tuple<int, CharT>>();
  test<CharT, std::pair<int, CharT>>();
  test<CharT, std::tuple<int, CharT, bool>>();
}

constexpr bool test() {
  test<char>();
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  test<wchar_t>();
#endif

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
