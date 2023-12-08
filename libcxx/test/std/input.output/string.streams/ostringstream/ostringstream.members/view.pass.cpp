//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// <sstream>

// template <class charT, class traits = char_traits<charT>, class Allocator = allocator<charT> >
// class basic_ostringstream

// basic_string_view<charT, traits> view() const noexcept;

#include <sstream>
#include <cassert>
#include <type_traits>

#include "make_string.h"
#include "test_macros.h"

#define STR(S) MAKE_STRING(CharT, S)
#define SV(S) MAKE_STRING_VIEW(CharT, S)

template <class CharT>
struct my_char_traits : public std::char_traits<CharT> {};

template <class CharT>
static void test() {
  std::basic_ostringstream<CharT> ss(STR(" 123 456"));
  static_assert(noexcept(ss.view()));
  assert(ss.view() == SV(" 123 456"));
  int i = 0;
  ss << i;
  assert(ss.view() == SV("0123 456"));
  ss << 456;
  assert(ss.view() == SV("0456 456"));
  ss.str(STR(" 789"));
  assert(ss.view() == SV(" 789"));
  ss << SV("abc");
  assert(ss.view() == SV("abc9"));

  const std::basic_ostringstream<CharT> css(STR("abc"));
  static_assert(noexcept(css.view()));
  assert(css.view() == SV("abc"));

  std::basic_ostringstream<CharT, my_char_traits<CharT>> tss;
  static_assert(std::is_same_v<decltype(tss.view()), std::basic_string_view<CharT, my_char_traits<CharT>>>);
}

int main(int, char**) {
  test<char>();
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  test<wchar_t>();
#endif
  return 0;
}
