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
// class basic_stringstream

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
  std::basic_stringstream<CharT> ss(STR(" 123 456 "));
  static_assert(noexcept(ss.view()));
  assert(ss.view() == SV(" 123 456 "));
  int i = 0;
  ss >> i;
  assert(i == 123);
  ss >> i;
  assert(i == 456);
  ss << i << ' ' << 123;
  assert(ss.view() == SV("456 1236 "));
  ss.str(STR("5466 89 "));
  ss >> i;
  assert(i == 5466);
  ss >> i;
  assert(i == 89);
  ss << i << ' ' << 321;
  assert(ss.view() == SV("89 3219 "));

  const std::basic_stringstream<CharT> css(STR("abc"));
  static_assert(noexcept(css.view()));
  assert(css.view() == SV("abc"));

  std::basic_stringstream<CharT, my_char_traits<CharT>> tss;
  static_assert(std::is_same_v<decltype(tss.view()), std::basic_string_view<CharT, my_char_traits<CharT>>>);
}

int main(int, char**) {
  test<char>();
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  test<wchar_t>();
#endif
  std::stringstream ss;
  ss.write("\xd1", 1);
  assert(ss.view().length() == 1);
  return 0;
}
