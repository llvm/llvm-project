//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// <spanstream>

//   template<class charT, class traits = char_traits<charT>>
//   class basic_ispanstream
//     : public basic_streambuf<charT, traits> {

//     // [spanbuf.cons], constructors
//
//     template<class ROS> explicit basic_ispanstream(ROS&& s);

#include <cassert>
#include <spanstream>

#include "constexpr_char_traits.h"
#include "nasty_string.h"
#include "test_macros.h"

#ifndef TEST_HAS_NO_NASTY_STRING
void test_sfinae_with_nasty_char() {
  using SpStream = std::basic_ispanstream<nasty_char, nasty_char_traits>;

  // TODO:
}
#endif // TEST_HAS_NO_NASTY_STRING

template <typename CharT, typename TraitsT = std::char_traits<CharT>>
void test_sfinae() {
  using SpStream = std::basic_ispanstream<CharT, TraitsT>;

  // TODO:
}

template <typename CharT, typename TraitsT = std::char_traits<CharT>>
void test() {
  using SpStream = std::basic_ispanstream<CharT, TraitsT>;

  // TODO:
}

int main(int, char**) {
#ifndef TEST_HAS_NO_NASTY_STRING
  test_sfinae_with_nasty_char();
#endif
  test_sfinae<char>();
  test_sfinae<char, constexpr_char_traits<char>>();
  test<char>();
  test<char, constexpr_char_traits<char>>();
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  test<wchar_t>();
  test<wchar_t, constexpr_char_traits<wchar_t>>();
#endif

  return 0;
}
