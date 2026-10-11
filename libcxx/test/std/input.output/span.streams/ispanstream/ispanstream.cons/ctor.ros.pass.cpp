//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++23

// <spanstream>

//   template<class charT, class traits = char_traits<charT>>
//   class basic_ispanstream
//     : public basic_istream<charT, traits> {

//     // [spanbuf.cons], constructors
//
//     template<class ROS> explicit basic_ispanstream(ROS&& s);

#include <cassert>
#include <concepts>
#include <spanstream>
#include <utility>

#include "constexpr_char_traits.h"
#include "nasty_string.h"
#include "test_convertible.h"
#include "test_macros.h"

#include "../../helper_types.h"

template <typename CharT, typename TraitsT>
void test_sfinae() {
  using SpStream = std::basic_ispanstream<CharT, TraitsT>;

  // Non-const convertible
  static_assert(std::constructible_from<SpStream, ReadOnlySpan<CharT>>);
  static_assert(!test_convertible<SpStream, ReadOnlySpan<CharT>>());

  // Const convertible
  static_assert(!std::constructible_from<SpStream, const ReadOnlySpan<CharT>>);
  static_assert(!test_convertible<SpStream, const ReadOnlySpan<CharT>>());

  // Non-const non-convertible
  static_assert(std::constructible_from<SpStream, NonReadOnlySpan<CharT>>);
  static_assert(!test_convertible<SpStream, NonReadOnlySpan<CharT>>());

  // Const non-convertible
  static_assert(!std::constructible_from<SpStream, const NonReadOnlySpan<CharT>>);
  static_assert(!test_convertible<SpStream, const NonReadOnlySpan<CharT>>());
}

template <typename CharT, typename TraitsT>
void test() {
  using SpStream = std::basic_ispanstream<CharT, TraitsT>;

  CharT arr[4];
  ReadOnlySpan<CharT, 4> ros{arr};
  assert(ros.size() == 4);

  {
    SpStream spSt(ros);
    assert(spSt.span().data() == arr);
    assert(spSt.span().size() == 4);
  }
  {
    SpStream spSt(std::move(ros));
    assert(spSt.span().data() == arr);
    assert(spSt.span().size() == 4);
  }
}

int main(int, char**) {
#ifndef TEST_HAS_NO_NASTY_STRING
  test_sfinae<nasty_char, nasty_char_traits>();
#endif

  test_sfinae<char, constexpr_char_traits<char>>();
  test_sfinae<char, std::char_traits<char>>();
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  test_sfinae<wchar_t, constexpr_char_traits<wchar_t>>();
  test_sfinae<wchar_t, std::char_traits<wchar_t>>();
#endif

#ifndef TEST_HAS_NO_NASTY_STRING
  test<nasty_char, nasty_char_traits>();
#endif

  test<char, constexpr_char_traits<char>>();
  test<char, std::char_traits<char>>();
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  test<wchar_t, constexpr_char_traits<wchar_t>>();
  test<wchar_t, std::char_traits<wchar_t>>();
#endif

  return 0;
}
