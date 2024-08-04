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
//     : public basic_istream<charT, traits> {

//     // [spanbuf.cons], constructors
//
//     template<class ROS> explicit basic_ispanstream(ROS&& s);

#include <cassert>
#include <concepts>
#include <spanstream>

#include "constexpr_char_traits.h"
#include "nasty_string.h"
#include "test_convertible.h"
#include "test_macros.h"

#include "../../helper_types.h"

#include <string_view>

using namespace std::string_view_literals;

template <class CharT>
constexpr auto input_view = "1 2 3 4 5"sv;

template <typename CharT, typename TraitsT = std::char_traits<CharT>>
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

template <typename CharT, typename TraitsT = std::char_traits<CharT>>
void test() {
  using SpStream = std::basic_ispanstream<CharT, TraitsT>;

  // TODO:
  CharT arr[4];
  ReadOnlySpan<CharT, 4> ros{arr};
  assert(ros.size() == 4);

  {
    SpStream spSt(ros);
    assert(spSt.span().data() == arr);
    assert(!spSt.span().empty());
    assert(spSt.span().size() == 4);
  }

  CharT arr1[6];
  ReadOnlySpan<CharT, 6> ros2{arr1};

  {
    SpStream spSt(ros2);
    assert(spSt.span().data() != arr);
    assert(spSt.span().data() == arr1);
    assert(!spSt.span().empty());
    assert(spSt.span().size() == 6);
  }

  {
    auto r = input_view<char>;
    SpStream spSt{r};
    assert(spSt.span().data() != arr);
    assert(spSt.span().data() != arr1);
    assert(!spSt.span().empty());
    assert(spSt.span().size() == 9);
  }
}

int main(int, char**) {
// #ifndef TEST_HAS_NO_NASTY_STRING
//   test_sfinae<nasty_char, nasty_char_traits>();
// #endif
//   test_sfinae<char>();
//   test_sfinae<char, constexpr_char_traits<char>>();
// #ifndef TEST_HAS_NO_NASTY_STRING
//   test<nasty_char, nasty_char_traits>();
// #endif

  test<char>();
//   test<char, constexpr_char_traits<char>>();
// #ifndef TEST_HAS_NO_WIDE_CHARACTERS
//   test<wchar_t>();
//   test<wchar_t, constexpr_char_traits<wchar_t>>();
// #endif

// #ifndef TEST_HAS_NO_CHAR8_T
//   test<char8_t>();
//   test<char8_t, constexpr_char_traits<char8_t>>();
// #endif
//   test<char16_t>();
//   test<char16_t, constexpr_char_traits<char16_t>>();
//   test<char32_t>();
//   test<char32_t, constexpr_char_traits<char32_t>>();

  return 0;
}
