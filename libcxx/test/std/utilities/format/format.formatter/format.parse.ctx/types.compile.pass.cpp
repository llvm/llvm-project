//===----------------------------------------------------------------------===//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: libcpp-no-concepts

// <format>

// Class typedefs:
// template<class charT>
// class basic_format_parse_context {
// public:
//   using char_type = charT;
//   using const_iterator = typename basic_string_view<charT>::const_iterator;
//   using iterator = const_iterator;
// }
//
// Namespace std typedefs:
// using format_parse_context = basic_format_parse_context<char>;
// using wformat_parse_context = basic_format_parse_context<wchar_t>;

#include <format>
#include <type_traits>

#include "test_macros.h"

template <class CharT>
constexpr void test() {
  static_assert(
      std::is_same_v<typename std::basic_format_parse_context<CharT>::char_type,
                     CharT>);
  static_assert(std::is_same_v<
                typename std::basic_format_parse_context<CharT>::const_iterator,
                typename std::basic_string_view<CharT>::const_iterator>);
  static_assert(
      std::is_same_v<
          typename std::basic_format_parse_context<CharT>::iterator,
          typename std::basic_format_parse_context<CharT>::const_iterator>);
}

constexpr void test() {
  test<char>();
  test<wchar_t>();
#ifndef _LIBCPP_NO_HAS_CHAR8_T
  test<char8_t>();
#endif
#ifndef _LIBCPP_HAS_NO_UNICODE_CHARS
  test<char16_t>();
  test<char32_t>();
#endif
}

static_assert(std::is_same_v<std::format_parse_context,
                             std::basic_format_parse_context<char> >);
static_assert(std::is_same_v<std::wformat_parse_context,
                             std::basic_format_parse_context<wchar_t> >);

// Required for MSVC internal test runner compatibility.
int main(int, char**) { return 0; }
