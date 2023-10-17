//===----------------------------------------------------------------------===//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// TODO FMT Investigate why this fails.
// UNSUPPORTED: stdlib=apple-libc++ && target={{.+}}-apple-macosx{{10.9|10.10|10.11|10.12|10.13|10.14|10.15|11.0|12.0}}

// <format>

// template<ranges::input_range R, class charT>
//   requires (K == range_format::string || K == range_format::debug_string)
// struct range-default-formatter<K, R, charT>

// template<class ParseContext>
//   constexpr typename ParseContext::iterator
//     parse(ParseContext& ctx);

// Note this tests the basics of this function. It's tested in more detail in
// the format.functions test.

#include <cassert>
#include <concepts>
#include <format>

#include "format.functions.tests.h"
#include "test_format_context.h"
#include "test_macros.h"

template <class FormatterT, class StringViewT>
constexpr void test_parse(StringViewT fmt, std::size_t offset) {
  using CharT    = typename StringViewT::value_type;
  auto parse_ctx = std::basic_format_parse_context<CharT>(fmt);
  FormatterT formatter;
  static_assert(std::semiregular<decltype(formatter)>);

  std::same_as<typename StringViewT::iterator> auto it = formatter.parse(parse_ctx);
  assert(it == fmt.end() - offset);
}

template <class StringViewT>
constexpr void test_formatters(StringViewT fmt, std::size_t offset) {
  using CharT = typename StringViewT::value_type;
  test_parse<std::formatter<test_range_format_string<std::basic_string<CharT>>, CharT>>(fmt, offset);
  test_parse<std::formatter<test_range_format_debug_string<std::basic_string<CharT>>, CharT>>(fmt, offset);
}

template <class CharT>
constexpr void test_char_type() {
  test_formatters(SV(""), 0);
  test_formatters(SV("}"), 1);
}

constexpr bool test() {
  test_char_type<char>();
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  test_char_type<wchar_t>();
#endif

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
