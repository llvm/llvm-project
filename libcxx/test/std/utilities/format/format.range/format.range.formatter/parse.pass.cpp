//===----------------------------------------------------------------------===//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20
// UNSUPPORTED: libcpp-has-no-incomplete-format

// TODO FMT Fix this test using GCC, it currently times out.
// UNSUPPORTED: gcc-12

// This test requires the dylib support introduced in D92214.
// XFAIL: use_system_cxx_lib && target={{.+}}-apple-macosx10.{{.+}}
// XFAIL: use_system_cxx_lib && target={{.+}}-apple-macosx11.{{.+}}

// <format>

// template<class T, class charT = char>
//   requires same_as<remove_cvref_t<T>, T> && formattable<T, charT>
// class range_formatter

// template<class ParseContext>
//   constexpr typename ParseContext::iterator
//     parse(ParseContext& ctx);

// Note this tests the basics of this function. It's tested in more detail in
// the format functions test.

#include <cassert>
#include <concepts>
#include <format>

#include "test_format_context.h"
#include "test_macros.h"
#include "make_string.h"

#define SV(S) MAKE_STRING_VIEW(CharT, S)

template <class StringViewT>
constexpr void test_parse(StringViewT fmt) {
  using CharT    = typename StringViewT::value_type;
  auto parse_ctx = std::basic_format_parse_context<CharT>(fmt);
  std::range_formatter<int, CharT> formatter;
  static_assert(std::semiregular<decltype(formatter)>);

  std::same_as<typename StringViewT::iterator> auto it = formatter.parse(parse_ctx);
  assert(it == fmt.end() - (!fmt.empty() && fmt.back() == '}'));
}

template <class CharT>
constexpr void test_fmt() {
  test_parse(SV(""));
  test_parse(SV(":d"));

  test_parse(SV("}"));
  test_parse(SV(":d}"));
}

constexpr bool test() {
  test_fmt<char>();
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  test_fmt<wchar_t>();
#endif

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
