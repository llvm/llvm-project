//===----------------------------------------------------------------------===//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// <format>

// template<ranges::input_range R, class charT>
//   requires (K == range_format::string || K == range_format::debug_string)
// struct range-default-formatter<K, R, charT>

// template<class FormatContext>
//   typename FormatContext::iterator
//     format(const T& ref, FormatContext& ctx) const;

// Note this tests the basics of this function. It's tested in more detail in
// the format.functions test.

#include <cassert>
#include <concepts>
#include <format>

#include "format.functions.tests.h"
#include "test_format_context.h"
#include "test_macros.h"

template <class StringViewT, class ArgT>
void test_format(StringViewT expected, ArgT arg) {
  using CharT      = typename StringViewT::value_type;
  using String     = std::basic_string<CharT>;
  using OutIt      = std::back_insert_iterator<String>;
  using FormatCtxT = std::basic_format_context<OutIt, CharT>;

  std::formatter<ArgT, CharT> formatter;

  String result;
  OutIt out             = std::back_inserter(result);
  FormatCtxT format_ctx = test_format_context_create<OutIt, CharT>(out, std::make_format_args<FormatCtxT>(arg));
  formatter.format(arg, format_ctx);
  assert(result == expected);
}

template <class CharT>
void test_fmt() {
  test_format(SV("hello"), test_range_format_string<std::basic_string<CharT>>{STR("hello")});
  test_format(SV("hello"), test_range_format_debug_string<std::basic_string<CharT>>{STR("hello")});
}

void test() {
  test_fmt<char>();
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  test_fmt<wchar_t>();
#endif
}

int main(int, char**) {
  test();

  return 0;
}
