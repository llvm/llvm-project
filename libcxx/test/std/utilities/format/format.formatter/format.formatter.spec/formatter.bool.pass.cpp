//===----------------------------------------------------------------------===//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: libcpp-no-concepts
// UNSUPPORTED: libcpp-has-no-incomplete-format

// <format>

// [format.formatter.spec]:
// Each header that declares the template `formatter` provides the following
// enabled specializations:
// For each `charT`, for each cv-unqualified arithmetic type `ArithmeticT`
// other than char, wchar_t, char8_t, char16_t, or char32_t, a specialization
//    template<> struct formatter<ArithmeticT, charT>
//
// This file tests with `ArithmeticT = bool`, for each valid `charT`.

#include <format>
#include <cassert>
#include <concepts>
#include <type_traits>

#include "test_macros.h"
#include "make_string.h"

#define STR(S) MAKE_STRING(CharT, S)

template <class StringT, class StringViewT>
void test(StringT expected, StringViewT fmt, bool arg) {
  using CharT = typename StringT::value_type;
  auto parse_ctx = std::basic_format_parse_context<CharT>(fmt);
  std::formatter<bool, CharT> formatter;
  static_assert(std::semiregular<decltype(formatter)>);

  auto it = formatter.parse(parse_ctx);
  assert(it == fmt.end() - (!fmt.empty() && fmt.back() == '}'));

  StringT result;
  auto out = std::back_inserter(result);
  using FormatCtxT = std::basic_format_context<decltype(out), CharT>;

  auto format_ctx = std::__format_context_create<decltype(out), CharT>(
      out, std::make_format_args<FormatCtxT>(arg));
  formatter.format(arg, format_ctx);
  assert(result == expected);
}

template <class StringT>
void test_termination_condition(StringT expected, StringT f, bool arg) {
  // The format-spec is valid if completely consumed or terminates at a '}'.
  // The valid inputs all end with a '}'. The test is executed twice:
  // - first with the terminating '}',
  // - second consuming the entire input.
  using CharT = typename StringT::value_type;
  std::basic_string_view<CharT> fmt{f};
  assert(fmt.back() == CharT('}') && "Pre-condition failure");

  test(expected, fmt, arg);
  fmt.remove_suffix(1);
  test(expected, fmt, arg);
}

template <class CharT>
void test_boolean() {
  test_termination_condition(STR("true"), STR("}"), true);
  test_termination_condition(STR("false"), STR("}"), false);
}

int main(int, char**) {
  test_boolean<char>();
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  test_boolean<wchar_t>();
#endif

  return 0;
}
