//===----------------------------------------------------------------------===//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: libcpp-has-no-incomplete-format
// TODO FMT Evaluate gcc-11 status
// UNSUPPORTED: gcc-11

// <format>

// [format.formatter.spec]:
// Each header that declares the template `formatter` provides the following
// enabled specializations:
// The specializations
//   template<> struct formatter<char, char>;
//   template<> struct formatter<char, wchar_t>;
//   template<> struct formatter<wchar_t, wchar_t>;

#include <format>
#include <cassert>
#include <concepts>
#include <type_traits>

#include "test_macros.h"
#include "make_string.h"

#define STR(S) MAKE_STRING(CharT, S)

template <class StringT, class StringViewT, class ArgumentT>
void test(StringT expected, StringViewT fmt, ArgumentT arg) {
  using CharT = typename StringT::value_type;
  auto parse_ctx = std::basic_format_parse_context<CharT>(fmt);
  std::formatter<ArgumentT, CharT> formatter;
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

template <class StringT, class ArgumentT>
void test_termination_condition(StringT expected, StringT f, ArgumentT arg) {
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

template <class ArgumentT, class CharT>
void test_char_type() {
  test_termination_condition(STR("a"), STR("}"), ArgumentT('a'));
  test_termination_condition(STR("z"), STR("}"), ArgumentT('z'));
  test_termination_condition(STR("A"), STR("}"), ArgumentT('A'));
  test_termination_condition(STR("Z"), STR("}"), ArgumentT('Z'));
  test_termination_condition(STR("0"), STR("}"), ArgumentT('0'));
  test_termination_condition(STR("9"), STR("}"), ArgumentT('9'));
}

int main(int, char**) {
  test_char_type<char, char>();
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  test_char_type<char, wchar_t>();
  test_char_type<wchar_t, wchar_t>();
#endif

  return 0;
}
