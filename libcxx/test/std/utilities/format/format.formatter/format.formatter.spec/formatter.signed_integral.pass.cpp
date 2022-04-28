//===----------------------------------------------------------------------===//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: libcpp-has-no-incomplete-format

// <format>

// [format.formatter.spec]:
// Each header that declares the template `formatter` provides the following
// enabled specializations:
// For each `charT`, for each cv-unqualified arithmetic type `ArithmeticT`
// other than char, wchar_t, char8_t, char16_t, or char32_t, a specialization
//    template<> struct formatter<ArithmeticT, charT>
//
// This file tests with `ArithmeticT = signed integer`, for each valid `charT`.
// Where `signed integer` is one of:
// - signed char
// - short
// - int
// - long
// - long long
// - __int128_t

#include <format>
#include <cassert>
#include <concepts>
#include <type_traits>

#include "test_macros.h"
#include "make_string.h"

#define STR(S) MAKE_STRING(CharT, S)

template <class StringT, class StringViewT, class ArithmeticT>
void test(StringT expected, StringViewT fmt, ArithmeticT arg) {
  using CharT = typename StringT::value_type;
  auto parse_ctx = std::basic_format_parse_context<CharT>(fmt);
  std::formatter<ArithmeticT, CharT> formatter;
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

template <class StringT, class ArithmeticT>
void test_termination_condition(StringT expected, StringT f, ArithmeticT arg) {
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

template <class Arithmetic, class CharT>
void test_signed_integral_type() {
  using A = Arithmetic;
  test_termination_condition(STR("-128"), STR("}"), A(-128));
  test_termination_condition(STR("0"), STR("}"), A(0));
  test_termination_condition(STR("127"), STR("}"), A(127));
  if (sizeof(A) > 1) {
    test_termination_condition(STR("-32768"), STR("}"), A(-32768));
    test_termination_condition(STR("32767"), STR("}"), A(32767));
  }
  if (sizeof(A) > 2) {
    test_termination_condition(STR("-2147483648"), STR("}"), A(-2147483648));
    test_termination_condition(STR("2147483647"), STR("}"), A(2147483647));
  }
  if (sizeof(A) > 4) {
    // -9223372036854775808 can't be used directly, it gives the following
    // diagnostic:
    // integer literal is too large to be represented in a signed integer type,
    // interpreting as unsigned [-Werror,-Wimplicitly-unsigned-literal]
    test_termination_condition(STR("-9223372036854775808"), STR("}"),
                               A(-9223372036854775807 - 1));
    test_termination_condition(STR("9223372036854775807"), STR("}"),
                               A(9223372036854775807));
  }

  // TODO FMT Implement the __int128_t minimum and maximum once the formatter
  // can handle these values.
}

template <class CharT>
void test_all_signed_integral_types() {
  test_signed_integral_type<signed char, CharT>();
  test_signed_integral_type<short, CharT>();
  test_signed_integral_type<int, CharT>();
  test_signed_integral_type<long, CharT>();
  test_signed_integral_type<long long, CharT>();
#ifndef TEST_HAS_NO_INT128
  test_signed_integral_type<__int128_t, CharT>();
#endif
}

int main(int, char**) {
  test_all_signed_integral_types<char>();
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  test_all_signed_integral_types<wchar_t>();
#endif

  return 0;
}
