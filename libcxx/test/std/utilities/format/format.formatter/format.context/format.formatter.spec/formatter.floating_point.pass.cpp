//===----------------------------------------------------------------------===//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: libcpp-no-concepts
// UNSUPPORTED: libcpp-has-no-incomplete-format
// UNSUPPORTED: LIBCXX-DEBUG-FIXME

// <format>

// [format.formatter.spec]:
// Each header that declares the template `formatter` provides the following
// enabled specializations:
// For each `charT`, for each cv-unqualified arithmetic type `ArithmeticT`
// other than char, wchar_t, char8_t, char16_t, or char32_t, a specialization
//    template<> struct formatter<ArithmeticT, charT>
//
// This file tests with `ArithmeticT = floating-point`, for each valid `charT`.
// Where `floating-point` is one of:
// - float
// - double
// - long double

// TODO FMT Enable after floating-point support has been enabled
#if 0
#include <format>
#include <cassert>
#include <concepts>
#include <type_traits>

#include "test_macros.h"
#include "make_string.h"

#define STR(S) MAKE_STRING(CharT, S)

template <class StringViewT, class ArithmeticT>
void test(StringViewT fmt, ArithmeticT arg) {
  using CharT = typename StringViewT::value_type;
  auto parse_ctx = std::basic_format_parse_context<CharT>(fmt);
  std::formatter<ArithmeticT, CharT> formatter;
  static_assert(std::semiregular<decltype(formatter)>);

  auto it = formatter.parse(parse_ctx);
  assert(it == fmt.end() - (!fmt.empty() && fmt.back() == '}'));

  std::basic_string<CharT> result;
  auto out = std::back_inserter(result);
  using FormatCtxT = std::basic_format_context<decltype(out), CharT>;

  auto format_ctx = std::__format_context_create<decltype(out), CharT>(
      out, std::make_format_args<FormatCtxT>(arg));
  formatter.format(arg, format_ctx);
  std::string expected = std::to_string(arg);
  assert(result == std::basic_string<CharT>(expected.begin(), expected.end()));
}

template <class StringT, class ArithmeticT>
void test_termination_condition(StringT f, ArithmeticT arg) {
  // The format-spec is valid if completely consumed or terminates at a '}'.
  // The valid inputs all end with a '}'. The test is executed twice:
  // - first with the terminating '}',
  // - second consuming the entire input.
  using CharT = typename StringT::value_type;
  std::basic_string_view<CharT> fmt{f};
  assert(fmt.back() == CharT('}') && "Pre-condition failure");

  test(fmt, arg);
  fmt.remove_suffix(1);
  test(fmt, arg);
}

template <class ArithmeticT, class CharT>
void test_float_type() {
  using A = ArithmeticT;
  test_termination_condition(STR("}"), A(-std::numeric_limits<float>::max()));
  test_termination_condition(STR("}"), A(-std::numeric_limits<float>::min()));
  test_termination_condition(STR("}"), A(-0.0));
  test_termination_condition(STR("}"), A(0.0));
  test_termination_condition(STR("}"), A(std::numeric_limits<float>::min()));
  test_termination_condition(STR("}"), A(std::numeric_limits<float>::max()));
  if (sizeof(A) > sizeof(float)) {
    test_termination_condition(STR("}"),
                               A(-std::numeric_limits<double>::max()));
    test_termination_condition(STR("}"),
                               A(-std::numeric_limits<double>::min()));
    test_termination_condition(STR("}"), A(std::numeric_limits<double>::min()));
    test_termination_condition(STR("}"), A(std::numeric_limits<double>::max()));
  }
  if (sizeof(A) > sizeof(double)) {
    test_termination_condition(STR("}"),
                               A(-std::numeric_limits<long double>::max()));
    test_termination_condition(STR("}"),
                               A(-std::numeric_limits<long double>::min()));
    test_termination_condition(STR("}"),
                               A(std::numeric_limits<long double>::min()));
    test_termination_condition(STR("}"),
                               A(std::numeric_limits<long double>::max()));
  }

  // TODO FMT Also test with special floating point values: +/-Inf NaN.
}

template <class CharT>
void test_all_float_types() {
  test_float_type<float, CharT>();
  test_float_type<double, CharT>();
  test_float_type<long double, CharT>();
}

int main(int, char**) {
  test_all_float_types<char>();
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  test_all_float_types<wchar_t>();
#endif

  return 0;
}
#else
int main(int, char**) { return 0; }
#endif
