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
// This file tests with `ArithmeticT = unsigned integer`, for each valid `charT`.
// Where `unsigned integer` is one of:
// - unsigned char
// - unsigned short
// - unsigned
// - unsigned long
// - unsigned long long
// - __uint128_t

#include <format>
#include <cassert>
#include <concepts>
#include <iterator>
#include <type_traits>

#include "test_format_context.h"
#include "test_macros.h"
#include "make_string.h"

#define STR(S) MAKE_STRING(CharT, S)

template <class StringT, class StringViewT, class ArithmeticT>
void test(StringT expected, StringViewT fmt, ArithmeticT arg, std::size_t offset) {
  using CharT = typename StringT::value_type;
  auto parse_ctx = std::basic_format_parse_context<CharT>(fmt);
  std::formatter<ArithmeticT, CharT> formatter;
  static_assert(std::semiregular<decltype(formatter)>);

  auto it = formatter.parse(parse_ctx);
  assert(it == fmt.end() - offset);

  StringT result;
  auto out = std::back_inserter(result);
  using FormatCtxT = std::basic_format_context<decltype(out), CharT>;

  FormatCtxT format_ctx =
      test_format_context_create<decltype(out), CharT>(out, std::make_format_args<FormatCtxT>(arg));
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

  test(expected, fmt, arg, 1);
  fmt.remove_suffix(1);
  test(expected, fmt, arg, 0);
}

template <class ArithmeticT, class CharT>
void test_unsigned_integral_type() {
  using A = ArithmeticT;
  test_termination_condition(STR("0"), STR("}"), A(0));
  test_termination_condition(STR("255"), STR("}"), A(255));
  if (sizeof(A) > 1)
    test_termination_condition(STR("65535"), STR("}"), A(65535));
  if (sizeof(A) > 2)
    test_termination_condition(STR("4294967295"), STR("}"), A(4294967295));
  if (sizeof(A) > 4)
    test_termination_condition(STR("8446744073709551615"), STR("}"),
                               A(8446744073709551615));
#ifndef TEST_HAS_NO_INT128
  if (sizeof(A) > 8)
    test_termination_condition(
        STR("340282366920938463463374607431768211455"), STR("}"), A(std::numeric_limits<__uint128_t>::max()));
#endif
  // Test __formatter::__transform (libc++ specific).
  test_termination_condition(STR("FF"), STR("X}"), A(255));
}

template <class CharT>
void test_all_unsigned_integral_types() {
  test_unsigned_integral_type<unsigned char, CharT>();
  test_unsigned_integral_type<unsigned short, CharT>();
  test_unsigned_integral_type<unsigned, CharT>();
  test_unsigned_integral_type<unsigned long, CharT>();
  test_unsigned_integral_type<unsigned long long, CharT>();
#ifndef TEST_HAS_NO_INT128
  test_unsigned_integral_type<__uint128_t, CharT>();
#endif
}

int main(int, char**) {
  test_all_unsigned_integral_types<char>();
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  test_all_unsigned_integral_types<wchar_t>();
#endif

  return 0;
}
