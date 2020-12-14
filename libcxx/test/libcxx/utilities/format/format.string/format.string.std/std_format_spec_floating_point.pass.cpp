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

// Tests the parsing of the format string as specified in [format.string.std].
// It validates whether the std-format-spec is valid for a floating-point type.

#include <format>
#include <cassert>
#ifndef _LIBCPP_HAS_NO_LOCALIZATION
#  include <iostream>
#endif

#include "concepts_precision.h"
#include "test_macros.h"
#include "make_string.h"
#include "test_exception.h"

#define CSTR(S) MAKE_CSTRING(CharT, S)

using namespace std::__format_spec;

template <class CharT>
using Parser = __parser_floating_point<CharT>;

template <class CharT>
struct Expected {
  CharT fill = CharT(' ');
  _Flags::_Alignment alignment = _Flags::_Alignment::__right;
  _Flags::_Sign sign = _Flags::_Sign::__default;
  bool alternate_form = false;
  bool zero_padding = false;
  uint32_t width = 0;
  bool width_as_arg = false;
  uint32_t precision = std::__format::__number_max;
  bool precision_as_arg = true;
  bool locale_specific_form = false;
  _Flags::_Type type = _Flags::_Type::__default;
};

template <class CharT>
constexpr void test(Expected<CharT> expected, size_t size, std::basic_string_view<CharT> fmt) {
  // Initialize parser with sufficient arguments to avoid the parsing to fail
  // due to insufficient arguments.
  std::basic_format_parse_context<CharT> parse_ctx(fmt, std::__format::__number_max);
  auto begin = parse_ctx.begin();
  auto end = parse_ctx.end();
  Parser<CharT> parser;
  auto it = parser.parse(parse_ctx);

  assert(begin == parse_ctx.begin());
  assert(end == parse_ctx.end());

  assert(begin + size == it);
  assert(parser.__fill == expected.fill);
  assert(parser.__alignment == expected.alignment);
  assert(parser.__sign == expected.sign);
  assert(parser.__alternate_form == expected.alternate_form);
  assert(parser.__zero_padding == expected.zero_padding);
  assert(parser.__width == expected.width);
  assert(parser.__width_as_arg == expected.width_as_arg);
  assert(parser.__precision == expected.precision);
  assert(parser.__precision_as_arg == expected.precision_as_arg);
  assert(parser.__locale_specific_form == expected.locale_specific_form);
  assert(parser.__type == expected.type);
}

template <class CharT>
constexpr void test(Expected<CharT> expected, size_t size, const CharT* f) {
  // The format-spec is valid if completely consumed or terminates at a '}'.
  // The valid inputs all end with a '}'. The test is executed twice:
  // - first with the terminating '}',
  // - second consuming the entire input.
  std::basic_string_view<CharT> fmt{f};
  assert(fmt.back() == CharT('}') && "Pre-condition failure");

  test(expected, size, fmt);
  fmt.remove_suffix(1);
  test(expected, size, fmt);
}

template <class CharT>
constexpr void test() {
  Parser<CharT> parser;

  assert(parser.__fill == CharT(' '));
  assert(parser.__alignment == _Flags::_Alignment::__default);
  assert(parser.__sign == _Flags::_Sign::__default);
  assert(parser.__alternate_form == false);
  assert(parser.__zero_padding == false);
  assert(parser.__width == 0);
  assert(parser.__width_as_arg == false);
  assert(parser.__precision == std::__format::__number_max);
  assert(parser.__precision_as_arg == true);
  assert(parser.__locale_specific_form == false);
  assert(parser.__type == _Flags::_Type::__default);

  // Depending on whether or not a precision is specified the results differ.
  // Table 65: Meaning of type options for floating-point types [tab:format.type.float]

  test({}, 0, CSTR("}"));
  test({.precision = 0, .precision_as_arg = false, .type = _Flags::_Type::__general_lower_case}, 2, CSTR(".0}"));
  test({.precision = 1, .precision_as_arg = true, .type = _Flags::_Type::__general_lower_case}, 4, CSTR(".{1}}"));

  test({.type = _Flags::_Type::__float_hexadecimal_lower_case}, 1, CSTR("a}"));
  test({.type = _Flags::_Type::__float_hexadecimal_upper_case}, 1, CSTR("A}"));

  test({.precision = 6, .precision_as_arg = false, .type = _Flags::_Type::__scientific_lower_case}, 1, CSTR("e}"));
  test({.precision = 0, .precision_as_arg = false, .type = _Flags::_Type::__scientific_lower_case}, 3, CSTR(".0e}"));
  test({.precision = 1, .precision_as_arg = true, .type = _Flags::_Type::__scientific_lower_case}, 5, CSTR(".{1}e}"));
  test({.precision = 6, .precision_as_arg = false, .type = _Flags::_Type::__scientific_upper_case}, 1, CSTR("E}"));
  test({.precision = 0, .precision_as_arg = false, .type = _Flags::_Type::__scientific_upper_case}, 3, CSTR(".0E}"));
  test({.precision = 1, .precision_as_arg = true, .type = _Flags::_Type::__scientific_upper_case}, 5, CSTR(".{1}E}"));

  test({.precision = 6, .precision_as_arg = false, .type = _Flags::_Type::__fixed_lower_case}, 1, CSTR("f}"));
  test({.precision = 0, .precision_as_arg = false, .type = _Flags::_Type::__fixed_lower_case}, 3, CSTR(".0f}"));
  test({.precision = 1, .precision_as_arg = true, .type = _Flags::_Type::__fixed_lower_case}, 5, CSTR(".{1}f}"));
  test({.precision = 6, .precision_as_arg = false, .type = _Flags::_Type::__fixed_upper_case}, 1, CSTR("F}"));
  test({.precision = 0, .precision_as_arg = false, .type = _Flags::_Type::__fixed_upper_case}, 3, CSTR(".0F}"));
  test({.precision = 1, .precision_as_arg = true, .type = _Flags::_Type::__fixed_upper_case}, 5, CSTR(".{1}F}"));

  test({.precision = 6, .precision_as_arg = false, .type = _Flags::_Type::__general_lower_case}, 1, CSTR("g}"));
  test({.precision = 0, .precision_as_arg = false, .type = _Flags::_Type::__general_lower_case}, 3, CSTR(".0g}"));
  test({.precision = 1, .precision_as_arg = true, .type = _Flags::_Type::__general_lower_case}, 5, CSTR(".{1}g}"));
  test({.precision = 6, .precision_as_arg = false, .type = _Flags::_Type::__general_upper_case}, 1, CSTR("G}"));
  test({.precision = 0, .precision_as_arg = false, .type = _Flags::_Type::__general_upper_case}, 3, CSTR(".0G}"));
  test({.precision = 1, .precision_as_arg = true, .type = _Flags::_Type::__general_upper_case}, 5, CSTR(".{1}G}"));

  // *** Align-fill ***
  test({.alignment = _Flags::_Alignment::__left}, 1, CSTR("<}"));
  test({.alignment = _Flags::_Alignment::__center}, 1, "^}");
  test({.alignment = _Flags::_Alignment::__right}, 1, ">}");

  test({.fill = CharT('L'), .alignment = _Flags::_Alignment::__left}, 2, CSTR("L<}"));
  test({.fill = CharT('#'), .alignment = _Flags::_Alignment::__center}, 2, CSTR("#^}"));
  test({.fill = CharT('0'), .alignment = _Flags::_Alignment::__right}, 2, CSTR("0>}"));

  test_exception<Parser<CharT>>("The format-spec fill field contains an invalid character", CSTR("{<"));
  test_exception<Parser<CharT>>("The format-spec fill field contains an invalid character", CSTR("}<"));

  // *** Sign ***
  test({.sign = _Flags::_Sign::__minus}, 1, CSTR("-}"));
  test({.sign = _Flags::_Sign::__plus}, 1, CSTR("+}"));
  test({.sign = _Flags::_Sign::__space}, 1, CSTR(" }"));

  // *** Alternate form ***
  test({.alternate_form = true}, 1, CSTR("#}"));

  // *** Zero padding ***
  // TODO FMT What to do with zero-padding without a width?
  // [format.string.std]/13
  //   A zero (0) character preceding the width field pads the field with
  //   leading zeros (following any indication of sign or base) to the field
  //   width, except when applied to an infinity or NaN.
  // Obviously it makes no sense, but should it be allowed or is it a format
  // error?
  test({.alignment = _Flags::_Alignment::__default, .zero_padding = true}, 1, CSTR("0}"));
  test({.alignment = _Flags::_Alignment::__left, .zero_padding = false}, 2, CSTR("<0}"));
  test({.alignment = _Flags::_Alignment::__center, .zero_padding = false}, 2, CSTR("^0}"));
  test({.alignment = _Flags::_Alignment::__right, .zero_padding = false}, 2, CSTR(">0}"));

  // *** Width ***
  test({.width = 0, .width_as_arg = false}, 0, CSTR("}"));
  test({.width = 1, .width_as_arg = false}, 1, CSTR("1}"));
  test({.width = 10, .width_as_arg = false}, 2, CSTR("10}"));
  test({.width = 1000, .width_as_arg = false}, 4, CSTR("1000}"));
  test({.width = 1000000, .width_as_arg = false}, 7, CSTR("1000000}"));

  test({.width = 0, .width_as_arg = true}, 2, CSTR("{}}"));
  test({.width = 0, .width_as_arg = true}, 3, CSTR("{0}}"));
  test({.width = 1, .width_as_arg = true}, 3, CSTR("{1}}"));

  test_exception<Parser<CharT>>("A format-spec width field shouldn't have a leading zero", CSTR("00"));

  static_assert(std::__format::__number_max == 2'147'483'647, "Update the assert and the test.");
  test({.width = 2'147'483'647, .width_as_arg = false}, 10, CSTR("2147483647}"));
  test_exception<Parser<CharT>>("The numeric value of the format-spec is too large", CSTR("2147483648"));
  test_exception<Parser<CharT>>("The numeric value of the format-spec is too large", CSTR("5000000000"));
  test_exception<Parser<CharT>>("The numeric value of the format-spec is too large", CSTR("10000000000"));

  test_exception<Parser<CharT>>("End of input while parsing format-spec arg-id", CSTR("{"));
  test_exception<Parser<CharT>>("Invalid arg-id", CSTR("{0"));
  test_exception<Parser<CharT>>("The arg-id of the format-spec starts with an invalid character", CSTR("{a"));
  test_exception<Parser<CharT>>("Invalid arg-id", CSTR("{1"));
  test_exception<Parser<CharT>>("Invalid arg-id", CSTR("{9"));
  test_exception<Parser<CharT>>("Invalid arg-id", CSTR("{9:"));
  test_exception<Parser<CharT>>("Invalid arg-id", CSTR("{9a"));
  static_assert(std::__format::__number_max == 2'147'483'647, "Update the assert and the test.");
  // Note the static_assert tests whether the arg-id is valid.
  // Therefore the following should be true arg-id < __format::__number_max.
  test({.width = 2'147'483'646, .width_as_arg = true}, 12, CSTR("{2147483646}}"));
  test_exception<Parser<CharT>>("The numeric value of the format-spec is too large", CSTR("{2147483648}"));
  test_exception<Parser<CharT>>("The numeric value of the format-spec is too large", CSTR("{5000000000}"));
  test_exception<Parser<CharT>>("The numeric value of the format-spec is too large", CSTR("{10000000000}"));

  // *** Precision ***
  test({.precision = 0, .precision_as_arg = false, .type = _Flags::_Type::__general_lower_case}, 2, CSTR(".0}"));
  test({.precision = 1, .precision_as_arg = false, .type = _Flags::_Type::__general_lower_case}, 2, CSTR(".1}"));
  test({.precision = 10, .precision_as_arg = false, .type = _Flags::_Type::__general_lower_case}, 3, CSTR(".10}"));
  test({.precision = 1000, .precision_as_arg = false, .type = _Flags::_Type::__general_lower_case}, 5, CSTR(".1000}"));
  test({.precision = 1000000, .precision_as_arg = false, .type = _Flags::_Type::__general_lower_case}, 8,
       CSTR(".1000000}"));

  test({.precision = 0, .precision_as_arg = true, .type = _Flags::_Type::__general_lower_case}, 3, CSTR(".{}}"));
  test({.precision = 0, .precision_as_arg = true, .type = _Flags::_Type::__general_lower_case}, 4, CSTR(".{0}}"));
  test({.precision = 1, .precision_as_arg = true, .type = _Flags::_Type::__general_lower_case}, 4, CSTR(".{1}}"));

  test_exception<Parser<CharT>>("The format-spec precision field doesn't contain a value or arg-id", CSTR(".a"));
  test_exception<Parser<CharT>>("The format-spec precision field doesn't contain a value or arg-id", CSTR(".:"));

  static_assert(std::__format::__number_max == 2'147'483'647, "Update the assert and the test.");
  test({.precision = 2'147'483'647, .precision_as_arg = false, .type = _Flags::_Type::__general_lower_case}, 11,
       CSTR(".2147483647}"));
  test_exception<Parser<CharT>>("The numeric value of the format-spec is too large", CSTR(".2147483648"));
  test_exception<Parser<CharT>>("The numeric value of the format-spec is too large", CSTR(".5000000000"));
  test_exception<Parser<CharT>>("The numeric value of the format-spec is too large", CSTR(".10000000000"));

  test_exception<Parser<CharT>>("End of input while parsing format-spec arg-id", CSTR(".{"));
  test_exception<Parser<CharT>>("Invalid arg-id", CSTR(".{0"));
  test_exception<Parser<CharT>>("The arg-id of the format-spec starts with an invalid character", CSTR(".{a"));
  test_exception<Parser<CharT>>("Invalid arg-id", CSTR(".{1"));
  test_exception<Parser<CharT>>("Invalid arg-id", CSTR(".{9"));
  test_exception<Parser<CharT>>("Invalid arg-id", CSTR(".{9:"));
  test_exception<Parser<CharT>>("Invalid arg-id", CSTR(".{9a"));

  static_assert(std::__format::__number_max == 2'147'483'647, "Update the assert and the test.");
  // Note the static_assert tests whether the arg-id is valid.
  // Therefore the following should be true arg-id < __format::__number_max.
  test({.precision = 2'147'483'646, .precision_as_arg = true, .type = _Flags::_Type::__general_lower_case}, 13,
       CSTR(".{2147483646}}"));
  test_exception<Parser<CharT>>("The numeric value of the format-spec is too large", CSTR(".{2147483648}"));
  test_exception<Parser<CharT>>("The numeric value of the format-spec is too large", CSTR(".{5000000000}"));
  test_exception<Parser<CharT>>("The numeric value of the format-spec is too large", CSTR(".{10000000000}"));

  // *** Width & Precision ***
  test({.width = 1,
        .width_as_arg = false,
        .precision = 0,
        .precision_as_arg = false,
        .type = _Flags::_Type::__general_lower_case},
       3, CSTR("1.0}"));
  test({.width = 0,
        .width_as_arg = true,
        .precision = 1,
        .precision_as_arg = true,
        .type = _Flags::_Type::__general_lower_case},
       5, CSTR("{}.{}}"));
  test({.width = 10,
        .width_as_arg = true,
        .precision = 9,
        .precision_as_arg = true,
        .type = _Flags::_Type::__general_lower_case},
       8, CSTR("{10}.{9}}"));

  // *** Locale-specific form ***
  test({.locale_specific_form = true}, 1, CSTR("L}"));

  // *** Type ***
  {
    const char* unsuported_type = "The format-spec type has a type not supported for a floating-point argument";
    const char* not_a_type = "The format-spec should consume the input or end with a '}'";

    test({.type = _Flags::_Type::__float_hexadecimal_upper_case}, 1, CSTR("A}"));
    test_exception<Parser<CharT>>(unsuported_type, CSTR("B}"));
    test_exception<Parser<CharT>>(not_a_type, CSTR("C}"));
    test_exception<Parser<CharT>>(not_a_type, CSTR("D}"));
    test({.precision = 6, .precision_as_arg = false, .type = _Flags::_Type::__scientific_upper_case}, 1, CSTR("E}"));
    test({.precision = 6, .precision_as_arg = false, .type = _Flags::_Type::__fixed_upper_case}, 1, CSTR("F}"));
    test({.precision = 6, .precision_as_arg = false, .type = _Flags::_Type::__general_upper_case}, 1, CSTR("G}"));
    test_exception<Parser<CharT>>(not_a_type, CSTR("H}"));
    test_exception<Parser<CharT>>(not_a_type, CSTR("I}"));
    test_exception<Parser<CharT>>(not_a_type, CSTR("J}"));
    test_exception<Parser<CharT>>(not_a_type, CSTR("K}"));
    test({.locale_specific_form = true}, 1, CSTR("L}"));
    test_exception<Parser<CharT>>(not_a_type, CSTR("M}"));
    test_exception<Parser<CharT>>(not_a_type, CSTR("N}"));
    test_exception<Parser<CharT>>(not_a_type, CSTR("O}"));
    test_exception<Parser<CharT>>(not_a_type, CSTR("P}"));
    test_exception<Parser<CharT>>(not_a_type, CSTR("Q}"));
    test_exception<Parser<CharT>>(not_a_type, CSTR("R}"));
    test_exception<Parser<CharT>>(not_a_type, CSTR("S}"));
    test_exception<Parser<CharT>>(not_a_type, CSTR("T}"));
    test_exception<Parser<CharT>>(not_a_type, CSTR("U}"));
    test_exception<Parser<CharT>>(not_a_type, CSTR("V}"));
    test_exception<Parser<CharT>>(not_a_type, CSTR("W}"));
    test_exception<Parser<CharT>>(unsuported_type, CSTR("X}"));
    test_exception<Parser<CharT>>(not_a_type, CSTR("Y}"));
    test_exception<Parser<CharT>>(not_a_type, CSTR("Z}"));

    test({.type = _Flags::_Type::__float_hexadecimal_lower_case}, 1, CSTR("a}"));
    test_exception<Parser<CharT>>(unsuported_type, CSTR("b}"));
    test_exception<Parser<CharT>>(unsuported_type, CSTR("c}"));
    test_exception<Parser<CharT>>(unsuported_type, CSTR("d}"));
    test({.precision = 6, .precision_as_arg = false, .type = _Flags::_Type::__scientific_lower_case}, 1, CSTR("e}"));
    test({.precision = 6, .precision_as_arg = false, .type = _Flags::_Type::__fixed_lower_case}, 1, CSTR("f}"));
    test({.precision = 6, .precision_as_arg = false, .type = _Flags::_Type::__general_lower_case}, 1, CSTR("g}"));
    test_exception<Parser<CharT>>(not_a_type, CSTR("h}"));
    test_exception<Parser<CharT>>(not_a_type, CSTR("i}"));
    test_exception<Parser<CharT>>(not_a_type, CSTR("j}"));
    test_exception<Parser<CharT>>(not_a_type, CSTR("k}"));
    test_exception<Parser<CharT>>(not_a_type, CSTR("l}"));
    test_exception<Parser<CharT>>(not_a_type, CSTR("m}"));
    test_exception<Parser<CharT>>(not_a_type, CSTR("n}"));
    test_exception<Parser<CharT>>(unsuported_type, CSTR("o}"));
    test_exception<Parser<CharT>>(unsuported_type, CSTR("p}"));
    test_exception<Parser<CharT>>(not_a_type, CSTR("q}"));
    test_exception<Parser<CharT>>(not_a_type, CSTR("r}"));
    test_exception<Parser<CharT>>(unsuported_type, CSTR("s}"));
    test_exception<Parser<CharT>>(not_a_type, CSTR("t}"));
    test_exception<Parser<CharT>>(not_a_type, CSTR("u}"));
    test_exception<Parser<CharT>>(not_a_type, CSTR("v}"));
    test_exception<Parser<CharT>>(not_a_type, CSTR("w}"));
    test_exception<Parser<CharT>>(unsuported_type, CSTR("x}"));
    test_exception<Parser<CharT>>(not_a_type, CSTR("y}"));
    test_exception<Parser<CharT>>(not_a_type, CSTR("z}"));
  }
  // **** General ***
  test_exception<Parser<CharT>>("The format-spec should consume the input or end with a '}'", CSTR("ss"));
}

constexpr bool test() {
  test<char>();
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  test<wchar_t>();
#endif

  return true;
}

int main(int, char**) {
#if !defined(_WIN32) && !defined(_AIX)
  // Make sure the parsers match the expectations. The layout of the
  // subobjects is chosen to minimize the size required.
  static_assert(sizeof(Parser<char>) == 3 * sizeof(uint32_t));
#  ifndef TEST_HAS_NO_WIDE_CHARACTERS
  static_assert(sizeof(Parser<wchar_t>) == (sizeof(wchar_t) <= 2 ? 3 * sizeof(uint32_t) : 4 * sizeof(uint32_t)));
#  endif
#endif

  test();
  static_assert(test());

  return 0;
}
