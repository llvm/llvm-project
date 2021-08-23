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
// It validates whether the std-format-spec is valid for a string type.

#include <format>
#include <cassert>
#ifndef _LIBCPP_HAS_NO_LOCALIZATION
# include <iostream>
#endif

#include "test_macros.h"
#include "make_string.h"
#include "test_exception.h"

#define CSTR(S) MAKE_CSTRING(CharT, S)

using namespace std::__format_spec;

template <class CharT>
using Parser = __parser_string<CharT>;

template <class CharT>
struct Expected {
  CharT fill = CharT(' ');
  _Flags::_Alignment alignment = _Flags::_Alignment::__left;
  uint32_t width = 0;
  bool width_as_arg = false;
  uint32_t precision = std::__format::__number_max;
  bool precision_as_arg = true;
  _Flags::_Type type = _Flags::_Type::__default;
};

template <class CharT>
constexpr void test(Expected<CharT> expected, size_t size,
                    std::basic_string_view<CharT> fmt) {
  // Initialize parser with sufficient arguments to avoid the parsing to fail
  // due to insufficient arguments.
  std::basic_format_parse_context<CharT> parse_ctx(fmt,
                                                   std::__format::__number_max);
  auto begin = parse_ctx.begin();
  auto end = parse_ctx.end();
  Parser<CharT> parser;
  auto it = parser.parse(parse_ctx);

  assert(begin == parse_ctx.begin());
  assert(end == parse_ctx.end());

  assert(begin + size == it);
  assert(parser.__fill == expected.fill);
  assert(parser.__alignment == expected.alignment);
  assert(parser.__sign == _Flags::_Sign::__default);
  assert(parser.__alternate_form == false);
  assert(parser.__zero_padding == false);
  assert(parser.__width == expected.width);
  assert(parser.__width_as_arg == expected.width_as_arg);
  assert(parser.__precision == expected.precision);
  assert(parser.__precision_as_arg == expected.precision_as_arg);
  assert(parser.__locale_specific_form == false);
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
  assert(parser.__alignment == _Flags::_Alignment::__left);
  assert(parser.__sign == _Flags::_Sign::__default);
  assert(parser.__alternate_form == false);
  assert(parser.__zero_padding == false);
  assert(parser.__width == 0);
  assert(parser.__width_as_arg == false);
  assert(parser.__precision == std::__format::__number_max);
  assert(parser.__precision_as_arg == true);
  assert(parser.__locale_specific_form == false);
  assert(parser.__type == _Flags::_Type::__default);

  test({}, 0, CSTR("}"));

  // *** Align-fill ***
  test({.alignment = _Flags::_Alignment::__left}, 1, CSTR("<}"));
  test({.alignment = _Flags::_Alignment::__center}, 1, "^}");
  test({.alignment = _Flags::_Alignment::__right}, 1, ">}");

  test({.fill = CharT('L'), .alignment = _Flags::_Alignment::__left}, 2,
       CSTR("L<}"));
  test({.fill = CharT('#'), .alignment = _Flags::_Alignment::__center}, 2,
       CSTR("#^}"));
  test({.fill = CharT('0'), .alignment = _Flags::_Alignment::__right}, 2,
       CSTR("0>}"));

  test_exception<Parser<CharT>>(
      "The format-spec fill field contains an invalid character", CSTR("{<"));
  test_exception<Parser<CharT>>(
      "The format-spec fill field contains an invalid character", CSTR("}<"));

  // *** Sign ***
  test_exception<Parser<CharT>>(
      "The format-spec should consume the input or end with a '}'", CSTR("+"));
  test_exception<Parser<CharT>>(
      "The format-spec should consume the input or end with a '}'", CSTR("-"));
  test_exception<Parser<CharT>>(
      "The format-spec should consume the input or end with a '}'", CSTR(" "));

  // *** Alternate form ***
  test_exception<Parser<CharT>>(
      "The format-spec should consume the input or end with a '}'", CSTR("#"));

  // *** Zero padding ***
  test_exception<Parser<CharT>>(
      "A format-spec width field shouldn't have a leading zero", CSTR("0"));

  // *** Width ***
  test({.width = 0, .width_as_arg = false}, 0, CSTR("}"));
  test({.width = 1, .width_as_arg = false}, 1, CSTR("1}"));
  test({.width = 10, .width_as_arg = false}, 2, CSTR("10}"));
  test({.width = 1000, .width_as_arg = false}, 4, CSTR("1000}"));
  test({.width = 1000000, .width_as_arg = false}, 7, CSTR("1000000}"));

  test({.width = 0, .width_as_arg = true}, 2, CSTR("{}}"));
  test({.width = 0, .width_as_arg = true}, 3, CSTR("{0}}"));
  test({.width = 1, .width_as_arg = true}, 3, CSTR("{1}}"));

  test_exception<Parser<CharT>>(
      "A format-spec width field shouldn't have a leading zero", CSTR("00"));

  static_assert(std::__format::__number_max == 2'147'483'647,
                "Update the assert and the test.");
  test({.width = 2'147'483'647, .width_as_arg = false}, 10,
       CSTR("2147483647}"));
  test_exception<Parser<CharT>>(
      "The numeric value of the format-spec is too large", CSTR("2147483648"));
  test_exception<Parser<CharT>>(
      "The numeric value of the format-spec is too large", CSTR("5000000000"));
  test_exception<Parser<CharT>>(
      "The numeric value of the format-spec is too large", CSTR("10000000000"));

  test_exception<Parser<CharT>>("End of input while parsing format-spec arg-id",
                                CSTR("{"));
  test_exception<Parser<CharT>>(
      "A format-spec arg-id should terminate at a '}'", CSTR("{0"));
  test_exception<Parser<CharT>>(
      "The arg-id of the format-spec starts with an invalid character",
      CSTR("{a"));
  test_exception<Parser<CharT>>(
      "A format-spec arg-id should terminate at a '}'", CSTR("{1"));
  test_exception<Parser<CharT>>(
      "A format-spec arg-id should terminate at a '}'", CSTR("{9"));
  test_exception<Parser<CharT>>(
      "A format-spec arg-id should terminate at a '}'", CSTR("{9:"));
  test_exception<Parser<CharT>>(
      "A format-spec arg-id should terminate at a '}'", CSTR("{9a"));

  static_assert(std::__format::__number_max == 2'147'483'647,
                "Update the assert and the test.");
  // Note the static_assert tests whether the arg-id is valid.
  // Therefore the following should be true arg-id < __format::__number_max.
  test({.width = 2'147'483'646, .width_as_arg = true}, 12,
       CSTR("{2147483646}}"));
  test_exception<Parser<CharT>>(
      "The numeric value of the format-spec is too large",
      CSTR("{2147483648}"));
  test_exception<Parser<CharT>>(
      "The numeric value of the format-spec is too large",
      CSTR("{5000000000}"));
  test_exception<Parser<CharT>>(
      "The numeric value of the format-spec is too large",
      CSTR("{10000000000}"));

  // *** Precision ***
  test({.precision = 0, .precision_as_arg = false}, 2, CSTR(".0}"));
  test({.precision = 1, .precision_as_arg = false}, 2, CSTR(".1}"));
  test({.precision = 10, .precision_as_arg = false}, 3, CSTR(".10}"));
  test({.precision = 1000, .precision_as_arg = false}, 5, CSTR(".1000}"));
  test({.precision = 1000000, .precision_as_arg = false}, 8, CSTR(".1000000}"));

  test({.precision = 0, .precision_as_arg = true}, 3, CSTR(".{}}"));
  test({.precision = 0, .precision_as_arg = true}, 4, CSTR(".{0}}"));
  test({.precision = 1, .precision_as_arg = true}, 4, CSTR(".{1}}"));

  test_exception<Parser<CharT>>(
      "A format-spec precision field shouldn't have a leading zero",
      CSTR(".00"));
  test_exception<Parser<CharT>>(
      "A format-spec precision field shouldn't have a leading zero",
      CSTR(".01"));
  test_exception<Parser<CharT>>(
      "The format-spec precision field doesn't contain a value or arg-id",
      CSTR(".a"));
  test_exception<Parser<CharT>>(
      "The format-spec precision field doesn't contain a value or arg-id",
      CSTR(".:"));

  static_assert(std::__format::__number_max == 2'147'483'647,
                "Update the assert and the test.");
  test({.precision = 2'147'483'647, .precision_as_arg = false}, 11,
       CSTR(".2147483647}"));
  test_exception<Parser<CharT>>(
      "The numeric value of the format-spec is too large", CSTR(".2147483648"));
  test_exception<Parser<CharT>>(
      "The numeric value of the format-spec is too large", CSTR(".5000000000"));
  test_exception<Parser<CharT>>(
      "The numeric value of the format-spec is too large",
      CSTR(".10000000000"));

  test_exception<Parser<CharT>>("End of input while parsing format-spec arg-id",
                                CSTR(".{"));
  test_exception<Parser<CharT>>(
      "A format-spec arg-id should terminate at a '}'", CSTR(".{0"));
  test_exception<Parser<CharT>>(
      "The arg-id of the format-spec starts with an invalid character",
      CSTR(".{a"));
  test_exception<Parser<CharT>>(
      "A format-spec arg-id should terminate at a '}'", CSTR(".{1"));
  test_exception<Parser<CharT>>(
      "A format-spec arg-id should terminate at a '}'", CSTR(".{9"));
  test_exception<Parser<CharT>>(
      "A format-spec arg-id should terminate at a '}'", CSTR(".{9:"));
  test_exception<Parser<CharT>>(
      "A format-spec arg-id should terminate at a '}'", CSTR(".{9a"));

  static_assert(std::__format::__number_max == 2'147'483'647,
                "Update the assert and the test.");
  // Note the static_assert tests whether the arg-id is valid.
  // Therefore the following should be true arg-id < __format::__number_max.
  test({.precision = 2'147'483'646, .precision_as_arg = true}, 13,
       CSTR(".{2147483646}}"));
  test_exception<Parser<CharT>>(
      "The numeric value of the format-spec is too large",
      CSTR(".{2147483648}"));
  test_exception<Parser<CharT>>(
      "The numeric value of the format-spec is too large",
      CSTR(".{5000000000}"));
  test_exception<Parser<CharT>>(
      "The numeric value of the format-spec is too large",
      CSTR(".{10000000000}"));

  // *** Width & Precision ***
  test({.width = 1,
        .width_as_arg = false,
        .precision = 0,
        .precision_as_arg = false},
       3, CSTR("1.0}"));
  test({.width = 0,
        .width_as_arg = true,
        .precision = 1,
        .precision_as_arg = true},
       5, CSTR("{}.{}}"));
  test({.width = 10,
        .width_as_arg = true,
        .precision = 9,
        .precision_as_arg = true},
       8, CSTR("{10}.{9}}"));

  // *** Locale-specific form ***
  test_exception<Parser<CharT>>(
      "The format-spec should consume the input or end with a '}'", CSTR("L"));

  // *** Type ***

  {
    const char* unsuported_type =
        "The format-spec type has a type not supported for a string argument";
    const char* not_a_type =
        "The format-spec should consume the input or end with a '}'";

    test_exception<Parser<CharT>>(unsuported_type, CSTR("A}"));
    test_exception<Parser<CharT>>(unsuported_type, CSTR("B}"));
    test_exception<Parser<CharT>>(not_a_type, CSTR("C}"));
    test_exception<Parser<CharT>>(not_a_type, CSTR("D}"));
    test_exception<Parser<CharT>>(unsuported_type, CSTR("E}"));
    test_exception<Parser<CharT>>(unsuported_type, CSTR("F}"));
    test_exception<Parser<CharT>>(unsuported_type, CSTR("G}"));
    test_exception<Parser<CharT>>(not_a_type, CSTR("H}"));
    test_exception<Parser<CharT>>(not_a_type, CSTR("I}"));
    test_exception<Parser<CharT>>(not_a_type, CSTR("J}"));
    test_exception<Parser<CharT>>(not_a_type, CSTR("K}"));
    test_exception<Parser<CharT>>(not_a_type, CSTR("L}"));
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

    test_exception<Parser<CharT>>(unsuported_type, CSTR("a}"));
    test_exception<Parser<CharT>>(unsuported_type, CSTR("b}"));
    test_exception<Parser<CharT>>(unsuported_type, CSTR("c}"));
    test_exception<Parser<CharT>>(unsuported_type, CSTR("d}"));
    test_exception<Parser<CharT>>(unsuported_type, CSTR("e}"));
    test_exception<Parser<CharT>>(unsuported_type, CSTR("f}"));
    test_exception<Parser<CharT>>(unsuported_type, CSTR("g}"));
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
    test({.type = _Flags::_Type::__string}, 1, CSTR("s}"));
    test_exception<Parser<CharT>>(not_a_type, CSTR("t}"));
    test_exception<Parser<CharT>>(not_a_type, CSTR("u}"));
    test_exception<Parser<CharT>>(not_a_type, CSTR("v}"));
    test_exception<Parser<CharT>>(not_a_type, CSTR("w}"));
    test_exception<Parser<CharT>>(unsuported_type, CSTR("x}"));
    test_exception<Parser<CharT>>(not_a_type, CSTR("y}"));
    test_exception<Parser<CharT>>(not_a_type, CSTR("z}"));
  }
  // **** General ***
  test_exception<Parser<CharT>>(
      "The format-spec should consume the input or end with a '}'", CSTR("ss"));
}

constexpr bool test() {
  test<char>();
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  test<wchar_t>();
#endif
#ifndef _LIBCPP_HAS_NO_CHAR8_T
  test<char8_t>();
#endif
#ifndef _LIBCPP_HAS_NO_UNICODE_CHARS
  test<char16_t>();
  test<char32_t>();
#endif

  return true;
}

int main(int, char**) {
#ifndef _WIN32
  // Make sure the parsers match the expectations. The layout of the
  // subobjects is chosen to minimize the size required.
  LIBCPP_STATIC_ASSERT(sizeof(Parser<char>) == 3 * sizeof(uint32_t));
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  LIBCPP_STATIC_ASSERT(
      sizeof(Parser<wchar_t>) ==
      (sizeof(wchar_t) <= 2 ? 3 * sizeof(uint32_t) : 4 * sizeof(uint32_t)));
#endif
#endif // _WIN32

  test();
  static_assert(test());

  return 0;
}
