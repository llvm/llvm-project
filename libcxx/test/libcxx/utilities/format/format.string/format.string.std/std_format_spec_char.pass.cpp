//===----------------------------------------------------------------------===//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: libcpp-no-concepts
// UNSUPPORTED: libcpp-has-no-incomplete-format
// XFAIL: LIBCXX-AIX-FIXME

// <format>

// Tests the parsing of the format string as specified in [format.string.std].
// It validates whether the std-format-spec is valid for a char type.

#include <format>
#include <cassert>
#ifndef _LIBCPP_HAS_NO_LOCALIZATION
# include <iostream>
#endif

#include "concepts_precision.h"
#include "test_macros.h"
#include "make_string.h"
#include "test_exception.h"

#define CSTR(S) MAKE_CSTRING(CharT, S)

using namespace std::__format_spec;

template <class CharT>
using Parser = __parser_char<CharT>;

template <class CharT>
struct Expected {
  CharT fill = CharT(' ');
  _Flags::_Alignment alignment = _Flags::_Alignment::__left;
  _Flags::_Sign sign = _Flags::_Sign::__default;
  bool alternate_form = false;
  bool zero_padding = false;
  uint32_t width = 0;
  bool width_as_arg = false;
  bool locale_specific_form = false;
  _Flags::_Type type = _Flags::_Type::__char;
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
  assert(parser.__sign == expected.sign);
  assert(parser.__alternate_form == expected.alternate_form);
  assert(parser.__zero_padding == expected.zero_padding);
  assert(parser.__width == expected.width);
  assert(parser.__width_as_arg == expected.width_as_arg);
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
constexpr void test_as_char() {

  test({}, 1, CSTR("c}"));

  // *** Align-fill ***
  test({.alignment = _Flags::_Alignment::__left}, 1, CSTR("<}"));
  test({.alignment = _Flags::_Alignment::__center}, 1, "^}");
  test({.alignment = _Flags::_Alignment::__right}, 1, ">}");

  test({.alignment = _Flags::_Alignment::__left}, 2, CSTR("<c}"));
  test({.alignment = _Flags::_Alignment::__center}, 2, "^c}");
  test({.alignment = _Flags::_Alignment::__right}, 2, ">c}");

  test({.fill = CharT('L'), .alignment = _Flags::_Alignment::__left}, 2,
       CSTR("L<}"));
  test({.fill = CharT('#'), .alignment = _Flags::_Alignment::__center}, 2,
       CSTR("#^}"));
  test({.fill = CharT('0'), .alignment = _Flags::_Alignment::__right}, 2,
       CSTR("0>}"));

  test({.fill = CharT('L'), .alignment = _Flags::_Alignment::__left}, 3,
       CSTR("L<c}"));
  test({.fill = CharT('#'), .alignment = _Flags::_Alignment::__center}, 3,
       CSTR("#^c}"));
  test({.fill = CharT('0'), .alignment = _Flags::_Alignment::__right}, 3,
       CSTR("0>c}"));

  // *** Sign ***
  test_exception<Parser<CharT>>(
      "A sign field isn't allowed in this format-spec", CSTR("-"));
  test_exception<Parser<CharT>>(
      "A sign field isn't allowed in this format-spec", CSTR("+"));
  test_exception<Parser<CharT>>(
      "A sign field isn't allowed in this format-spec", CSTR(" "));

  test_exception<Parser<CharT>>(
      "A sign field isn't allowed in this format-spec", CSTR("-c"));
  test_exception<Parser<CharT>>(
      "A sign field isn't allowed in this format-spec", CSTR("+c"));
  test_exception<Parser<CharT>>(
      "A sign field isn't allowed in this format-spec", CSTR(" c"));

  // *** Alternate form ***
  test_exception<Parser<CharT>>(
      "An alternate form field isn't allowed in this format-spec", CSTR("#}"));
  test_exception<Parser<CharT>>(
      "An alternate form field isn't allowed in this format-spec", CSTR("#c}"));

  // *** Zero padding ***
  test_exception<Parser<CharT>>(
      "A zero-padding field isn't allowed in this format-spec", CSTR("0}"));
  test_exception<Parser<CharT>>(
      "A zero-padding field isn't allowed in this format-spec", CSTR("0c}"));

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
  test_exception<Parser<CharT>>("Invalid arg-id", CSTR("{0"));
  test_exception<Parser<CharT>>(
      "The arg-id of the format-spec starts with an invalid character",
      CSTR("{a"));
  test_exception<Parser<CharT>>("Invalid arg-id", CSTR("{1"));
  test_exception<Parser<CharT>>("Invalid arg-id", CSTR("{9"));
  test_exception<Parser<CharT>>("Invalid arg-id", CSTR("{9:"));
  test_exception<Parser<CharT>>("Invalid arg-id", CSTR("{9a"));

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
  test_exception<Parser<CharT>>(
      "The format-spec should consume the input or end with a '}'", CSTR("."));
  test_exception<Parser<CharT>>(
      "The format-spec should consume the input or end with a '}'", CSTR(".1"));

  // *** Locale-specific form ***
  // Note the flag is allowed, but has no effect.
  test({.locale_specific_form = true}, 1, CSTR("L}"));
  test({.locale_specific_form = true}, 2, CSTR("Lc}"));
}

template <class CharT>
constexpr void test_as_integer() {

  test({.alignment = _Flags::_Alignment::__right,
        .type = _Flags::_Type::__decimal},
       1, CSTR("d}"));

  // *** Align-fill ***
  test({.alignment = _Flags::_Alignment::__left,
        .type = _Flags::_Type::__decimal},
       2, CSTR("<d}"));
  test({.alignment = _Flags::_Alignment::__center,
        .type = _Flags::_Type::__decimal},
       2, "^d}");
  test({.alignment = _Flags::_Alignment::__right,
        .type = _Flags::_Type::__decimal},
       2, ">d}");

  test({.fill = CharT('L'),
        .alignment = _Flags::_Alignment::__left,
        .type = _Flags::_Type::__decimal},
       3, CSTR("L<d}"));
  test({.fill = CharT('#'),
        .alignment = _Flags::_Alignment::__center,
        .type = _Flags::_Type::__decimal},
       3, CSTR("#^d}"));
  test({.fill = CharT('0'),
        .alignment = _Flags::_Alignment::__right,
        .type = _Flags::_Type::__decimal},
       3, CSTR("0>d}"));

  // *** Sign ***
  test({.alignment = _Flags::_Alignment::__right,
        .sign = _Flags::_Sign::__minus,
        .type = _Flags::_Type::__decimal},
       2, CSTR("-d}"));
  test({.alignment = _Flags::_Alignment::__right,
        .sign = _Flags::_Sign::__plus,
        .type = _Flags::_Type::__decimal},
       2, CSTR("+d}"));
  test({.alignment = _Flags::_Alignment::__right,
        .sign = _Flags::_Sign::__space,
        .type = _Flags::_Type::__decimal},
       2, CSTR(" d}"));

  // *** Alternate form ***
  test({.alignment = _Flags::_Alignment::__right,
        .alternate_form = true,
        .type = _Flags::_Type::__decimal},
       2, CSTR("#d}"));

  // *** Zero padding ***
  test({.alignment = _Flags::_Alignment::__default,
        .zero_padding = true,
        .type = _Flags::_Type::__decimal},
       2, CSTR("0d}"));
  test({.alignment = _Flags::_Alignment::__center,
        .type = _Flags::_Type::__decimal},
       3, CSTR("^0d}"));

  // *** Width ***
  test({.alignment = _Flags::_Alignment::__right,
        .width = 0,
        .width_as_arg = false,
        .type = _Flags::_Type::__decimal},
       1, CSTR("d}"));
  test({.alignment = _Flags::_Alignment::__right,
        .width = 1,
        .width_as_arg = false,
        .type = _Flags::_Type::__decimal},
       2, CSTR("1d}"));
  test({.alignment = _Flags::_Alignment::__right,
        .width = 10,
        .width_as_arg = false,
        .type = _Flags::_Type::__decimal},
       3, CSTR("10d}"));
  test({.alignment = _Flags::_Alignment::__right,
        .width = 1000,
        .width_as_arg = false,
        .type = _Flags::_Type::__decimal},
       5, CSTR("1000d}"));
  test({.alignment = _Flags::_Alignment::__right,
        .width = 1000000,
        .width_as_arg = false,
        .type = _Flags::_Type::__decimal},
       8, CSTR("1000000d}"));

  test({.alignment = _Flags::_Alignment::__right,
        .width = 0,
        .width_as_arg = true,
        .type = _Flags::_Type::__decimal},
       3, CSTR("{}d}"));
  test({.alignment = _Flags::_Alignment::__right,
        .width = 0,
        .width_as_arg = true,
        .type = _Flags::_Type::__decimal},
       4, CSTR("{0}d}"));
  test({.alignment = _Flags::_Alignment::__right,
        .width = 1,
        .width_as_arg = true,
        .type = _Flags::_Type::__decimal},
       4, CSTR("{1}d}"));

  // *** Precision ***
  test_exception<Parser<CharT>>(
      "The format-spec should consume the input or end with a '}'", CSTR("."));
  test_exception<Parser<CharT>>(
      "The format-spec should consume the input or end with a '}'", CSTR(".1"));

  // *** Locale-specific form ***
  test({.alignment = _Flags::_Alignment::__right,
        .locale_specific_form = true,
        .type = _Flags::_Type::__decimal},
       2, CSTR("Ld}"));
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
  static_assert(!has_precision<decltype(parser)>);
  static_assert(!has_precision_as_arg<decltype(parser)>);
  assert(parser.__locale_specific_form == false);
  assert(parser.__type == _Flags::_Type::__default);

  test({}, 0, CSTR("}"));

  test_as_char<CharT>();
  test_as_integer<CharT>();

  // *** Type ***
  {
    const char* unsuported_type =
        "The format-spec type has a type not supported for a char argument";
    const char* not_a_type =
        "The format-spec should consume the input or end with a '}'";

    test_exception<Parser<CharT>>(unsuported_type, CSTR("A}"));
    test({.alignment = _Flags::_Alignment::__right,
          .type = _Flags::_Type::__binary_upper_case},
         1, CSTR("B}"));
    test_exception<Parser<CharT>>(not_a_type, CSTR("C}"));
    test_exception<Parser<CharT>>(not_a_type, CSTR("D}"));
    test_exception<Parser<CharT>>(unsuported_type, CSTR("E}"));
    test_exception<Parser<CharT>>(unsuported_type, CSTR("F}"));
    test_exception<Parser<CharT>>(unsuported_type, CSTR("G}"));
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
    test({.alignment = _Flags::_Alignment::__right,
          .type = _Flags::_Type::__hexadecimal_upper_case},
         1, CSTR("X}"));
    test_exception<Parser<CharT>>(not_a_type, CSTR("Y}"));
    test_exception<Parser<CharT>>(not_a_type, CSTR("Z}"));

    test_exception<Parser<CharT>>(unsuported_type, CSTR("a}"));
    test({.alignment = _Flags::_Alignment::__right,
          .type = _Flags::_Type::__binary_lower_case},
         1, CSTR("b}"));
    test({.type = _Flags::_Type::__char}, 1, CSTR("c}"));
    test({.alignment = _Flags::_Alignment::__right,
          .type = _Flags::_Type::__decimal},
         1, CSTR("d}"));
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
    test({.alignment = _Flags::_Alignment::__right,
          .type = _Flags::_Type::__octal},
         1, CSTR("o}"));
    test_exception<Parser<CharT>>(unsuported_type, CSTR("p}"));
    test_exception<Parser<CharT>>(not_a_type, CSTR("q}"));
    test_exception<Parser<CharT>>(not_a_type, CSTR("r}"));
    test_exception<Parser<CharT>>(unsuported_type, CSTR("s}"));
    test_exception<Parser<CharT>>(not_a_type, CSTR("t}"));
    test_exception<Parser<CharT>>(not_a_type, CSTR("u}"));
    test_exception<Parser<CharT>>(not_a_type, CSTR("v}"));
    test_exception<Parser<CharT>>(not_a_type, CSTR("w}"));
    test({.alignment = _Flags::_Alignment::__right,
          .type = _Flags::_Type::__hexadecimal_lower_case},
         1, CSTR("x}"));
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

  return true;
}

int main(int, char**) {
#ifndef _WIN32
  // TODO FMT Investigate why this doesn't work.
  // (Wait until LWG-3576 has been resolved.)
  // Make sure the parsers match the expectations. The layout of the
  // subobjects is chosen to minimize the size required.
  static_assert(sizeof(Parser<char>) == 2 * sizeof(uint32_t));
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  static_assert(
      sizeof(Parser<wchar_t>) ==
      (sizeof(wchar_t) <= 2 ? 2 * sizeof(uint32_t) : 3 * sizeof(uint32_t)));
#endif
#endif // _WIN32

  test();
  static_assert(test());

  return 0;
}
