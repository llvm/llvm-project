//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// ADDITIONAL_COMPILE_FLAGS: -funsigned-char

// <format>

// C++23 the formatter is a debug-enabled specialization.
// [format.formatter.spec]:
// Each header that declares the template `formatter` provides the following
// enabled specializations:
//   template<> struct formatter<char, char>;
//   template<> struct formatter<char, wchar_t>;
//   template<> struct formatter<wchar_t, wchar_t>;

// P2909R4 "Fix formatting of code units as integers (Dude, where’s my char?)"
// changed the behaviour of char (and wchar_t) when their underlying type is signed.

#include <format>
#include <cassert>
#include <concepts>
#include <iterator>
#include <type_traits>

#include "test_format_context.h"
#include "test_macros.h"
#include "make_string.h"
#include "assert_macros.h"
#include "concat_macros.h"

#define STR(S) MAKE_STRING(CharT, S)
#define SV(S) MAKE_STRING_VIEW(CharT, S)

template <class StringT, class StringViewT, class ArgumentT>
void test(StringT expected, StringViewT fmt, ArgumentT arg) {
  using CharT    = typename StringT::value_type;
  auto parse_ctx = std::basic_format_parse_context<CharT>(fmt);
  std::formatter<ArgumentT, CharT> formatter;
  static_assert(std::semiregular<decltype(formatter)>);

  formatter.parse(parse_ctx);

  StringT result;
  auto out         = std::back_inserter(result);
  using FormatCtxT = std::basic_format_context<decltype(out), CharT>;

  FormatCtxT format_ctx = test_format_context_create<decltype(out), CharT>(out, std::make_format_args<FormatCtxT>(arg));
  formatter.format(arg, format_ctx);
  TEST_REQUIRE(result == expected,
               TEST_WRITE_CONCATENATED(
                   "\nFormat string   ", fmt, "\nExpected output ", expected, "\nActual output   ", result, '\n'));
}

template <class CharT>
void test() {
  test(STR("\x00"), STR("}"), '\x00');
  test(STR("a"), STR("}"), 'a');
  test(STR("\x80"), STR("}"), '\x80');
  test(STR("\xff"), STR("}"), '\xff');

  test(STR("\x00"), STR("c}"), '\x00');
  test(STR("a"), STR("c}"), 'a');
  test(STR("\x80"), STR("c}"), '\x80');
  test(STR("\xff"), STR("c}"), '\xff');

#if TEST_STD_VER > 20
  test(STR(R"('\u{0}')"), STR("?}"), '\x00');
  test(STR("'a'"), STR("?}"), 'a');
#  ifndef TEST_HAS_NO_UNICODE
  if constexpr (std::same_as<CharT, char>) {
    test(STR(R"('\x{80}')"), STR("?}"), '\x80');
    test(STR(R"('\x{ff}')"), STR("?}"), '\xff');
  }
#    ifndef TEST_HAS_NO_WIDE_CHARACTERS
  else {
    test(STR(R"('\u{80}')"), STR("?}"), '\x80');
    test(STR("'\u00ff'"), STR("?}"), '\xff');
  }
#    endif // TEST_HAS_NO_WIDE_CHARACTERS
#  endif   // TEST_HAS_NO_UNICODE
#endif   // TEST_STD_VER > 20

  test(STR("10000000"), STR("b}"), char(128));
  test(STR("11111111"), STR("b}"), char(255));
  test(STR("0"), STR("b}"), char(0));
  test(STR("1"), STR("b}"), char(1));
  test(STR("1111111"), STR("b}"), char(127));

  test(STR("10000000"), STR("B}"), char(128));
  test(STR("11111111"), STR("B}"), char(255));
  test(STR("0"), STR("B}"), char(0));
  test(STR("1"), STR("B}"), char(1));
  test(STR("1111111"), STR("B}"), char(127));

  test(STR("128"), STR("d}"), char(128));
  test(STR("255"), STR("d}"), char(255));
  test(STR("0"), STR("d}"), char(0));
  test(STR("1"), STR("d}"), char(1));
  test(STR("127"), STR("d}"), char(127));

  test(STR("200"), STR("o}"), char(128));
  test(STR("377"), STR("o}"), char(255));
  test(STR("0"), STR("o}"), char(0));
  test(STR("1"), STR("o}"), char(1));
  test(STR("177"), STR("o}"), char(127));

  test(STR("80"), STR("x}"), char(128));
  test(STR("ff"), STR("x}"), char(255));
  test(STR("0"), STR("x}"), char(0));
  test(STR("1"), STR("x}"), char(1));
  test(STR("7f"), STR("x}"), char(127));

  test(STR("80"), STR("X}"), char(128));
  test(STR("FF"), STR("X}"), char(255));
  test(STR("0"), STR("X}"), char(0));
  test(STR("1"), STR("X}"), char(1));
  test(STR("7F"), STR("X}"), char(127));
}

int main(int, char**) {
  test<char>();
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  test<wchar_t>();
#endif

  return 0;
}
