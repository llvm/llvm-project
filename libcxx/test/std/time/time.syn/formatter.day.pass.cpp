//===----------------------------------------------------------------------===//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: no-localization
// UNSUPPORTED: libcpp-has-no-incomplete-format

// TODO FMT Investigate Windows issues.
// UNSUPPORTED: msvc, target={{.+}}-windows-gnu

// REQUIRES: locale.fr_FR.UTF-8
// REQUIRES: locale.ja_JP.UTF-8

// <chrono>

// template<class charT> struct formatter<chrono::day, charT>;

#include <chrono>
#include <format>

#include <cassert>
#include <concepts>
#include <locale>
#include <iostream>
#include <type_traits>

#include "formatter_tests.h"
#include "make_string.h"
#include "platform_support.h" // locale name macros
#include "string_literal.h"
#include "test_macros.h"

template <class CharT>
static void test_no_chrono_specs() {
  using namespace std::literals::chrono_literals;

  // Valid day
  check(SV("01"), SV("{}"), 1d);
  check(SV("*01*"), SV("{:*^4}"), 1d);
  check(SV("*01"), SV("{:*>3}"), 1d);

  // Invalid day
  check(SV("00 is not a valid day"), SV("{}"), 0d);
  check(SV("*00 is not a valid day*"), SV("{:*^23}"), 0d);
}

template <class CharT>
static void test_valid_values() {
  using namespace std::literals::chrono_literals;

  constexpr std::basic_string_view<CharT> fmt  = SV("{:%%d='%d'%t%%Od='%Od'%t%%e='%e'%t%%Oe='%Oe'%n}");
  constexpr std::basic_string_view<CharT> lfmt = SV("{:L%%d='%d'%t%%Od='%Od'%t%%e='%e'%t%%Oe='%Oe'%n}");

  const std::locale loc(LOCALE_ja_JP_UTF_8);
  std::locale::global(std::locale(LOCALE_fr_FR_UTF_8));

  // Non localized output using C-locale
  check(SV("%d='00'\t%Od='00'\t%e=' 0'\t%Oe=' 0'\n"), fmt, 0d);
  check(SV("%d='01'\t%Od='01'\t%e=' 1'\t%Oe=' 1'\n"), fmt, 1d);
  check(SV("%d='31'\t%Od='31'\t%e='31'\t%Oe='31'\n"), fmt, 31d);
#if defined(_AIX)
  check(SV("%d='55'\t%Od='55'\t%e='55'\t%Oe='55'\n"), fmt, 255d);
#else
  check(SV("%d='255'\t%Od='255'\t%e='255'\t%Oe='255'\n"), fmt, 255d);
#endif

  // Use the global locale (fr_FR)
  check(SV("%d='00'\t%Od='00'\t%e=' 0'\t%Oe=' 0'\n"), lfmt, 0d);
  check(SV("%d='01'\t%Od='01'\t%e=' 1'\t%Oe=' 1'\n"), lfmt, 1d);
  check(SV("%d='31'\t%Od='31'\t%e='31'\t%Oe='31'\n"), lfmt, 31d);
#if defined(_AIX)
  check(SV("%d='55'\t%Od='55'\t%e='55'\t%Oe='55'\n"), lfmt, 255d);
#else
  check(SV("%d='255'\t%Od='255'\t%e='255'\t%Oe='255'\n"), lfmt, 255d);
#endif

  // Use supplied locale (ja_JP). This locale has a different alternate on some platforms.
#if defined(__APPLE__) || defined(_AIX)
  check(loc, SV("%d='00'\t%Od='00'\t%e=' 0'\t%Oe=' 0'\n"), lfmt, 0d);
  check(loc, SV("%d='01'\t%Od='01'\t%e=' 1'\t%Oe=' 1'\n"), lfmt, 1d);
  check(loc, SV("%d='31'\t%Od='31'\t%e='31'\t%Oe='31'\n"), lfmt, 31d);
#  if defined(_AIX)
  check(SV("%d='55'\t%Od='55'\t%e='55'\t%Oe='55'\n"), fmt, 255d);
#  else
  check(SV("%d='255'\t%Od='255'\t%e='255'\t%Oe='255'\n"), fmt, 255d);
#  endif
#else  // defined(__APPLE__) || defined(_AIX)
  check(loc, SV("%d='00'\t%Od='〇'\t%e=' 0'\t%Oe='〇'\n"), lfmt, 0d);
  check(loc, SV("%d='01'\t%Od='一'\t%e=' 1'\t%Oe='一'\n"), lfmt, 1d);
  check(loc, SV("%d='31'\t%Od='三十一'\t%e='31'\t%Oe='三十一'\n"), lfmt, 31d);
  check(loc, SV("%d='255'\t%Od='255'\t%e='255'\t%Oe='255'\n"), lfmt, 255d);
#endif // defined(__APPLE__) || defined(_AIX)

  std::locale::global(std::locale::classic());
}

template <class CharT>
static void test() {
  using namespace std::literals::chrono_literals;

  test_no_chrono_specs<CharT>();
  test_valid_values<CharT>();
  check_invalid_types<CharT>({SV("d"), SV("e"), SV("Od"), SV("Oe")}, 0d);

  check_exception("Expected '%' or '}' in the chrono format-string", SV("{:A"), 0d);
  check_exception("The chrono-specs contains a '{'", SV("{:%%{"), 0d);
  check_exception("End of input while parsing the modifier chrono conversion-spec", SV("{:%"), 0d);
  check_exception("End of input while parsing the modifier E", SV("{:%E"), 0d);
  check_exception("End of input while parsing the modifier O", SV("{:%O"), 0d);

  // Precision not allowed
  check_exception("Expected '%' or '}' in the chrono format-string", SV("{:.3}"), 0d);
}

int main(int, char**) {
  test<char>();

#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  test<wchar_t>();
#endif

  return 0;
}
