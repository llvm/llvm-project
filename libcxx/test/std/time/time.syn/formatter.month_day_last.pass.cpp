//===----------------------------------------------------------------------===//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// XFAIL: LIBCXX-FREEBSD-FIXME

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: no-localization
// UNSUPPORTED: libcpp-has-no-incomplete-format

// TODO FMT It seems GCC uses too much memory in the CI and fails.
// UNSUPPORTED: gcc-12

// TODO FMT This test should not require std::to_chars(floating-point)
// XFAIL: availability-fp_to_chars-missing

// REQUIRES: locale.fr_FR.UTF-8
// REQUIRES: locale.ja_JP.UTF-8

// <chrono>

// template<class charT> struct formatter<chrono::month_day_last, charT>;

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
  // Valid month
  check(SV("Jan/last"), SV("{}"), std::chrono::month_day_last{std::chrono::month{1}});
  check(SV("*Jan/last*"), SV("{:*^10}"), std::chrono::month_day_last{std::chrono::month{1}});
  check(SV("*Jan/last"), SV("{:*>9}"), std::chrono::month_day_last{std::chrono::month{1}});

  // Invalid month
  check(SV("0 is not a valid month/last"), SV("{}"), std::chrono::month_day_last{std::chrono::month{0}});
  check(SV("*0 is not a valid month/last*"), SV("{:*^29}"), std::chrono::month_day_last{std::chrono::month{0}});
}

template <class CharT>
static void test_valid_values() {
  // Test that %b, %h, and %B throw an exception.
  check_exception("formatting a month name from an invalid month number",
                  SV("{:%b}"),
                  std::chrono::month_day_last{std::chrono::month{200}});
  check_exception("formatting a month name from an invalid month number",
                  SV("{:%b}"),
                  std::chrono::month_day_last{std::chrono::month{13}});
  check_exception("formatting a month name from an invalid month number",
                  SV("{:%b}"),
                  std::chrono::month_day_last{std::chrono::month{255}});

  check_exception("formatting a month name from an invalid month number",
                  SV("{:%h}"),
                  std::chrono::month_day_last{std::chrono::month{0}});
  check_exception("formatting a month name from an invalid month number",
                  SV("{:%h}"),
                  std::chrono::month_day_last{std::chrono::month{13}});
  check_exception("formatting a month name from an invalid month number",
                  SV("{:%h}"),
                  std::chrono::month_day_last{std::chrono::month{255}});

  check_exception("formatting a month name from an invalid month number",
                  SV("{:%B}"),
                  std::chrono::month_day_last{std::chrono::month{0}});
  check_exception("formatting a month name from an invalid month number",
                  SV("{:%B}"),
                  std::chrono::month_day_last{std::chrono::month{13}});
  check_exception("formatting a month name from an invalid month number",
                  SV("{:%B}"),
                  std::chrono::month_day_last{std::chrono::month{255}});

  constexpr std::basic_string_view<CharT> fmt  = SV("{:%%b='%b'%t%%B='%B'%t%%h='%h'%t%%m='%m'%t%%Om='%Om'%n}");
  constexpr std::basic_string_view<CharT> lfmt = SV("{:L%%b='%b'%t%%B='%B'%t%%h='%h'%t%%m='%m'%t%%Om='%Om'%n}");

  const std::locale loc(LOCALE_ja_JP_UTF_8);
  std::locale::global(std::locale(LOCALE_fr_FR_UTF_8));

  // Non localized output using C-locale
  check(SV("%b='Jan'\t%B='January'\t%h='Jan'\t%m='01'\t%Om='01'\n"),
        fmt,
        std::chrono::month_day_last{std::chrono::January});
  check(SV("%b='Feb'\t%B='February'\t%h='Feb'\t%m='02'\t%Om='02'\n"),
        fmt,
        std::chrono::month_day_last{std::chrono::February});
  check(
      SV("%b='Mar'\t%B='March'\t%h='Mar'\t%m='03'\t%Om='03'\n"), fmt, std::chrono::month_day_last{std::chrono::March});
  check(
      SV("%b='Apr'\t%B='April'\t%h='Apr'\t%m='04'\t%Om='04'\n"), fmt, std::chrono::month_day_last{std::chrono::April});
  check(SV("%b='May'\t%B='May'\t%h='May'\t%m='05'\t%Om='05'\n"), fmt, std::chrono::month_day_last{std::chrono::May});
  check(SV("%b='Jun'\t%B='June'\t%h='Jun'\t%m='06'\t%Om='06'\n"), fmt, std::chrono::month_day_last{std::chrono::June});
  check(SV("%b='Jul'\t%B='July'\t%h='Jul'\t%m='07'\t%Om='07'\n"), fmt, std::chrono::month_day_last{std::chrono::July});
  check(SV("%b='Aug'\t%B='August'\t%h='Aug'\t%m='08'\t%Om='08'\n"),
        fmt,
        std::chrono::month_day_last{std::chrono::August});
  check(SV("%b='Sep'\t%B='September'\t%h='Sep'\t%m='09'\t%Om='09'\n"),
        fmt,
        std::chrono::month_day_last{std::chrono::September});
  check(SV("%b='Oct'\t%B='October'\t%h='Oct'\t%m='10'\t%Om='10'\n"),
        fmt,
        std::chrono::month_day_last{std::chrono::October});
  check(SV("%b='Nov'\t%B='November'\t%h='Nov'\t%m='11'\t%Om='11'\n"),
        fmt,
        std::chrono::month_day_last{std::chrono::November});
  check(SV("%b='Dec'\t%B='December'\t%h='Dec'\t%m='12'\t%Om='12'\n"),
        fmt,
        std::chrono::month_day_last{std::chrono::December});

  // Use the global locale (fr_FR)
#if defined(__APPLE__)
  check(SV("%b='jan'\t%B='janvier'\t%h='jan'\t%m='01'\t%Om='01'\n"),
        lfmt,
        std::chrono::month_day_last{std::chrono::January});
  check(SV("%b='fév'\t%B='février'\t%h='fév'\t%m='02'\t%Om='02'\n"),
        lfmt,
        std::chrono::month_day_last{std::chrono::February});
  check(
      SV("%b='mar'\t%B='mars'\t%h='mar'\t%m='03'\t%Om='03'\n"), lfmt, std::chrono::month_day_last{std::chrono::March});
  check(
      SV("%b='avr'\t%B='avril'\t%h='avr'\t%m='04'\t%Om='04'\n"), lfmt, std::chrono::month_day_last{std::chrono::April});
  check(SV("%b='mai'\t%B='mai'\t%h='mai'\t%m='05'\t%Om='05'\n"), lfmt, std::chrono::month_day_last{std::chrono::May});
  check(SV("%b='jui'\t%B='juin'\t%h='jui'\t%m='06'\t%Om='06'\n"), lfmt, std::chrono::month_day_last{std::chrono::June});
  check(SV("%b='jul'\t%B='juillet'\t%h='jul'\t%m='07'\t%Om='07'\n"),
        lfmt,
        std::chrono::month_day_last{std::chrono::July});
  check(
      SV("%b='aoû'\t%B='août'\t%h='aoû'\t%m='08'\t%Om='08'\n"), lfmt, std::chrono::month_day_last{std::chrono::August});
  check(SV("%b='sep'\t%B='septembre'\t%h='sep'\t%m='09'\t%Om='09'\n"),
        lfmt,
        std::chrono::month_day_last{std::chrono::September});
  check(SV("%b='oct'\t%B='octobre'\t%h='oct'\t%m='10'\t%Om='10'\n"),
        lfmt,
        std::chrono::month_day_last{std::chrono::October});
  check(SV("%b='nov'\t%B='novembre'\t%h='nov'\t%m='11'\t%Om='11'\n"),
        lfmt,
        std::chrono::month_day_last{std::chrono::November});
  check(SV("%b='déc'\t%B='décembre'\t%h='déc'\t%m='12'\t%Om='12'\n"),
        lfmt,
        std::chrono::month_day_last{std::chrono::December});
#else    // defined(__APPLE__)
  check(SV("%b='janv.'\t%B='janvier'\t%h='janv.'\t%m='01'\t%Om='01'\n"),
        lfmt,
        std::chrono::month_day_last{std::chrono::January});
  check(SV("%b='févr.'\t%B='février'\t%h='févr.'\t%m='02'\t%Om='02'\n"),
        lfmt,
        std::chrono::month_day_last{std::chrono::February});
  check(SV("%b='mars'\t%B='mars'\t%h='mars'\t%m='03'\t%Om='03'\n"),
        lfmt,
        std::chrono::month_day_last{std::chrono::March});
  check(
#  if defined(_WIN32) || defined(_AIX)
      SV("%b='avr.'\t%B='avril'\t%h='avr.'\t%m='04'\t%Om='04'\n"),
#  else  // defined(_WIN32) || defined(_AIX)
      SV("%b='avril'\t%B='avril'\t%h='avril'\t%m='04'\t%Om='04'\n"),
#  endif // defined(_WIN32) || defined(_AIX)
      lfmt,
      std::chrono::month_day_last{std::chrono::April});
  check(SV("%b='mai'\t%B='mai'\t%h='mai'\t%m='05'\t%Om='05'\n"), lfmt, std::chrono::month_day_last{std::chrono::May});
  check(
      SV("%b='juin'\t%B='juin'\t%h='juin'\t%m='06'\t%Om='06'\n"), lfmt, std::chrono::month_day_last{std::chrono::June});
  check(SV("%b='juil.'\t%B='juillet'\t%h='juil.'\t%m='07'\t%Om='07'\n"),
        lfmt,
        std::chrono::month_day_last{std::chrono::July});
  check(SV("%b='août'\t%B='août'\t%h='août'\t%m='08'\t%Om='08'\n"),
        lfmt,
        std::chrono::month_day_last{std::chrono::August});
  check(SV("%b='sept.'\t%B='septembre'\t%h='sept.'\t%m='09'\t%Om='09'\n"),
        lfmt,
        std::chrono::month_day_last{std::chrono::September});
  check(SV("%b='oct.'\t%B='octobre'\t%h='oct.'\t%m='10'\t%Om='10'\n"),
        lfmt,
        std::chrono::month_day_last{std::chrono::October});
  check(SV("%b='nov.'\t%B='novembre'\t%h='nov.'\t%m='11'\t%Om='11'\n"),
        lfmt,
        std::chrono::month_day_last{std::chrono::November});
  check(SV("%b='déc.'\t%B='décembre'\t%h='déc.'\t%m='12'\t%Om='12'\n"),
        lfmt,
        std::chrono::month_day_last{std::chrono::December});
#endif   // defined(__APPLE__)

  // Use supplied locale (ja_JP)
#ifdef _WIN32
  check(loc,
        SV("%b='1'\t%B='1月'\t%h='1'\t%m='01'\t%Om='01'\n"),
        lfmt,
        std::chrono::month_day_last{std::chrono::January});
  check(loc,
        SV("%b='2'\t%B='2月'\t%h='2'\t%m='02'\t%Om='02'\n"),
        lfmt,
        std::chrono::month_day_last{std::chrono::February});
  check(
      loc, SV("%b='3'\t%B='3月'\t%h='3'\t%m='03'\t%Om='03'\n"), lfmt, std::chrono::month_day_last{std::chrono::March});
  check(
      loc, SV("%b='4'\t%B='4月'\t%h='4'\t%m='04'\t%Om='04'\n"), lfmt, std::chrono::month_day_last{std::chrono::April});
  check(loc, SV("%b='5'\t%B='5月'\t%h='5'\t%m='05'\t%Om='05'\n"), lfmt, std::chrono::month_day_last{std::chrono::May});
  check(loc, SV("%b='6'\t%B='6月'\t%h='6'\t%m='06'\t%Om='06'\n"), lfmt, std::chrono::month_day_last{std::chrono::June});
  check(loc, SV("%b='7'\t%B='7月'\t%h='7'\t%m='07'\t%Om='07'\n"), lfmt, std::chrono::month_day_last{std::chrono::July});
  check(
      loc, SV("%b='8'\t%B='8月'\t%h='8'\t%m='08'\t%Om='08'\n"), lfmt, std::chrono::month_day_last{std::chrono::August});
  check(loc,
        SV("%b='9'\t%B='9月'\t%h='9'\t%m='09'\t%Om='09'\n"),
        lfmt,
        std::chrono::month_day_last{std::chrono::September});
  check(loc,
        SV("%b='10'\t%B='10月'\t%h='10'\t%m='10'\t%Om='10'\n"),
        lfmt,
        std::chrono::month_day_last{std::chrono::October});
  check(loc,
        SV("%b='11'\t%B='11月'\t%h='11'\t%m='11'\t%Om='11'\n"),
        lfmt,
        std::chrono::month_day_last{std::chrono::November});
  check(loc,
        SV("%b='12'\t%B='12月'\t%h='12'\t%m='12'\t%Om='12'\n"),
        lfmt,
        std::chrono::month_day_last{std::chrono::December});
#elif defined(__APPLE__) // defined(_WIN32)
  check(loc,
        SV("%b=' 1'\t%B='1月'\t%h=' 1'\t%m='01'\t%Om='01'\n"),
        lfmt,
        std::chrono::month_day_last{std::chrono::January});
  check(loc,
        SV("%b=' 2'\t%B='2月'\t%h=' 2'\t%m='02'\t%Om='02'\n"),
        lfmt,
        std::chrono::month_day_last{std::chrono::February});
  check(loc,
        SV("%b=' 3'\t%B='3月'\t%h=' 3'\t%m='03'\t%Om='03'\n"),
        lfmt,
        std::chrono::month_day_last{std::chrono::March});
  check(loc,
        SV("%b=' 4'\t%B='4月'\t%h=' 4'\t%m='04'\t%Om='04'\n"),
        lfmt,
        std::chrono::month_day_last{std::chrono::April});
  check(
      loc, SV("%b=' 5'\t%B='5月'\t%h=' 5'\t%m='05'\t%Om='05'\n"), lfmt, std::chrono::month_day_last{std::chrono::May});
  check(
      loc, SV("%b=' 6'\t%B='6月'\t%h=' 6'\t%m='06'\t%Om='06'\n"), lfmt, std::chrono::month_day_last{std::chrono::June});
  check(
      loc, SV("%b=' 7'\t%B='7月'\t%h=' 7'\t%m='07'\t%Om='07'\n"), lfmt, std::chrono::month_day_last{std::chrono::July});
  check(loc,
        SV("%b=' 8'\t%B='8月'\t%h=' 8'\t%m='08'\t%Om='08'\n"),
        lfmt,
        std::chrono::month_day_last{std::chrono::August});
  check(loc,
        SV("%b=' 9'\t%B='9月'\t%h=' 9'\t%m='09'\t%Om='09'\n"),
        lfmt,
        std::chrono::month_day_last{std::chrono::September});
  check(loc,
        SV("%b='10'\t%B='10月'\t%h='10'\t%m='10'\t%Om='10'\n"),
        lfmt,
        std::chrono::month_day_last{std::chrono::October});
  check(loc,
        SV("%b='11'\t%B='11月'\t%h='11'\t%m='11'\t%Om='11'\n"),
        lfmt,
        std::chrono::month_day_last{std::chrono::November});
  check(loc,
        SV("%b='12'\t%B='12月'\t%h='12'\t%m='12'\t%Om='12'\n"),
        lfmt,
        std::chrono::month_day_last{std::chrono::December});
#elif defined(_AIX)      // defined(_WIN32)
  check(loc,
        SV("%b='1月'\t%B='1月'\t%h='1月'\t%m='01'\t%Om='01'\n"),
        lfmt,
        std::chrono::month_day_last{std::chrono::January});
  check(loc,
        SV("%b='2月'\t%B='2月'\t%h='2月'\t%m='02'\t%Om='02'\n"),
        lfmt,
        std::chrono::month_day_last{std::chrono::February});
  check(loc,
        SV("%b='3月'\t%B='3月'\t%h='3月'\t%m='03'\t%Om='03'\n"),
        lfmt,
        std::chrono::month_day_last{std::chrono::March});
  check(loc,
        SV("%b='4月'\t%B='4月'\t%h='4月'\t%m='04'\t%Om='04'\n"),
        lfmt,
        std::chrono::month_day_last{std::chrono::April});
  check(loc,
        SV("%b='5月'\t%B='5月'\t%h='5月'\t%m='05'\t%Om='05'\n"),
        lfmt,
        std::chrono::month_day_last{std::chrono::May});
  check(loc,
        SV("%b='6月'\t%B='6月'\t%h='6月'\t%m='06'\t%Om='06'\n"),
        lfmt,
        std::chrono::month_day_last{std::chrono::June});
  check(loc,
        SV("%b='7月'\t%B='7月'\t%h='7月'\t%m='07'\t%Om='07'\n"),
        lfmt,
        std::chrono::month_day_last{std::chrono::July});
  check(loc,
        SV("%b='8月'\t%B='8月'\t%h='8月'\t%m='08'\t%Om='08'\n"),
        lfmt,
        std::chrono::month_day_last{std::chrono::August});
  check(loc,
        SV("%b='9月'\t%B='9月'\t%h='9月'\t%m='09'\t%Om='09'\n"),
        lfmt,
        std::chrono::month_day_last{std::chrono::September});
  check(loc,
        SV("%b='10月'\t%B='10月'\t%h='10月'\t%m='10'\t%Om='10'\n"),
        lfmt,
        std::chrono::month_day_last{std::chrono::October});
  check(loc,
        SV("%b='11月'\t%B='11月'\t%h='11月'\t%m='11'\t%Om='11'\n"),
        lfmt,
        std::chrono::month_day_last{std::chrono::November});
  check(loc,
        SV("%b='12月'\t%B='12月'\t%h='12月'\t%m='12'\t%Om='12'\n"),
        lfmt,
        std::chrono::month_day_last{std::chrono::December});
#else                    // defined(_WIN32)
  check(loc,
        SV("%b=' 1月'\t%B='1月'\t%h=' 1月'\t%m='01'\t%Om='一'\n"),
        lfmt,
        std::chrono::month_day_last{std::chrono::January});
  check(loc,
        SV("%b=' 2月'\t%B='2月'\t%h=' 2月'\t%m='02'\t%Om='二'\n"),
        lfmt,
        std::chrono::month_day_last{std::chrono::February});
  check(loc,
        SV("%b=' 3月'\t%B='3月'\t%h=' 3月'\t%m='03'\t%Om='三'\n"),
        lfmt,
        std::chrono::month_day_last{std::chrono::March});
  check(loc,
        SV("%b=' 4月'\t%B='4月'\t%h=' 4月'\t%m='04'\t%Om='四'\n"),
        lfmt,
        std::chrono::month_day_last{std::chrono::April});
  check(loc,
        SV("%b=' 5月'\t%B='5月'\t%h=' 5月'\t%m='05'\t%Om='五'\n"),
        lfmt,
        std::chrono::month_day_last{std::chrono::May});
  check(loc,
        SV("%b=' 6月'\t%B='6月'\t%h=' 6月'\t%m='06'\t%Om='六'\n"),
        lfmt,
        std::chrono::month_day_last{std::chrono::June});
  check(loc,
        SV("%b=' 7月'\t%B='7月'\t%h=' 7月'\t%m='07'\t%Om='七'\n"),
        lfmt,
        std::chrono::month_day_last{std::chrono::July});
  check(loc,
        SV("%b=' 8月'\t%B='8月'\t%h=' 8月'\t%m='08'\t%Om='八'\n"),
        lfmt,
        std::chrono::month_day_last{std::chrono::August});
  check(loc,
        SV("%b=' 9月'\t%B='9月'\t%h=' 9月'\t%m='09'\t%Om='九'\n"),
        lfmt,
        std::chrono::month_day_last{std::chrono::September});
  check(loc,
        SV("%b='10月'\t%B='10月'\t%h='10月'\t%m='10'\t%Om='十'\n"),
        lfmt,
        std::chrono::month_day_last{std::chrono::October});
  check(loc,
        SV("%b='11月'\t%B='11月'\t%h='11月'\t%m='11'\t%Om='十一'\n"),
        lfmt,
        std::chrono::month_day_last{std::chrono::November});
  check(loc,
        SV("%b='12月'\t%B='12月'\t%h='12月'\t%m='12'\t%Om='十二'\n"),
        lfmt,
        std::chrono::month_day_last{std::chrono::December});
#endif                   // defined(_WIN32)

  std::locale::global(std::locale::classic());
}

template <class CharT>
static void test() {
  test_no_chrono_specs<CharT>();
  test_valid_values<CharT>();
  check_invalid_types<CharT>(
      {SV("b"), SV("B"), SV("h"), SV("m"), SV("Om")}, std::chrono::month_day_last{std::chrono::January});

  check_exception(
      "Expected '%' or '}' in the chrono format-string", SV("{:A"), std::chrono::month_day_last{std::chrono::January});
  check_exception("The chrono-specs contains a '{'", SV("{:%%{"), std::chrono::month_day_last{std::chrono::January});
  check_exception("End of input while parsing the modifier chrono conversion-spec",
                  SV("{:%"),
                  std::chrono::month_day_last{std::chrono::January});
  check_exception(
      "End of input while parsing the modifier E", SV("{:%E"), std::chrono::month_day_last{std::chrono::January});
  check_exception(
      "End of input while parsing the modifier O", SV("{:%O"), std::chrono::month_day_last{std::chrono::January});

  // Precision not allowed
  check_exception("Expected '%' or '}' in the chrono format-string",
                  SV("{:.3}"),
                  std::chrono::month_day_last{std::chrono::January});
}

int main(int, char**) {
  test<char>();

#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  test<wchar_t>();
#endif

  return 0;
}
