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

// REQUIRES: locale.fr_FR.UTF-8
// REQUIRES: locale.ja_JP.UTF-8

// <chrono>

// template<class charT> struct formatter<chrono::month_weekday, charT>;

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
  // Month: valid,   weekday: valid,   index: invalid
  check(SV("Jan/Sun[0 is not a valid index]"),
        SV("{}"),
        std::chrono::month_weekday{std::chrono::month{1}, std::chrono::weekday_indexed{std::chrono::weekday{0}, 0}});
  check(SV("*Jan/Sun[0 is not a valid index]*"),
        SV("{:*^33}"),
        std::chrono::month_weekday{std::chrono::month{1}, std::chrono::weekday_indexed{std::chrono::weekday{0}, 0}});
  check(SV("*Jan/Sun[0 is not a valid index]"),
        SV("{:*>32}"),
        std::chrono::month_weekday{std::chrono::month{1}, std::chrono::weekday_indexed{std::chrono::weekday{0}, 0}});

  // Month: valid,   weekday: invalid, index: valid
  check(SV("Jan/8 is not a valid weekday[1]"),
        SV("{}"),
        std::chrono::month_weekday{std::chrono::month{1}, std::chrono::weekday_indexed{std::chrono::weekday{8}, 1}});
  check(SV("*Jan/8 is not a valid weekday[1]*"),
        SV("{:*^33}"),
        std::chrono::month_weekday{std::chrono::month{1}, std::chrono::weekday_indexed{std::chrono::weekday{8}, 1}});
  check(SV("*Jan/8 is not a valid weekday[1]"),
        SV("{:*>32}"),
        std::chrono::month_weekday{std::chrono::month{1}, std::chrono::weekday_indexed{std::chrono::weekday{8}, 1}});

  // Month: valid,   weekday: invalid,   index: invalid
  check(SV("Jan/8 is not a valid weekday[0 is not a valid index]"),
        SV("{}"),
        std::chrono::month_weekday{std::chrono::month{1}, std::chrono::weekday_indexed{std::chrono::weekday{8}, 0}});
  check(SV("*Jan/8 is not a valid weekday[0 is not a valid index]*"),
        SV("{:*^54}"),
        std::chrono::month_weekday{std::chrono::month{1}, std::chrono::weekday_indexed{std::chrono::weekday{8}, 0}});
  check(SV("*Jan/8 is not a valid weekday[0 is not a valid index]"),
        SV("{:*>53}"),
        std::chrono::month_weekday{std::chrono::month{1}, std::chrono::weekday_indexed{std::chrono::weekday{8}, 0}});

  // Month: invalid, weekday: valid,   index: valid
  check(SV("0 is not a valid month/Sun[1]"),
        SV("{}"),
        std::chrono::month_weekday{std::chrono::month{0}, std::chrono::weekday_indexed{std::chrono::weekday{0}, 1}});
  check(SV("*0 is not a valid month/Sun[1]*"),
        SV("{:*^31}"),
        std::chrono::month_weekday{std::chrono::month{0}, std::chrono::weekday_indexed{std::chrono::weekday{0}, 1}});
  check(SV("*0 is not a valid month/Sun[1]"),
        SV("{:*>30}"),
        std::chrono::month_weekday{std::chrono::month{0}, std::chrono::weekday_indexed{std::chrono::weekday{0}, 1}});

  // Month: invalid, weekday: valid,   index: invalid
  check(SV("0 is not a valid month/Sun[0 is not a valid index]"),
        SV("{}"),
        std::chrono::month_weekday{std::chrono::month{0}, std::chrono::weekday_indexed{std::chrono::weekday{0}, 0}});
  check(SV("*0 is not a valid month/Sun[0 is not a valid index]*"),
        SV("{:*^52}"),
        std::chrono::month_weekday{std::chrono::month{0}, std::chrono::weekday_indexed{std::chrono::weekday{0}, 0}});
  check(SV("*0 is not a valid month/Sun[0 is not a valid index]"),
        SV("{:*>51}"),
        std::chrono::month_weekday{std::chrono::month{0}, std::chrono::weekday_indexed{std::chrono::weekday{0}, 0}});

  // Month: invalid, weekday: invalid, index: valid
  check(SV("0 is not a valid month/8 is not a valid weekday[1]"),
        SV("{}"),
        std::chrono::month_weekday{std::chrono::month{0}, std::chrono::weekday_indexed{std::chrono::weekday{8}, 1}});
  check(SV("*0 is not a valid month/8 is not a valid weekday[1]*"),
        SV("{:*^52}"),
        std::chrono::month_weekday{std::chrono::month{0}, std::chrono::weekday_indexed{std::chrono::weekday{8}, 1}});
  check(SV("*0 is not a valid month/8 is not a valid weekday[1]"),
        SV("{:*>51}"),
        std::chrono::month_weekday{std::chrono::month{0}, std::chrono::weekday_indexed{std::chrono::weekday{8}, 1}});

  // Month: invalid, weekday: valid,   index: invalid
  check(SV("0 is not a valid month/8 is not a valid weekday[0 is not a valid index]"),
        SV("{}"),
        std::chrono::month_weekday{std::chrono::month{0}, std::chrono::weekday_indexed{std::chrono::weekday{8}, 0}});
  check(SV("*0 is not a valid month/8 is not a valid weekday[0 is not a valid index]*"),
        SV("{:*^73}"),
        std::chrono::month_weekday{std::chrono::month{0}, std::chrono::weekday_indexed{std::chrono::weekday{8}, 0}});
  check(SV("*0 is not a valid month/8 is not a valid weekday[0 is not a valid index]"),
        SV("{:*>72}"),
        std::chrono::month_weekday{std::chrono::month{0}, std::chrono::weekday_indexed{std::chrono::weekday{8}, 0}});
}

template <class CharT>
static void test_invalid_values() {
  // Test that %a, %b, %h, %a, and %B throw an exception.
  check_exception(
      "formatting a weekday name needs a valid weekday",
      SV("{:%a}"),
      std::chrono::month_weekday{std::chrono::month{1}, std::chrono::weekday_indexed{std::chrono::weekday{8}, 1}});
  check_exception(
      "formatting a weekday name needs a valid weekday",
      SV("{:%a}"),
      std::chrono::month_weekday{std::chrono::month{1}, std::chrono::weekday_indexed{std::chrono::weekday{255}, 1}});

  check_exception(
      "formatting a month name from an invalid month number",
      SV("{:%b}"),
      std::chrono::month_weekday{std::chrono::month{0}, std::chrono::weekday_indexed{std::chrono::weekday{0}, 1}});
  check_exception(
      "formatting a month name from an invalid month number",
      SV("{:%b}"),
      std::chrono::month_weekday{std::chrono::month{13}, std::chrono::weekday_indexed{std::chrono::weekday{0}, 1}});
  check_exception(
      "formatting a month name from an invalid month number",
      SV("{:%b}"),
      std::chrono::month_weekday{std::chrono::month{255}, std::chrono::weekday_indexed{std::chrono::weekday{0}, 1}});

  check_exception(
      "formatting a month name from an invalid month number",
      SV("{:%h}"),
      std::chrono::month_weekday{std::chrono::month{0}, std::chrono::weekday_indexed{std::chrono::weekday{0}, 1}});
  check_exception(
      "formatting a month name from an invalid month number",
      SV("{:%h}"),
      std::chrono::month_weekday{std::chrono::month{13}, std::chrono::weekday_indexed{std::chrono::weekday{0}, 1}});
  check_exception(
      "formatting a month name from an invalid month number",
      SV("{:%h}"),
      std::chrono::month_weekday{std::chrono::month{255}, std::chrono::weekday_indexed{std::chrono::weekday{0}, 1}});

  check_exception(
      "formatting a weekday name needs a valid weekday",
      SV("{:%A}"),
      std::chrono::month_weekday{std::chrono::month{1}, std::chrono::weekday_indexed{std::chrono::weekday{8}, 1}});
  check_exception(
      "formatting a weekday name needs a valid weekday",
      SV("{:%A}"),
      std::chrono::month_weekday{std::chrono::month{1}, std::chrono::weekday_indexed{std::chrono::weekday{255}, 1}});

  check_exception(
      "formatting a month name from an invalid month number",
      SV("{:%B}"),
      std::chrono::month_weekday{std::chrono::month{0}, std::chrono::weekday_indexed{std::chrono::weekday{0}, 1}});
  check_exception(
      "formatting a month name from an invalid month number",
      SV("{:%B}"),
      std::chrono::month_weekday{std::chrono::month{13}, std::chrono::weekday_indexed{std::chrono::weekday{0}, 1}});
  check_exception(
      "formatting a month name from an invalid month number",
      SV("{:%B}"),
      std::chrono::month_weekday{std::chrono::month{255}, std::chrono::weekday_indexed{std::chrono::weekday{0}, 1}});
}

template <class CharT>
static void test_valid_month() {
  constexpr std::basic_string_view<CharT> fmt  = SV("{:%%b='%b'%t%%B='%B'%t%%h='%h'%t%%m='%m'%t%%Om='%Om'%n}");
  constexpr std::basic_string_view<CharT> lfmt = SV("{:L%%b='%b'%t%%B='%B'%t%%h='%h'%t%%m='%m'%t%%Om='%Om'%n}");

  const std::locale loc(LOCALE_ja_JP_UTF_8);
  std::locale::global(std::locale(LOCALE_fr_FR_UTF_8));

  // Non localized output using C-locale
  check(SV("%b='Jan'\t%B='January'\t%h='Jan'\t%m='01'\t%Om='01'\n"),
        fmt,
        std::chrono::month_weekday{std::chrono::January, std::chrono::weekday_indexed{std::chrono::weekday{0}, 1}});
  check(SV("%b='Feb'\t%B='February'\t%h='Feb'\t%m='02'\t%Om='02'\n"),
        fmt,
        std::chrono::month_weekday{std::chrono::February, std::chrono::weekday_indexed{std::chrono::weekday{0}, 1}});
  check(SV("%b='Mar'\t%B='March'\t%h='Mar'\t%m='03'\t%Om='03'\n"),
        fmt,
        std::chrono::month_weekday{std::chrono::March, std::chrono::weekday_indexed{std::chrono::weekday{0}, 1}});
  check(SV("%b='Apr'\t%B='April'\t%h='Apr'\t%m='04'\t%Om='04'\n"),
        fmt,
        std::chrono::month_weekday{std::chrono::April, std::chrono::weekday_indexed{std::chrono::weekday{0}, 1}});
  check(SV("%b='May'\t%B='May'\t%h='May'\t%m='05'\t%Om='05'\n"),
        fmt,
        std::chrono::month_weekday{std::chrono::May, std::chrono::weekday_indexed{std::chrono::weekday{0}, 1}});
  check(SV("%b='Jun'\t%B='June'\t%h='Jun'\t%m='06'\t%Om='06'\n"),
        fmt,
        std::chrono::month_weekday{std::chrono::June, std::chrono::weekday_indexed{std::chrono::weekday{0}, 1}});
  check(SV("%b='Jul'\t%B='July'\t%h='Jul'\t%m='07'\t%Om='07'\n"),
        fmt,
        std::chrono::month_weekday{std::chrono::July, std::chrono::weekday_indexed{std::chrono::weekday{0}, 1}});
  check(SV("%b='Aug'\t%B='August'\t%h='Aug'\t%m='08'\t%Om='08'\n"),
        fmt,
        std::chrono::month_weekday{std::chrono::August, std::chrono::weekday_indexed{std::chrono::weekday{0}, 1}});
  check(SV("%b='Sep'\t%B='September'\t%h='Sep'\t%m='09'\t%Om='09'\n"),
        fmt,
        std::chrono::month_weekday{std::chrono::September, std::chrono::weekday_indexed{std::chrono::weekday{0}, 1}});
  check(SV("%b='Oct'\t%B='October'\t%h='Oct'\t%m='10'\t%Om='10'\n"),
        fmt,
        std::chrono::month_weekday{std::chrono::October, std::chrono::weekday_indexed{std::chrono::weekday{0}, 1}});
  check(SV("%b='Nov'\t%B='November'\t%h='Nov'\t%m='11'\t%Om='11'\n"),
        fmt,
        std::chrono::month_weekday{std::chrono::November, std::chrono::weekday_indexed{std::chrono::weekday{0}, 1}});
  check(SV("%b='Dec'\t%B='December'\t%h='Dec'\t%m='12'\t%Om='12'\n"),
        fmt,
        std::chrono::month_weekday{std::chrono::December, std::chrono::weekday_indexed{std::chrono::weekday{0}, 1}});

  // Use the global locale (fr_FR)
#if defined(__APPLE__)
  check(SV("%b='jan'\t%B='janvier'\t%h='jan'\t%m='01'\t%Om='01'\n"),
        lfmt,
        std::chrono::month_weekday{std::chrono::January, std::chrono::weekday_indexed{std::chrono::weekday{0}, 1}});
  check(SV("%b='fév'\t%B='février'\t%h='fév'\t%m='02'\t%Om='02'\n"),
        lfmt,
        std::chrono::month_weekday{std::chrono::February, std::chrono::weekday_indexed{std::chrono::weekday{0}, 1}});
  check(SV("%b='mar'\t%B='mars'\t%h='mar'\t%m='03'\t%Om='03'\n"),
        lfmt,
        std::chrono::month_weekday{std::chrono::March, std::chrono::weekday_indexed{std::chrono::weekday{0}, 1}});
  check(SV("%b='avr'\t%B='avril'\t%h='avr'\t%m='04'\t%Om='04'\n"),
        lfmt,
        std::chrono::month_weekday{std::chrono::April, std::chrono::weekday_indexed{std::chrono::weekday{0}, 1}});
  check(SV("%b='mai'\t%B='mai'\t%h='mai'\t%m='05'\t%Om='05'\n"),
        lfmt,
        std::chrono::month_weekday{std::chrono::May, std::chrono::weekday_indexed{std::chrono::weekday{0}, 1}});
  check(SV("%b='jui'\t%B='juin'\t%h='jui'\t%m='06'\t%Om='06'\n"),
        lfmt,
        std::chrono::month_weekday{std::chrono::June, std::chrono::weekday_indexed{std::chrono::weekday{0}, 1}});
  check(SV("%b='jul'\t%B='juillet'\t%h='jul'\t%m='07'\t%Om='07'\n"),
        lfmt,
        std::chrono::month_weekday{std::chrono::July, std::chrono::weekday_indexed{std::chrono::weekday{0}, 1}});
  check(SV("%b='aoû'\t%B='août'\t%h='aoû'\t%m='08'\t%Om='08'\n"),
        lfmt,
        std::chrono::month_weekday{std::chrono::August, std::chrono::weekday_indexed{std::chrono::weekday{0}, 1}});
  check(SV("%b='sep'\t%B='septembre'\t%h='sep'\t%m='09'\t%Om='09'\n"),
        lfmt,
        std::chrono::month_weekday{std::chrono::September, std::chrono::weekday_indexed{std::chrono::weekday{0}, 1}});
  check(SV("%b='oct'\t%B='octobre'\t%h='oct'\t%m='10'\t%Om='10'\n"),
        lfmt,
        std::chrono::month_weekday{std::chrono::October, std::chrono::weekday_indexed{std::chrono::weekday{0}, 1}});
  check(SV("%b='nov'\t%B='novembre'\t%h='nov'\t%m='11'\t%Om='11'\n"),
        lfmt,
        std::chrono::month_weekday{std::chrono::November, std::chrono::weekday_indexed{std::chrono::weekday{0}, 1}});
  check(SV("%b='déc'\t%B='décembre'\t%h='déc'\t%m='12'\t%Om='12'\n"),
        lfmt,
        std::chrono::month_weekday{std::chrono::December, std::chrono::weekday_indexed{std::chrono::weekday{0}, 1}});
#else    // defined(__APPLE__)
  check(SV("%b='janv.'\t%B='janvier'\t%h='janv.'\t%m='01'\t%Om='01'\n"),
        lfmt,
        std::chrono::month_weekday{std::chrono::January, std::chrono::weekday_indexed{std::chrono::weekday{0}, 1}});
  check(SV("%b='févr.'\t%B='février'\t%h='févr.'\t%m='02'\t%Om='02'\n"),
        lfmt,
        std::chrono::month_weekday{std::chrono::February, std::chrono::weekday_indexed{std::chrono::weekday{0}, 1}});
  check(SV("%b='mars'\t%B='mars'\t%h='mars'\t%m='03'\t%Om='03'\n"),
        lfmt,
        std::chrono::month_weekday{std::chrono::March, std::chrono::weekday_indexed{std::chrono::weekday{0}, 1}});
  check(
#  if defined(_WIN32) || defined(_AIX)
      SV("%b='avr.'\t%B='avril'\t%h='avr.'\t%m='04'\t%Om='04'\n"),
#  else  // defined(_WIN32) || defined(_AIX)
      SV("%b='avril'\t%B='avril'\t%h='avril'\t%m='04'\t%Om='04'\n"),
#  endif // defined(_WIN32) || defined(_AIX)
      lfmt,
      std::chrono::month_weekday{std::chrono::April, std::chrono::weekday_indexed{std::chrono::weekday{0}, 1}});
  check(SV("%b='mai'\t%B='mai'\t%h='mai'\t%m='05'\t%Om='05'\n"),
        lfmt,
        std::chrono::month_weekday{std::chrono::May, std::chrono::weekday_indexed{std::chrono::weekday{0}, 1}});
  check(SV("%b='juin'\t%B='juin'\t%h='juin'\t%m='06'\t%Om='06'\n"),
        lfmt,
        std::chrono::month_weekday{std::chrono::June, std::chrono::weekday_indexed{std::chrono::weekday{0}, 1}});
  check(SV("%b='juil.'\t%B='juillet'\t%h='juil.'\t%m='07'\t%Om='07'\n"),
        lfmt,
        std::chrono::month_weekday{std::chrono::July, std::chrono::weekday_indexed{std::chrono::weekday{0}, 1}});
  check(SV("%b='août'\t%B='août'\t%h='août'\t%m='08'\t%Om='08'\n"),
        lfmt,
        std::chrono::month_weekday{std::chrono::August, std::chrono::weekday_indexed{std::chrono::weekday{0}, 1}});
  check(SV("%b='sept.'\t%B='septembre'\t%h='sept.'\t%m='09'\t%Om='09'\n"),
        lfmt,
        std::chrono::month_weekday{std::chrono::September, std::chrono::weekday_indexed{std::chrono::weekday{0}, 1}});
  check(SV("%b='oct.'\t%B='octobre'\t%h='oct.'\t%m='10'\t%Om='10'\n"),
        lfmt,
        std::chrono::month_weekday{std::chrono::October, std::chrono::weekday_indexed{std::chrono::weekday{0}, 1}});
  check(SV("%b='nov.'\t%B='novembre'\t%h='nov.'\t%m='11'\t%Om='11'\n"),
        lfmt,
        std::chrono::month_weekday{std::chrono::November, std::chrono::weekday_indexed{std::chrono::weekday{0}, 1}});
  check(SV("%b='déc.'\t%B='décembre'\t%h='déc.'\t%m='12'\t%Om='12'\n"),
        lfmt,
        std::chrono::month_weekday{std::chrono::December, std::chrono::weekday_indexed{std::chrono::weekday{0}, 1}});
#endif   // defined(__APPLE__)

  // Use supplied locale (ja_JP)
#ifdef _WIN32
  check(loc,
        SV("%b='1'\t%B='1月'\t%h='1'\t%m='01'\t%Om='01'\n"),
        lfmt,
        std::chrono::month_weekday{std::chrono::January, std::chrono::weekday_indexed{std::chrono::weekday{0}, 1}});
  check(loc,
        SV("%b='2'\t%B='2月'\t%h='2'\t%m='02'\t%Om='02'\n"),
        lfmt,
        std::chrono::month_weekday{std::chrono::February, std::chrono::weekday_indexed{std::chrono::weekday{0}, 1}});
  check(loc,
        SV("%b='3'\t%B='3月'\t%h='3'\t%m='03'\t%Om='03'\n"),
        lfmt,
        std::chrono::month_weekday{std::chrono::March, std::chrono::weekday_indexed{std::chrono::weekday{0}, 1}});
  check(loc,
        SV("%b='4'\t%B='4月'\t%h='4'\t%m='04'\t%Om='04'\n"),
        lfmt,
        std::chrono::month_weekday{std::chrono::April, std::chrono::weekday_indexed{std::chrono::weekday{0}, 1}});
  check(loc,
        SV("%b='5'\t%B='5月'\t%h='5'\t%m='05'\t%Om='05'\n"),
        lfmt,
        std::chrono::month_weekday{std::chrono::May, std::chrono::weekday_indexed{std::chrono::weekday{0}, 1}});
  check(loc,
        SV("%b='6'\t%B='6月'\t%h='6'\t%m='06'\t%Om='06'\n"),
        lfmt,
        std::chrono::month_weekday{std::chrono::June, std::chrono::weekday_indexed{std::chrono::weekday{0}, 1}});
  check(loc,
        SV("%b='7'\t%B='7月'\t%h='7'\t%m='07'\t%Om='07'\n"),
        lfmt,
        std::chrono::month_weekday{std::chrono::July, std::chrono::weekday_indexed{std::chrono::weekday{0}, 1}});
  check(loc,
        SV("%b='8'\t%B='8月'\t%h='8'\t%m='08'\t%Om='08'\n"),
        lfmt,
        std::chrono::month_weekday{std::chrono::August, std::chrono::weekday_indexed{std::chrono::weekday{0}, 1}});
  check(loc,
        SV("%b='9'\t%B='9月'\t%h='9'\t%m='09'\t%Om='09'\n"),
        lfmt,
        std::chrono::month_weekday{std::chrono::September, std::chrono::weekday_indexed{std::chrono::weekday{0}, 1}});
  check(loc,
        SV("%b='10'\t%B='10月'\t%h='10'\t%m='10'\t%Om='10'\n"),
        lfmt,
        std::chrono::month_weekday{std::chrono::October, std::chrono::weekday_indexed{std::chrono::weekday{0}, 1}});
  check(loc,
        SV("%b='11'\t%B='11月'\t%h='11'\t%m='11'\t%Om='11'\n"),
        lfmt,
        std::chrono::month_weekday{std::chrono::November, std::chrono::weekday_indexed{std::chrono::weekday{0}, 1}});
  check(loc,
        SV("%b='12'\t%B='12月'\t%h='12'\t%m='12'\t%Om='12'\n"),
        lfmt,
        std::chrono::month_weekday{std::chrono::December, std::chrono::weekday_indexed{std::chrono::weekday{0}, 1}});
#elif defined(__APPLE__) // defined(_WIN32)
  check(loc,
        SV("%b=' 1'\t%B='1月'\t%h=' 1'\t%m='01'\t%Om='01'\n"),
        lfmt,
        std::chrono::month_weekday{std::chrono::January, std::chrono::weekday_indexed{std::chrono::weekday{0}, 1}});
  check(loc,
        SV("%b=' 2'\t%B='2月'\t%h=' 2'\t%m='02'\t%Om='02'\n"),
        lfmt,
        std::chrono::month_weekday{std::chrono::February, std::chrono::weekday_indexed{std::chrono::weekday{0}, 1}});
  check(loc,
        SV("%b=' 3'\t%B='3月'\t%h=' 3'\t%m='03'\t%Om='03'\n"),
        lfmt,
        std::chrono::month_weekday{std::chrono::March, std::chrono::weekday_indexed{std::chrono::weekday{0}, 1}});
  check(loc,
        SV("%b=' 4'\t%B='4月'\t%h=' 4'\t%m='04'\t%Om='04'\n"),
        lfmt,
        std::chrono::month_weekday{std::chrono::April, std::chrono::weekday_indexed{std::chrono::weekday{0}, 1}});
  check(loc,
        SV("%b=' 5'\t%B='5月'\t%h=' 5'\t%m='05'\t%Om='05'\n"),
        lfmt,
        std::chrono::month_weekday{std::chrono::May, std::chrono::weekday_indexed{std::chrono::weekday{0}, 1}});
  check(loc,
        SV("%b=' 6'\t%B='6月'\t%h=' 6'\t%m='06'\t%Om='06'\n"),
        lfmt,
        std::chrono::month_weekday{std::chrono::June, std::chrono::weekday_indexed{std::chrono::weekday{0}, 1}});
  check(loc,
        SV("%b=' 7'\t%B='7月'\t%h=' 7'\t%m='07'\t%Om='07'\n"),
        lfmt,
        std::chrono::month_weekday{std::chrono::July, std::chrono::weekday_indexed{std::chrono::weekday{0}, 1}});
  check(loc,
        SV("%b=' 8'\t%B='8月'\t%h=' 8'\t%m='08'\t%Om='08'\n"),
        lfmt,
        std::chrono::month_weekday{std::chrono::August, std::chrono::weekday_indexed{std::chrono::weekday{0}, 1}});
  check(loc,
        SV("%b=' 9'\t%B='9月'\t%h=' 9'\t%m='09'\t%Om='09'\n"),
        lfmt,
        std::chrono::month_weekday{std::chrono::September, std::chrono::weekday_indexed{std::chrono::weekday{0}, 1}});
  check(loc,
        SV("%b='10'\t%B='10月'\t%h='10'\t%m='10'\t%Om='10'\n"),
        lfmt,
        std::chrono::month_weekday{std::chrono::October, std::chrono::weekday_indexed{std::chrono::weekday{0}, 1}});
  check(loc,
        SV("%b='11'\t%B='11月'\t%h='11'\t%m='11'\t%Om='11'\n"),
        lfmt,
        std::chrono::month_weekday{std::chrono::November, std::chrono::weekday_indexed{std::chrono::weekday{0}, 1}});
  check(loc,
        SV("%b='12'\t%B='12月'\t%h='12'\t%m='12'\t%Om='12'\n"),
        lfmt,
        std::chrono::month_weekday{std::chrono::December, std::chrono::weekday_indexed{std::chrono::weekday{0}, 1}});
#elif defined(_AIX)      // _WIN32
  check(loc,
        SV("%b='1月'\t%B='1月'\t%h='1月'\t%m='01'\t%Om='01'\n"),
        lfmt,
        std::chrono::month_weekday{std::chrono::January, std::chrono::weekday_indexed{std::chrono::weekday{0}, 1}});
  check(loc,
        SV("%b='2月'\t%B='2月'\t%h='2月'\t%m='02'\t%Om='02'\n"),
        lfmt,
        std::chrono::month_weekday{std::chrono::February, std::chrono::weekday_indexed{std::chrono::weekday{0}, 1}});
  check(loc,
        SV("%b='3月'\t%B='3月'\t%h='3月'\t%m='03'\t%Om='03'\n"),
        lfmt,
        std::chrono::month_weekday{std::chrono::March, std::chrono::weekday_indexed{std::chrono::weekday{0}, 1}});
  check(loc,
        SV("%b='4月'\t%B='4月'\t%h='4月'\t%m='04'\t%Om='04'\n"),
        lfmt,
        std::chrono::month_weekday{std::chrono::April, std::chrono::weekday_indexed{std::chrono::weekday{0}, 1}});
  check(loc,
        SV("%b='5月'\t%B='5月'\t%h='5月'\t%m='05'\t%Om='05'\n"),
        lfmt,
        std::chrono::month_weekday{std::chrono::May, std::chrono::weekday_indexed{std::chrono::weekday{0}, 1}});
  check(loc,
        SV("%b='6月'\t%B='6月'\t%h='6月'\t%m='06'\t%Om='06'\n"),
        lfmt,
        std::chrono::month_weekday{std::chrono::June, std::chrono::weekday_indexed{std::chrono::weekday{0}, 1}});
  check(loc,
        SV("%b='7月'\t%B='7月'\t%h='7月'\t%m='07'\t%Om='07'\n"),
        lfmt,
        std::chrono::month_weekday{std::chrono::July, std::chrono::weekday_indexed{std::chrono::weekday{0}, 1}});
  check(loc,
        SV("%b='8月'\t%B='8月'\t%h='8月'\t%m='08'\t%Om='08'\n"),
        lfmt,
        std::chrono::month_weekday{std::chrono::August, std::chrono::weekday_indexed{std::chrono::weekday{0}, 1}});
  check(loc,
        SV("%b='9月'\t%B='9月'\t%h='9月'\t%m='09'\t%Om='09'\n"),
        lfmt,
        std::chrono::month_weekday{std::chrono::September, std::chrono::weekday_indexed{std::chrono::weekday{0}, 1}});
  check(loc,
        SV("%b='10月'\t%B='10月'\t%h='10月'\t%m='10'\t%Om='10'\n"),
        lfmt,
        std::chrono::month_weekday{std::chrono::October, std::chrono::weekday_indexed{std::chrono::weekday{0}, 1}});
  check(loc,
        SV("%b='11月'\t%B='11月'\t%h='11月'\t%m='11'\t%Om='11'\n"),
        lfmt,
        std::chrono::month_weekday{std::chrono::November, std::chrono::weekday_indexed{std::chrono::weekday{0}, 1}});
  check(loc,
        SV("%b='12月'\t%B='12月'\t%h='12月'\t%m='12'\t%Om='12'\n"),
        lfmt,
        std::chrono::month_weekday{std::chrono::December, std::chrono::weekday_indexed{std::chrono::weekday{0}, 1}});
#else                    // _WIN32
  check(loc,
        SV("%b=' 1月'\t%B='1月'\t%h=' 1月'\t%m='01'\t%Om='一'\n"),
        lfmt,
        std::chrono::month_weekday{std::chrono::January, std::chrono::weekday_indexed{std::chrono::weekday{0}, 1}});
  check(loc,
        SV("%b=' 2月'\t%B='2月'\t%h=' 2月'\t%m='02'\t%Om='二'\n"),
        lfmt,
        std::chrono::month_weekday{std::chrono::February, std::chrono::weekday_indexed{std::chrono::weekday{0}, 1}});
  check(loc,
        SV("%b=' 3月'\t%B='3月'\t%h=' 3月'\t%m='03'\t%Om='三'\n"),
        lfmt,
        std::chrono::month_weekday{std::chrono::March, std::chrono::weekday_indexed{std::chrono::weekday{0}, 1}});
  check(loc,
        SV("%b=' 4月'\t%B='4月'\t%h=' 4月'\t%m='04'\t%Om='四'\n"),
        lfmt,
        std::chrono::month_weekday{std::chrono::April, std::chrono::weekday_indexed{std::chrono::weekday{0}, 1}});
  check(loc,
        SV("%b=' 5月'\t%B='5月'\t%h=' 5月'\t%m='05'\t%Om='五'\n"),
        lfmt,
        std::chrono::month_weekday{std::chrono::May, std::chrono::weekday_indexed{std::chrono::weekday{0}, 1}});
  check(loc,
        SV("%b=' 6月'\t%B='6月'\t%h=' 6月'\t%m='06'\t%Om='六'\n"),
        lfmt,
        std::chrono::month_weekday{std::chrono::June, std::chrono::weekday_indexed{std::chrono::weekday{0}, 1}});
  check(loc,
        SV("%b=' 7月'\t%B='7月'\t%h=' 7月'\t%m='07'\t%Om='七'\n"),
        lfmt,
        std::chrono::month_weekday{std::chrono::July, std::chrono::weekday_indexed{std::chrono::weekday{0}, 1}});
  check(loc,
        SV("%b=' 8月'\t%B='8月'\t%h=' 8月'\t%m='08'\t%Om='八'\n"),
        lfmt,
        std::chrono::month_weekday{std::chrono::August, std::chrono::weekday_indexed{std::chrono::weekday{0}, 1}});
  check(loc,
        SV("%b=' 9月'\t%B='9月'\t%h=' 9月'\t%m='09'\t%Om='九'\n"),
        lfmt,
        std::chrono::month_weekday{std::chrono::September, std::chrono::weekday_indexed{std::chrono::weekday{0}, 1}});
  check(loc,
        SV("%b='10月'\t%B='10月'\t%h='10月'\t%m='10'\t%Om='十'\n"),
        lfmt,
        std::chrono::month_weekday{std::chrono::October, std::chrono::weekday_indexed{std::chrono::weekday{0}, 1}});
  check(loc,
        SV("%b='11月'\t%B='11月'\t%h='11月'\t%m='11'\t%Om='十一'\n"),
        lfmt,
        std::chrono::month_weekday{std::chrono::November, std::chrono::weekday_indexed{std::chrono::weekday{0}, 1}});
  check(loc,
        SV("%b='12月'\t%B='12月'\t%h='12月'\t%m='12'\t%Om='十二'\n"),
        lfmt,
        std::chrono::month_weekday{std::chrono::December, std::chrono::weekday_indexed{std::chrono::weekday{0}, 1}});
#endif                   // _WIN32

  std::locale::global(std::locale::classic());
}

template <class CharT>
static void test_valid_weekday() {
  constexpr std::basic_string_view<CharT> fmt =
      SV("{:%%u='%u'%t%%Ou='%Ou'%t%%w='%w'%t%%Ow='%Ow'%t%%a='%a'%t%%A='%A'%n}");
  constexpr std::basic_string_view<CharT> lfmt =
      SV("{:L%%u='%u'%t%%Ou='%Ou'%t%%w='%w'%t%%Ow='%Ow'%t%%a='%a'%t%%A='%A'%n}");

  const std::locale loc(LOCALE_ja_JP_UTF_8);
  std::locale::global(std::locale(LOCALE_fr_FR_UTF_8));

  // Non localized output using C-locale
  check(SV("%u='7'\t%Ou='7'\t%w='0'\t%Ow='0'\t%a='Sun'\t%A='Sunday'\n"),
        fmt,
        std::chrono::month_weekday{std::chrono::January, std::chrono::weekday_indexed{std::chrono::weekday{0}, 1}});
  check(SV("%u='1'\t%Ou='1'\t%w='1'\t%Ow='1'\t%a='Mon'\t%A='Monday'\n"),
        fmt,
        std::chrono::month_weekday{std::chrono::January, std::chrono::weekday_indexed{std::chrono::weekday{1}, 1}});
  check(SV("%u='2'\t%Ou='2'\t%w='2'\t%Ow='2'\t%a='Tue'\t%A='Tuesday'\n"),
        fmt,
        std::chrono::month_weekday{std::chrono::January, std::chrono::weekday_indexed{std::chrono::weekday{2}, 1}});
  check(SV("%u='3'\t%Ou='3'\t%w='3'\t%Ow='3'\t%a='Wed'\t%A='Wednesday'\n"),
        fmt,
        std::chrono::month_weekday{std::chrono::January, std::chrono::weekday_indexed{std::chrono::weekday{3}, 1}});
  check(SV("%u='4'\t%Ou='4'\t%w='4'\t%Ow='4'\t%a='Thu'\t%A='Thursday'\n"),
        fmt,
        std::chrono::month_weekday{std::chrono::January, std::chrono::weekday_indexed{std::chrono::weekday{4}, 1}});
  check(SV("%u='5'\t%Ou='5'\t%w='5'\t%Ow='5'\t%a='Fri'\t%A='Friday'\n"),
        fmt,
        std::chrono::month_weekday{std::chrono::January, std::chrono::weekday_indexed{std::chrono::weekday{5}, 1}});
  check(SV("%u='6'\t%Ou='6'\t%w='6'\t%Ow='6'\t%a='Sat'\t%A='Saturday'\n"),
        fmt,
        std::chrono::month_weekday{std::chrono::January, std::chrono::weekday_indexed{std::chrono::weekday{6}, 1}});
  check(SV("%u='7'\t%Ou='7'\t%w='0'\t%Ow='0'\t%a='Sun'\t%A='Sunday'\n"),
        fmt,
        std::chrono::month_weekday{std::chrono::January, std::chrono::weekday_indexed{std::chrono::weekday{7}, 1}});

  // Use the global locale (fr_FR)
#if defined(__APPLE__)
  check(SV("%u='7'\t%Ou='7'\t%w='0'\t%Ow='0'\t%a='Dim'\t%A='Dimanche'\n"),
        lfmt,
        std::chrono::month_weekday{std::chrono::January, std::chrono::weekday_indexed{std::chrono::weekday{0}, 1}});
  check(SV("%u='1'\t%Ou='1'\t%w='1'\t%Ow='1'\t%a='Lun'\t%A='Lundi'\n"),
        lfmt,
        std::chrono::month_weekday{std::chrono::January, std::chrono::weekday_indexed{std::chrono::weekday{1}, 1}});
  check(SV("%u='2'\t%Ou='2'\t%w='2'\t%Ow='2'\t%a='Mar'\t%A='Mardi'\n"),
        lfmt,
        std::chrono::month_weekday{std::chrono::January, std::chrono::weekday_indexed{std::chrono::weekday{2}, 1}});
  check(SV("%u='3'\t%Ou='3'\t%w='3'\t%Ow='3'\t%a='Mer'\t%A='Mercredi'\n"),
        lfmt,
        std::chrono::month_weekday{std::chrono::January, std::chrono::weekday_indexed{std::chrono::weekday{3}, 1}});
  check(SV("%u='4'\t%Ou='4'\t%w='4'\t%Ow='4'\t%a='Jeu'\t%A='Jeudi'\n"),
        lfmt,
        std::chrono::month_weekday{std::chrono::January, std::chrono::weekday_indexed{std::chrono::weekday{4}, 1}});
  check(SV("%u='5'\t%Ou='5'\t%w='5'\t%Ow='5'\t%a='Ven'\t%A='Vendredi'\n"),
        lfmt,
        std::chrono::month_weekday{std::chrono::January, std::chrono::weekday_indexed{std::chrono::weekday{5}, 1}});
  check(SV("%u='6'\t%Ou='6'\t%w='6'\t%Ow='6'\t%a='Sam'\t%A='Samedi'\n"),
        lfmt,
        std::chrono::month_weekday{std::chrono::January, std::chrono::weekday_indexed{std::chrono::weekday{6}, 1}});
  check(SV("%u='7'\t%Ou='7'\t%w='0'\t%Ow='0'\t%a='Dim'\t%A='Dimanche'\n"),
        lfmt,
        std::chrono::month_weekday{std::chrono::January, std::chrono::weekday_indexed{std::chrono::weekday{7}, 1}});
#else  // defined(__APPLE__)
  check(SV("%u='7'\t%Ou='7'\t%w='0'\t%Ow='0'\t%a='dim.'\t%A='dimanche'\n"),
        lfmt,
        std::chrono::month_weekday{std::chrono::January, std::chrono::weekday_indexed{std::chrono::weekday{0}, 1}});
  check(SV("%u='1'\t%Ou='1'\t%w='1'\t%Ow='1'\t%a='lun.'\t%A='lundi'\n"),
        lfmt,
        std::chrono::month_weekday{std::chrono::January, std::chrono::weekday_indexed{std::chrono::weekday{1}, 1}});
  check(SV("%u='2'\t%Ou='2'\t%w='2'\t%Ow='2'\t%a='mar.'\t%A='mardi'\n"),
        lfmt,
        std::chrono::month_weekday{std::chrono::January, std::chrono::weekday_indexed{std::chrono::weekday{2}, 1}});
  check(SV("%u='3'\t%Ou='3'\t%w='3'\t%Ow='3'\t%a='mer.'\t%A='mercredi'\n"),
        lfmt,
        std::chrono::month_weekday{std::chrono::January, std::chrono::weekday_indexed{std::chrono::weekday{3}, 1}});
  check(SV("%u='4'\t%Ou='4'\t%w='4'\t%Ow='4'\t%a='jeu.'\t%A='jeudi'\n"),
        lfmt,
        std::chrono::month_weekday{std::chrono::January, std::chrono::weekday_indexed{std::chrono::weekday{4}, 1}});
  check(SV("%u='5'\t%Ou='5'\t%w='5'\t%Ow='5'\t%a='ven.'\t%A='vendredi'\n"),
        lfmt,
        std::chrono::month_weekday{std::chrono::January, std::chrono::weekday_indexed{std::chrono::weekday{5}, 1}});
  check(SV("%u='6'\t%Ou='6'\t%w='6'\t%Ow='6'\t%a='sam.'\t%A='samedi'\n"),
        lfmt,
        std::chrono::month_weekday{std::chrono::January, std::chrono::weekday_indexed{std::chrono::weekday{6}, 1}});
  check(SV("%u='7'\t%Ou='7'\t%w='0'\t%Ow='0'\t%a='dim.'\t%A='dimanche'\n"),
        lfmt,
        std::chrono::month_weekday{std::chrono::January, std::chrono::weekday_indexed{std::chrono::weekday{7}, 1}});
#endif // defined(__APPLE__)

  // Use supplied locale (ja_JP).
  // This locale has a different alternate, but not on all platforms
#if defined(_WIN32) || defined(__APPLE__) || defined(_AIX)
  check(loc,
        SV("%u='7'\t%Ou='7'\t%w='0'\t%Ow='0'\t%a='日'\t%A='日曜日'\n"),
        lfmt,
        std::chrono::month_weekday{std::chrono::January, std::chrono::weekday_indexed{std::chrono::weekday{0}, 1}});
  check(loc,
        SV("%u='1'\t%Ou='1'\t%w='1'\t%Ow='1'\t%a='月'\t%A='月曜日'\n"),
        lfmt,
        std::chrono::month_weekday{std::chrono::January, std::chrono::weekday_indexed{std::chrono::weekday{1}, 1}});
  check(loc,
        SV("%u='2'\t%Ou='2'\t%w='2'\t%Ow='2'\t%a='火'\t%A='火曜日'\n"),
        lfmt,
        std::chrono::month_weekday{std::chrono::January, std::chrono::weekday_indexed{std::chrono::weekday{2}, 1}});
  check(loc,
        SV("%u='3'\t%Ou='3'\t%w='3'\t%Ow='3'\t%a='水'\t%A='水曜日'\n"),
        lfmt,
        std::chrono::month_weekday{std::chrono::January, std::chrono::weekday_indexed{std::chrono::weekday{3}, 1}});
  check(loc,
        SV("%u='4'\t%Ou='4'\t%w='4'\t%Ow='4'\t%a='木'\t%A='木曜日'\n"),
        lfmt,
        std::chrono::month_weekday{std::chrono::January, std::chrono::weekday_indexed{std::chrono::weekday{4}, 1}});
  check(loc,
        SV("%u='5'\t%Ou='5'\t%w='5'\t%Ow='5'\t%a='金'\t%A='金曜日'\n"),
        lfmt,
        std::chrono::month_weekday{std::chrono::January, std::chrono::weekday_indexed{std::chrono::weekday{5}, 1}});
  check(loc,
        SV("%u='6'\t%Ou='6'\t%w='6'\t%Ow='6'\t%a='土'\t%A='土曜日'\n"),
        lfmt,
        std::chrono::month_weekday{std::chrono::January, std::chrono::weekday_indexed{std::chrono::weekday{6}, 1}});
  check(loc,
        SV("%u='7'\t%Ou='7'\t%w='0'\t%Ow='0'\t%a='日'\t%A='日曜日'\n"),
        lfmt,
        std::chrono::month_weekday{std::chrono::January, std::chrono::weekday_indexed{std::chrono::weekday{7}, 1}});
#else  // defined(_WIN32) || defined(__APPLE__) || defined(_AIX)
  check(loc,
        SV("%u='7'\t%Ou='七'\t%w='0'\t%Ow='〇'\t%a='日'\t%A='日曜日'\n"),
        lfmt,
        std::chrono::month_weekday{std::chrono::January, std::chrono::weekday_indexed{std::chrono::weekday{0}, 1}});
  check(loc,
        SV("%u='1'\t%Ou='一'\t%w='1'\t%Ow='一'\t%a='月'\t%A='月曜日'\n"),
        lfmt,
        std::chrono::month_weekday{std::chrono::January, std::chrono::weekday_indexed{std::chrono::weekday{1}, 1}});
  check(loc,
        SV("%u='2'\t%Ou='二'\t%w='2'\t%Ow='二'\t%a='火'\t%A='火曜日'\n"),
        lfmt,
        std::chrono::month_weekday{std::chrono::January, std::chrono::weekday_indexed{std::chrono::weekday{2}, 1}});
  check(loc,
        SV("%u='3'\t%Ou='三'\t%w='3'\t%Ow='三'\t%a='水'\t%A='水曜日'\n"),
        lfmt,
        std::chrono::month_weekday{std::chrono::January, std::chrono::weekday_indexed{std::chrono::weekday{3}, 1}});
  check(loc,
        SV("%u='4'\t%Ou='四'\t%w='4'\t%Ow='四'\t%a='木'\t%A='木曜日'\n"),
        lfmt,
        std::chrono::month_weekday{std::chrono::January, std::chrono::weekday_indexed{std::chrono::weekday{4}, 1}});
  check(loc,
        SV("%u='5'\t%Ou='五'\t%w='5'\t%Ow='五'\t%a='金'\t%A='金曜日'\n"),
        lfmt,
        std::chrono::month_weekday{std::chrono::January, std::chrono::weekday_indexed{std::chrono::weekday{5}, 1}});
  check(loc,
        SV("%u='6'\t%Ou='六'\t%w='6'\t%Ow='六'\t%a='土'\t%A='土曜日'\n"),
        lfmt,
        std::chrono::month_weekday{std::chrono::January, std::chrono::weekday_indexed{std::chrono::weekday{6}, 1}});
  check(loc,
        SV("%u='7'\t%Ou='七'\t%w='0'\t%Ow='〇'\t%a='日'\t%A='日曜日'\n"),
        lfmt,
        std::chrono::month_weekday{std::chrono::January, std::chrono::weekday_indexed{std::chrono::weekday{7}, 1}});
#endif // defined(_WIN32) || defined(__APPLE__) || defined(_AIX)

  std::locale::global(std::locale::classic());
}

template <class CharT>
static void test_valid_values() {
  test_valid_month<CharT>();
  test_valid_weekday<CharT>();
}

template <class CharT>
static void test() {
  test_no_chrono_specs<CharT>();
  test_invalid_values<CharT>();
  test_valid_values<CharT>();
  check_invalid_types<CharT>(
      {SV("a"), SV("A"), SV("b"), SV("B"), SV("h"), SV("m"), SV("u"), SV("w"), SV("Om"), SV("Ou"), SV("Ow")},
      std::chrono::month_weekday{std::chrono::January, std::chrono::weekday_indexed{std::chrono::weekday{0}, 1}});

  check_exception(
      "Expected '%' or '}' in the chrono format-string",
      SV("{:A"),
      std::chrono::month_weekday{std::chrono::January, std::chrono::weekday_indexed{std::chrono::weekday{0}, 1}});
  check_exception(
      "The chrono-specs contains a '{'",
      SV("{:%%{"),
      std::chrono::month_weekday{std::chrono::January, std::chrono::weekday_indexed{std::chrono::weekday{0}, 1}});
  check_exception(
      "End of input while parsing the modifier chrono conversion-spec",
      SV("{:%"),
      std::chrono::month_weekday{std::chrono::January, std::chrono::weekday_indexed{std::chrono::weekday{0}, 1}});
  check_exception(
      "End of input while parsing the modifier E",
      SV("{:%E"),
      std::chrono::month_weekday{std::chrono::January, std::chrono::weekday_indexed{std::chrono::weekday{0}, 1}});
  check_exception(
      "End of input while parsing the modifier O",
      SV("{:%O"),
      std::chrono::month_weekday{std::chrono::January, std::chrono::weekday_indexed{std::chrono::weekday{0}, 1}});

  // Precision not allowed
  check_exception(
      "Expected '%' or '}' in the chrono format-string",
      SV("{:.3}"),
      std::chrono::month_weekday{std::chrono::January, std::chrono::weekday_indexed{std::chrono::weekday{0}, 1}});
}

int main(int, char**) {
  test<char>();

#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  test<wchar_t>();
#endif

  return 0;
}
