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

// template<class charT> struct formatter<chrono::year_month_day_last, charT>;

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
  // Valid year, valid month
  check(SV("1970/Jan/last"),
        SV("{}"),
        std::chrono::year_month_day_last{std::chrono::year{1970}, std::chrono::month_day_last{std::chrono::month{1}}});
  check(SV("*1970/Jan/last*"),
        SV("{:*^15}"),
        std::chrono::year_month_day_last{std::chrono::year{1970}, std::chrono::month_day_last{std::chrono::month{1}}});
  check(SV("*1970/Jan/last"),
        SV("{:*>14}"),
        std::chrono::year_month_day_last{std::chrono::year{1970}, std::chrono::month_day_last{std::chrono::month{1}}});

  // Valid year, invalid month
  check(SV("1970/0 is not a valid month/last"),
        SV("{}"),
        std::chrono::year_month_day_last{std::chrono::year{1970}, std::chrono::month_day_last{std::chrono::month{0}}});
  check(SV("*1970/0 is not a valid month/last*"),
        SV("{:*^34}"),
        std::chrono::year_month_day_last{std::chrono::year{1970}, std::chrono::month_day_last{std::chrono::month{0}}});
  check(SV("*1970/0 is not a valid month/last"),
        SV("{:*>33}"),
        std::chrono::year_month_day_last{std::chrono::year{1970}, std::chrono::month_day_last{std::chrono::month{0}}});

  // Invalid year, valid month
  check(
      SV("-32768 is not a valid year/Jan/last"),
      SV("{}"),
      std::chrono::year_month_day_last{std::chrono::year{-32768}, std::chrono::month_day_last{std::chrono::month{1}}});
  check(
      SV("*-32768 is not a valid year/Jan/last*"),
      SV("{:*^37}"),
      std::chrono::year_month_day_last{std::chrono::year{-32768}, std::chrono::month_day_last{std::chrono::month{1}}});
  check(
      SV("*-32768 is not a valid year/Jan/last"),
      SV("{:*>36}"),
      std::chrono::year_month_day_last{std::chrono::year{-32768}, std::chrono::month_day_last{std::chrono::month{1}}});

  // Invalid year, invalid month
  check(
      SV("-32768 is not a valid year/0 is not a valid month/last"),
      SV("{}"),
      std::chrono::year_month_day_last{std::chrono::year{-32768}, std::chrono::month_day_last{std::chrono::month{0}}});
  check(
      SV("*-32768 is not a valid year/0 is not a valid month/last*"),
      SV("{:*^56}"),
      std::chrono::year_month_day_last{std::chrono::year{-32768}, std::chrono::month_day_last{std::chrono::month{0}}});
  check(
      SV("*-32768 is not a valid year/0 is not a valid month/last"),
      SV("{:*>55}"),
      std::chrono::year_month_day_last{std::chrono::year{-32768}, std::chrono::month_day_last{std::chrono::month{0}}});
}

// TODO FMT Should x throw?
template <class CharT>
static void test_invalid_values() {
  // Test that %a, %A, %b, %B, %h, %j, %u, %U, %V, %w, %W, %Ou, %OU, %OV, %Ow, and %OW throw an exception.
  check_exception(
      "formatting a weekday name needs a valid weekday",
      SV("{:%A}"),
      std::chrono::year_month_day_last{std::chrono::year{1970}, std::chrono::month_day_last{std::chrono::month{0}}});
  check_exception(
      "formatting a weekday name needs a valid weekday",
      SV("{:%A}"),
      std::chrono::year_month_day_last{std::chrono::year{-32768}, std::chrono::month_day_last{std::chrono::month{1}}});

  check_exception(
      "formatting a weekday name needs a valid weekday",
      SV("{:%a}"),
      std::chrono::year_month_day_last{std::chrono::year{1970}, std::chrono::month_day_last{std::chrono::month{0}}});
  check_exception(
      "formatting a weekday name needs a valid weekday",
      SV("{:%a}"),
      std::chrono::year_month_day_last{std::chrono::year{-32768}, std::chrono::month_day_last{std::chrono::month{1}}});

  check_exception(
      "formatting a month name from an invalid month number",
      SV("{:%B}"),
      std::chrono::year_month_day_last{std::chrono::year{1970}, std::chrono::month_day_last{std::chrono::month{0}}});
  check_exception(
      "formatting a month name from an invalid month number",
      SV("{:%B}"),
      std::chrono::year_month_day_last{std::chrono::year{1970}, std::chrono::month_day_last{std::chrono::month{13}}});
  check_exception(
      "formatting a month name from an invalid month number",
      SV("{:%B}"),
      std::chrono::year_month_day_last{std::chrono::year{1970}, std::chrono::month_day_last{std::chrono::month{255}}});

  check_exception(
      "formatting a month name from an invalid month number",
      SV("{:%b}"),
      std::chrono::year_month_day_last{std::chrono::year{1970}, std::chrono::month_day_last{std::chrono::month{200}}});
  check_exception(
      "formatting a month name from an invalid month number",
      SV("{:%b}"),
      std::chrono::year_month_day_last{std::chrono::year{1970}, std::chrono::month_day_last{std::chrono::month{13}}});
  check_exception(
      "formatting a month name from an invalid month number",
      SV("{:%b}"),
      std::chrono::year_month_day_last{std::chrono::year{1970}, std::chrono::month_day_last{std::chrono::month{255}}});

  check_exception(
      "formatting a month name from an invalid month number",
      SV("{:%h}"),
      std::chrono::year_month_day_last{std::chrono::year{1970}, std::chrono::month_day_last{std::chrono::month{0}}});
  check_exception(
      "formatting a month name from an invalid month number",
      SV("{:%h}"),
      std::chrono::year_month_day_last{std::chrono::year{1970}, std::chrono::month_day_last{std::chrono::month{13}}});
  check_exception(
      "formatting a month name from an invalid month number",
      SV("{:%h}"),
      std::chrono::year_month_day_last{std::chrono::year{1970}, std::chrono::month_day_last{std::chrono::month{255}}});

  check_exception(
      "formatting a day of year needs a valid date",
      SV("{:%j}"),
      std::chrono::year_month_day_last{std::chrono::year{1970}, std::chrono::month_day_last{std::chrono::month{0}}});
  check_exception(
      "formatting a day of year needs a valid date",
      SV("{:%j}"),
      std::chrono::year_month_day_last{std::chrono::year{-32768}, std::chrono::month_day_last{std::chrono::month{1}}});

  check_exception(
      "formatting a weekday needs a valid weekday",
      SV("{:%u}"),
      std::chrono::year_month_day_last{std::chrono::year{1970}, std::chrono::month_day_last{std::chrono::month{0}}});
  check_exception(
      "formatting a weekday needs a valid weekday",
      SV("{:%u}"),
      std::chrono::year_month_day_last{std::chrono::year{-32768}, std::chrono::month_day_last{std::chrono::month{1}}});

  check_exception(
      "formatting a week of year needs a valid date",
      SV("{:%U}"),
      std::chrono::year_month_day_last{std::chrono::year{1970}, std::chrono::month_day_last{std::chrono::month{0}}});
  check_exception(
      "formatting a week of year needs a valid date",
      SV("{:%U}"),
      std::chrono::year_month_day_last{std::chrono::year{-32768}, std::chrono::month_day_last{std::chrono::month{1}}});

  check_exception(
      "formatting a week of year needs a valid date",
      SV("{:%V}"),
      std::chrono::year_month_day_last{std::chrono::year{1970}, std::chrono::month_day_last{std::chrono::month{0}}});
  check_exception(
      "formatting a week of year needs a valid date",
      SV("{:%V}"),
      std::chrono::year_month_day_last{std::chrono::year{-32768}, std::chrono::month_day_last{std::chrono::month{1}}});

  check_exception(
      "formatting a weekday needs a valid weekday",
      SV("{:%w}"),
      std::chrono::year_month_day_last{std::chrono::year{1970}, std::chrono::month_day_last{std::chrono::month{0}}});
  check_exception(
      "formatting a weekday needs a valid weekday",
      SV("{:%w}"),
      std::chrono::year_month_day_last{std::chrono::year{-32768}, std::chrono::month_day_last{std::chrono::month{1}}});

  check_exception(
      "formatting a week of year needs a valid date",
      SV("{:%W}"),
      std::chrono::year_month_day_last{std::chrono::year{1970}, std::chrono::month_day_last{std::chrono::month{0}}});
  check_exception(
      "formatting a week of year needs a valid date",
      SV("{:%W}"),
      std::chrono::year_month_day_last{std::chrono::year{-32768}, std::chrono::month_day_last{std::chrono::month{1}}});

  check_exception(
      "formatting a weekday needs a valid weekday",
      SV("{:%Ou}"),
      std::chrono::year_month_day_last{std::chrono::year{1970}, std::chrono::month_day_last{std::chrono::month{0}}});
  check_exception(
      "formatting a weekday needs a valid weekday",
      SV("{:%Ou}"),
      std::chrono::year_month_day_last{std::chrono::year{-32768}, std::chrono::month_day_last{std::chrono::month{1}}});

  check_exception(
      "formatting a week of year needs a valid date",
      SV("{:%OU}"),
      std::chrono::year_month_day_last{std::chrono::year{1970}, std::chrono::month_day_last{std::chrono::month{0}}});
  check_exception(
      "formatting a week of year needs a valid date",
      SV("{:%OU}"),
      std::chrono::year_month_day_last{std::chrono::year{-32768}, std::chrono::month_day_last{std::chrono::month{1}}});

  check_exception(
      "formatting a week of year needs a valid date",
      SV("{:%OV}"),
      std::chrono::year_month_day_last{std::chrono::year{1970}, std::chrono::month_day_last{std::chrono::month{0}}});
  check_exception(
      "formatting a week of year needs a valid date",
      SV("{:%OV}"),
      std::chrono::year_month_day_last{std::chrono::year{-32768}, std::chrono::month_day_last{std::chrono::month{1}}});

  check_exception(
      "formatting a weekday needs a valid weekday",
      SV("{:%Ow}"),
      std::chrono::year_month_day_last{std::chrono::year{1970}, std::chrono::month_day_last{std::chrono::month{0}}});
  check_exception(
      "formatting a weekday needs a valid weekday",
      SV("{:%Ow}"),
      std::chrono::year_month_day_last{std::chrono::year{-32768}, std::chrono::month_day_last{std::chrono::month{1}}});

  check_exception(
      "formatting a week of year needs a valid date",
      SV("{:%OW}"),
      std::chrono::year_month_day_last{std::chrono::year{1970}, std::chrono::month_day_last{std::chrono::month{0}}});
  check_exception(
      "formatting a week of year needs a valid date",
      SV("{:%OW}"),
      std::chrono::year_month_day_last{std::chrono::year{-32768}, std::chrono::month_day_last{std::chrono::month{1}}});
}

template <class CharT>
static void test_valid_md_values() {
  constexpr std::basic_string_view<CharT> fmt =
      SV("{:%%b='%b'%t%%B='%B'%t%%h='%h'%t%%m='%m'%t%%Om='%Om'%t%%d='%d'%t%%e='%e'%t%%Od='%Od'%t%%Oe='%Oe'%n}");
  constexpr std::basic_string_view<CharT> lfmt =
      SV("{:L%%b='%b'%t%%B='%B'%t%%h='%h'%t%%m='%m'%t%%Om='%Om'%t%%d='%d'%t%%e='%e'%t%%Od='%Od'%t%%Oe='%Oe'%n}");

  const std::locale loc(LOCALE_ja_JP_UTF_8);
  std::locale::global(std::locale(LOCALE_fr_FR_UTF_8));

  // Non localized output using C-locale
  check(SV("%b='Jan'\t%B='January'\t%h='Jan'\t%m='01'\t%Om='01'\t%d='31'\t%e='31'\t%Od='31'\t%Oe='31'\n"),
        fmt,
        std::chrono::year_month_day_last{std::chrono::year{1970}, std::chrono::month_day_last{std::chrono::January}});
  check(SV("%b='Feb'\t%B='February'\t%h='Feb'\t%m='02'\t%Om='02'\t%d='28'\t%e='28'\t%Od='28'\t%Oe='28'\n"),
        fmt,
        std::chrono::year_month_day_last{std::chrono::year{1970}, std::chrono::month_day_last{std::chrono::February}});
  check(SV("%b='Mar'\t%B='March'\t%h='Mar'\t%m='03'\t%Om='03'\t%d='31'\t%e='31'\t%Od='31'\t%Oe='31'\n"),
        fmt,
        std::chrono::year_month_day_last{std::chrono::year{1970}, std::chrono::month_day_last{std::chrono::March}});
  check(SV("%b='Apr'\t%B='April'\t%h='Apr'\t%m='04'\t%Om='04'\t%d='30'\t%e='30'\t%Od='30'\t%Oe='30'\n"),
        fmt,
        std::chrono::year_month_day_last{std::chrono::year{1970}, std::chrono::month_day_last{std::chrono::April}});
  check(SV("%b='May'\t%B='May'\t%h='May'\t%m='05'\t%Om='05'\t%d='31'\t%e='31'\t%Od='31'\t%Oe='31'\n"),
        fmt,
        std::chrono::year_month_day_last{std::chrono::year{1970}, std::chrono::month_day_last{std::chrono::May}});
  check(SV("%b='Jun'\t%B='June'\t%h='Jun'\t%m='06'\t%Om='06'\t%d='30'\t%e='30'\t%Od='30'\t%Oe='30'\n"),
        fmt,
        std::chrono::year_month_day_last{std::chrono::year{1970}, std::chrono::month_day_last{std::chrono::June}});
  check(SV("%b='Jul'\t%B='July'\t%h='Jul'\t%m='07'\t%Om='07'\t%d='31'\t%e='31'\t%Od='31'\t%Oe='31'\n"),
        fmt,
        std::chrono::year_month_day_last{std::chrono::year{1970}, std::chrono::month_day_last{std::chrono::July}});
  check(SV("%b='Aug'\t%B='August'\t%h='Aug'\t%m='08'\t%Om='08'\t%d='31'\t%e='31'\t%Od='31'\t%Oe='31'\n"),
        fmt,
        std::chrono::year_month_day_last{std::chrono::year{1970}, std::chrono::month_day_last{std::chrono::August}});
  check(SV("%b='Sep'\t%B='September'\t%h='Sep'\t%m='09'\t%Om='09'\t%d='30'\t%e='30'\t%Od='30'\t%Oe='30'\n"),
        fmt,
        std::chrono::year_month_day_last{std::chrono::year{1970}, std::chrono::month_day_last{std::chrono::September}});
  check(SV("%b='Oct'\t%B='October'\t%h='Oct'\t%m='10'\t%Om='10'\t%d='31'\t%e='31'\t%Od='31'\t%Oe='31'\n"),
        fmt,
        std::chrono::year_month_day_last{std::chrono::year{1970}, std::chrono::month_day_last{std::chrono::October}});
  check(SV("%b='Nov'\t%B='November'\t%h='Nov'\t%m='11'\t%Om='11'\t%d='30'\t%e='30'\t%Od='30'\t%Oe='30'\n"),
        fmt,
        std::chrono::year_month_day_last{std::chrono::year{1970}, std::chrono::month_day_last{std::chrono::November}});
  check(SV("%b='Dec'\t%B='December'\t%h='Dec'\t%m='12'\t%Om='12'\t%d='31'\t%e='31'\t%Od='31'\t%Oe='31'\n"),
        fmt,
        std::chrono::year_month_day_last{std::chrono::year{1970}, std::chrono::month_day_last{std::chrono::December}});

  // Use the global locale (fr_FR)
#if defined(__APPLE__)
  check(SV("%b='jan'\t%B='janvier'\t%h='jan'\t%m='01'\t%Om='01'\t%d='31'\t%e='31'\t%Od='31'\t%Oe='31'\n"),
        lfmt,
        std::chrono::year_month_day_last{std::chrono::year{1970}, std::chrono::month_day_last{std::chrono::January}});
  check(SV("%b='fév'\t%B='février'\t%h='fév'\t%m='02'\t%Om='02'\t%d='28'\t%e='28'\t%Od='28'\t%Oe='28'\n"),
        lfmt,
        std::chrono::year_month_day_last{std::chrono::year{1970}, std::chrono::month_day_last{std::chrono::February}});
  check(SV("%b='mar'\t%B='mars'\t%h='mar'\t%m='03'\t%Om='03'\t%d='31'\t%e='31'\t%Od='31'\t%Oe='31'\n"),
        lfmt,
        std::chrono::year_month_day_last{std::chrono::year{1970}, std::chrono::month_day_last{std::chrono::March}});
  check(SV("%b='avr'\t%B='avril'\t%h='avr'\t%m='04'\t%Om='04'\t%d='30'\t%e='30'\t%Od='30'\t%Oe='30'\n"),
        lfmt,
        std::chrono::year_month_day_last{std::chrono::year{1970}, std::chrono::month_day_last{std::chrono::April}});
  check(SV("%b='mai'\t%B='mai'\t%h='mai'\t%m='05'\t%Om='05'\t%d='31'\t%e='31'\t%Od='31'\t%Oe='31'\n"),
        lfmt,
        std::chrono::year_month_day_last{std::chrono::year{1970}, std::chrono::month_day_last{std::chrono::May}});
  check(SV("%b='jui'\t%B='juin'\t%h='jui'\t%m='06'\t%Om='06'\t%d='30'\t%e='30'\t%Od='30'\t%Oe='30'\n"),
        lfmt,
        std::chrono::year_month_day_last{std::chrono::year{1970}, std::chrono::month_day_last{std::chrono::June}});
  check(SV("%b='jul'\t%B='juillet'\t%h='jul'\t%m='07'\t%Om='07'\t%d='31'\t%e='31'\t%Od='31'\t%Oe='31'\n"),
        lfmt,
        std::chrono::year_month_day_last{std::chrono::year{1970}, std::chrono::month_day_last{std::chrono::July}});
  check(SV("%b='aoû'\t%B='août'\t%h='aoû'\t%m='08'\t%Om='08'\t%d='31'\t%e='31'\t%Od='31'\t%Oe='31'\n"),
        lfmt,
        std::chrono::year_month_day_last{std::chrono::year{1970}, std::chrono::month_day_last{std::chrono::August}});
  check(SV("%b='sep'\t%B='septembre'\t%h='sep'\t%m='09'\t%Om='09'\t%d='30'\t%e='30'\t%Od='30'\t%Oe='30'\n"),
        lfmt,
        std::chrono::year_month_day_last{std::chrono::year{1970}, std::chrono::month_day_last{std::chrono::September}});
  check(SV("%b='oct'\t%B='octobre'\t%h='oct'\t%m='10'\t%Om='10'\t%d='31'\t%e='31'\t%Od='31'\t%Oe='31'\n"),
        lfmt,
        std::chrono::year_month_day_last{std::chrono::year{1970}, std::chrono::month_day_last{std::chrono::October}});
  check(SV("%b='nov'\t%B='novembre'\t%h='nov'\t%m='11'\t%Om='11'\t%d='30'\t%e='30'\t%Od='30'\t%Oe='30'\n"),
        lfmt,
        std::chrono::year_month_day_last{std::chrono::year{1970}, std::chrono::month_day_last{std::chrono::November}});
  check(SV("%b='déc'\t%B='décembre'\t%h='déc'\t%m='12'\t%Om='12'\t%d='31'\t%e='31'\t%Od='31'\t%Oe='31'\n"),
        lfmt,
        std::chrono::year_month_day_last{std::chrono::year{1970}, std::chrono::month_day_last{std::chrono::December}});
#else    // defined(__APPLE__)
  check(SV("%b='janv.'\t%B='janvier'\t%h='janv.'\t%m='01'\t%Om='01'\t%d='31'\t%e='31'\t%Od='31'\t%Oe='31'\n"),
        lfmt,
        std::chrono::year_month_day_last{std::chrono::year{1970}, std::chrono::month_day_last{std::chrono::January}});
  check(SV("%b='févr.'\t%B='février'\t%h='févr.'\t%m='02'\t%Om='02'\t%d='28'\t%e='28'\t%Od='28'\t%Oe='28'\n"),
        lfmt,
        std::chrono::year_month_day_last{std::chrono::year{1970}, std::chrono::month_day_last{std::chrono::February}});
  check(SV("%b='mars'\t%B='mars'\t%h='mars'\t%m='03'\t%Om='03'\t%d='31'\t%e='31'\t%Od='31'\t%Oe='31'\n"),
        lfmt,
        std::chrono::year_month_day_last{std::chrono::year{1970}, std::chrono::month_day_last{std::chrono::March}});
  check(
#  if defined(_WIN32) || defined(_AIX)
      SV("%b='avr.'\t%B='avril'\t%h='avr.'\t%m='04'\t%Om='04'\t%d='30'\t%e='30'\t%Od='30'\t%Oe='30'\n"),
#  else  // defined(_WIN32) || defined(_AIX)
      SV("%b='avril'\t%B='avril'\t%h='avril'\t%m='04'\t%Om='04'\t%d='30'\t%e='30'\t%Od='30'\t%Oe='30'\n"),
#  endif // defined(_WIN32) || defined(_AIX)
      lfmt,
      std::chrono::year_month_day_last{std::chrono::year{1970}, std::chrono::month_day_last{std::chrono::April}});
  check(SV("%b='mai'\t%B='mai'\t%h='mai'\t%m='05'\t%Om='05'\t%d='31'\t%e='31'\t%Od='31'\t%Oe='31'\n"),
        lfmt,
        std::chrono::year_month_day_last{std::chrono::year{1970}, std::chrono::month_day_last{std::chrono::May}});
  check(SV("%b='juin'\t%B='juin'\t%h='juin'\t%m='06'\t%Om='06'\t%d='30'\t%e='30'\t%Od='30'\t%Oe='30'\n"),
        lfmt,
        std::chrono::year_month_day_last{std::chrono::year{1970}, std::chrono::month_day_last{std::chrono::June}});
  check(SV("%b='juil.'\t%B='juillet'\t%h='juil.'\t%m='07'\t%Om='07'\t%d='31'\t%e='31'\t%Od='31'\t%Oe='31'\n"),
        lfmt,
        std::chrono::year_month_day_last{std::chrono::year{1970}, std::chrono::month_day_last{std::chrono::July}});
  check(SV("%b='août'\t%B='août'\t%h='août'\t%m='08'\t%Om='08'\t%d='31'\t%e='31'\t%Od='31'\t%Oe='31'\n"),
        lfmt,
        std::chrono::year_month_day_last{std::chrono::year{1970}, std::chrono::month_day_last{std::chrono::August}});
  check(SV("%b='sept.'\t%B='septembre'\t%h='sept.'\t%m='09'\t%Om='09'\t%d='30'\t%e='30'\t%Od='30'\t%Oe='30'\n"),
        lfmt,
        std::chrono::year_month_day_last{std::chrono::year{1970}, std::chrono::month_day_last{std::chrono::September}});
  check(SV("%b='oct.'\t%B='octobre'\t%h='oct.'\t%m='10'\t%Om='10'\t%d='31'\t%e='31'\t%Od='31'\t%Oe='31'\n"),
        lfmt,
        std::chrono::year_month_day_last{std::chrono::year{1970}, std::chrono::month_day_last{std::chrono::October}});
  check(SV("%b='nov.'\t%B='novembre'\t%h='nov.'\t%m='11'\t%Om='11'\t%d='30'\t%e='30'\t%Od='30'\t%Oe='30'\n"),
        lfmt,
        std::chrono::year_month_day_last{std::chrono::year{1970}, std::chrono::month_day_last{std::chrono::November}});
  check(SV("%b='déc.'\t%B='décembre'\t%h='déc.'\t%m='12'\t%Om='12'\t%d='31'\t%e='31'\t%Od='31'\t%Oe='31'\n"),
        lfmt,
        std::chrono::year_month_day_last{std::chrono::year{1970}, std::chrono::month_day_last{std::chrono::December}});
#endif   // defined(__APPLE__)

  // Use supplied locale (ja_JP)
#if defined(_WIN32)
  check(loc,
        SV("%b='1'\t%B='1月'\t%h='1'\t%m='01'\t%Om='01'\t%d='31'\t%e='31'\t%Od='31'\t%Oe='31'\n"),
        lfmt,
        std::chrono::year_month_day_last{std::chrono::year{1970}, std::chrono::month_day_last{std::chrono::January}});
  check(loc,
        SV("%b='2'\t%B='2月'\t%h='2'\t%m='02'\t%Om='02'\t%d='28'\t%e='28'\t%Od='28'\t%Oe='28'\n"),
        lfmt,
        std::chrono::year_month_day_last{std::chrono::year{1970}, std::chrono::month_day_last{std::chrono::February}});
  check(loc,
        SV("%b='3'\t%B='3月'\t%h='3'\t%m='03'\t%Om='03'\t%d='31'\t%e='31'\t%Od='31'\t%Oe='31'\n"),
        lfmt,
        std::chrono::year_month_day_last{std::chrono::year{1970}, std::chrono::month_day_last{std::chrono::March}});
  check(loc,
        SV("%b='4'\t%B='4月'\t%h='4'\t%m='04'\t%Om='04'\t%d='30'\t%e='30'\t%Od='30'\t%Oe='30'\n"),
        lfmt,
        std::chrono::year_month_day_last{std::chrono::year{1970}, std::chrono::month_day_last{std::chrono::April}});
  check(loc,
        SV("%b='5'\t%B='5月'\t%h='5'\t%m='05'\t%Om='05'\t%d='31'\t%e='31'\t%Od='31'\t%Oe='31'\n"),
        lfmt,
        std::chrono::year_month_day_last{std::chrono::year{1970}, std::chrono::month_day_last{std::chrono::May}});
  check(loc,
        SV("%b='6'\t%B='6月'\t%h='6'\t%m='06'\t%Om='06'\t%d='30'\t%e='30'\t%Od='30'\t%Oe='30'\n"),
        lfmt,
        std::chrono::year_month_day_last{std::chrono::year{1970}, std::chrono::month_day_last{std::chrono::June}});
  check(loc,
        SV("%b='7'\t%B='7月'\t%h='7'\t%m='07'\t%Om='07'\t%d='31'\t%e='31'\t%Od='31'\t%Oe='31'\n"),
        lfmt,
        std::chrono::year_month_day_last{std::chrono::year{1970}, std::chrono::month_day_last{std::chrono::July}});
  check(loc,
        SV("%b='8'\t%B='8月'\t%h='8'\t%m='08'\t%Om='08'\t%d='31'\t%e='31'\t%Od='31'\t%Oe='31'\n"),
        lfmt,
        std::chrono::year_month_day_last{std::chrono::year{1970}, std::chrono::month_day_last{std::chrono::August}});
  check(loc,
        SV("%b='9'\t%B='9月'\t%h='9'\t%m='09'\t%Om='09'\t%d='30'\t%e='30'\t%Od='30'\t%Oe='30'\n"),
        lfmt,
        std::chrono::year_month_day_last{std::chrono::year{1970}, std::chrono::month_day_last{std::chrono::September}});
  check(loc,
        SV("%b='10'\t%B='10月'\t%h='10'\t%m='10'\t%Om='10'\t%d='31'\t%e='31'\t%Od='31'\t%Oe='31'\n"),
        lfmt,
        std::chrono::year_month_day_last{std::chrono::year{1970}, std::chrono::month_day_last{std::chrono::October}});
  check(loc,
        SV("%b='11'\t%B='11月'\t%h='11'\t%m='11'\t%Om='11'\t%d='30'\t%e='30'\t%Od='30'\t%Oe='30'\n"),
        lfmt,
        std::chrono::year_month_day_last{std::chrono::year{1970}, std::chrono::month_day_last{std::chrono::November}});
  check(loc,
        SV("%b='12'\t%B='12月'\t%h='12'\t%m='12'\t%Om='12'\t%d='31'\t%e='31'\t%Od='31'\t%Oe='31'\n"),
        lfmt,
        std::chrono::year_month_day_last{std::chrono::year{1970}, std::chrono::month_day_last{std::chrono::December}});
#elif defined(_AIX)      // defined(_WIN32)
  check(loc,
        SV("%b='1月'\t%B='1月'\t%h='1月'\t%m='01'\t%Om='01'\t%d='31'\t%e='31'\t%Od='31'\t%Oe='31'\n"),
        lfmt,
        std::chrono::year_month_day_last{std::chrono::year{1970}, std::chrono::month_day_last{std::chrono::January}});
  check(loc,
        SV("%b='2月'\t%B='2月'\t%h='2月'\t%m='02'\t%Om='02'\t%d='28'\t%e='28'\t%Od='28'\t%Oe='28'\n"),
        lfmt,
        std::chrono::year_month_day_last{std::chrono::year{1970}, std::chrono::month_day_last{std::chrono::February}});
  check(loc,
        SV("%b='3月'\t%B='3月'\t%h='3月'\t%m='03'\t%Om='03'\t%d='31'\t%e='31'\t%Od='31'\t%Oe='31'\n"),
        lfmt,
        std::chrono::year_month_day_last{std::chrono::year{1970}, std::chrono::month_day_last{std::chrono::March}});
  check(loc,
        SV("%b='4月'\t%B='4月'\t%h='4月'\t%m='04'\t%Om='04'\t%d='30'\t%e='30'\t%Od='30'\t%Oe='30'\n"),
        lfmt,
        std::chrono::year_month_day_last{std::chrono::year{1970}, std::chrono::month_day_last{std::chrono::April}});
  check(loc,
        SV("%b='5月'\t%B='5月'\t%h='5月'\t%m='05'\t%Om='05'\t%d='31'\t%e='31'\t%Od='31'\t%Oe='31'\n"),
        lfmt,
        std::chrono::year_month_day_last{std::chrono::year{1970}, std::chrono::month_day_last{std::chrono::May}});
  check(loc,
        SV("%b='6月'\t%B='6月'\t%h='6月'\t%m='06'\t%Om='06'\t%d='30'\t%e='30'\t%Od='30'\t%Oe='30'\n"),
        lfmt,
        std::chrono::year_month_day_last{std::chrono::year{1970}, std::chrono::month_day_last{std::chrono::June}});
  check(loc,
        SV("%b='7月'\t%B='7月'\t%h='7月'\t%m='07'\t%Om='07'\t%d='31'\t%e='31'\t%Od='31'\t%Oe='31'\n"),
        lfmt,
        std::chrono::year_month_day_last{std::chrono::year{1970}, std::chrono::month_day_last{std::chrono::July}});
  check(loc,
        SV("%b='8月'\t%B='8月'\t%h='8月'\t%m='08'\t%Om='08'\t%d='31'\t%e='31'\t%Od='31'\t%Oe='31'\n"),
        lfmt,
        std::chrono::year_month_day_last{std::chrono::year{1970}, std::chrono::month_day_last{std::chrono::August}});
  check(loc,
        SV("%b='9月'\t%B='9月'\t%h='9月'\t%m='09'\t%Om='09'\t%d='30'\t%e='30'\t%Od='30'\t%Oe='30'\n"),
        lfmt,
        std::chrono::year_month_day_last{std::chrono::year{1970}, std::chrono::month_day_last{std::chrono::September}});
  check(loc,
        SV("%b='10月'\t%B='10月'\t%h='10月'\t%m='10'\t%Om='10'\t%d='31'\t%e='31'\t%Od='31'\t%Oe='31'\n"),
        lfmt,
        std::chrono::year_month_day_last{std::chrono::year{1970}, std::chrono::month_day_last{std::chrono::October}});
  check(loc,
        SV("%b='11月'\t%B='11月'\t%h='11月'\t%m='11'\t%Om='11'\t%d='30'\t%e='30'\t%Od='30'\t%Oe='30'\n"),
        lfmt,
        std::chrono::year_month_day_last{std::chrono::year{1970}, std::chrono::month_day_last{std::chrono::November}});
  check(loc,
        SV("%b='12月'\t%B='12月'\t%h='12月'\t%m='12'\t%Om='12'\t%d='31'\t%e='31'\t%Od='31'\t%Oe='31'\n"),
        lfmt,
        std::chrono::year_month_day_last{std::chrono::year{1970}, std::chrono::month_day_last{std::chrono::December}});
#elif defined(__APPLE__) // defined(_WIN32)
  check(loc,
        SV("%b=' 1'\t%B='1月'\t%h=' 1'\t%m='01'\t%Om='01'\t%d='31'\t%e='31'\t%Od='31'\t%Oe='31'\n"),
        lfmt,
        std::chrono::year_month_day_last{std::chrono::year{1970}, std::chrono::month_day_last{std::chrono::January}});
  check(loc,
        SV("%b=' 2'\t%B='2月'\t%h=' 2'\t%m='02'\t%Om='02'\t%d='28'\t%e='28'\t%Od='28'\t%Oe='28'\n"),
        lfmt,
        std::chrono::year_month_day_last{std::chrono::year{1970}, std::chrono::month_day_last{std::chrono::February}});
  check(loc,
        SV("%b=' 3'\t%B='3月'\t%h=' 3'\t%m='03'\t%Om='03'\t%d='31'\t%e='31'\t%Od='31'\t%Oe='31'\n"),
        lfmt,
        std::chrono::year_month_day_last{std::chrono::year{1970}, std::chrono::month_day_last{std::chrono::March}});
  check(loc,
        SV("%b=' 4'\t%B='4月'\t%h=' 4'\t%m='04'\t%Om='04'\t%d='30'\t%e='30'\t%Od='30'\t%Oe='30'\n"),
        lfmt,
        std::chrono::year_month_day_last{std::chrono::year{1970}, std::chrono::month_day_last{std::chrono::April}});
  check(loc,
        SV("%b=' 5'\t%B='5月'\t%h=' 5'\t%m='05'\t%Om='05'\t%d='31'\t%e='31'\t%Od='31'\t%Oe='31'\n"),
        lfmt,
        std::chrono::year_month_day_last{std::chrono::year{1970}, std::chrono::month_day_last{std::chrono::May}});
  check(loc,
        SV("%b=' 6'\t%B='6月'\t%h=' 6'\t%m='06'\t%Om='06'\t%d='30'\t%e='30'\t%Od='30'\t%Oe='30'\n"),
        lfmt,
        std::chrono::year_month_day_last{std::chrono::year{1970}, std::chrono::month_day_last{std::chrono::June}});
  check(loc,
        SV("%b=' 7'\t%B='7月'\t%h=' 7'\t%m='07'\t%Om='07'\t%d='31'\t%e='31'\t%Od='31'\t%Oe='31'\n"),
        lfmt,
        std::chrono::year_month_day_last{std::chrono::year{1970}, std::chrono::month_day_last{std::chrono::July}});
  check(loc,
        SV("%b=' 8'\t%B='8月'\t%h=' 8'\t%m='08'\t%Om='08'\t%d='31'\t%e='31'\t%Od='31'\t%Oe='31'\n"),
        lfmt,
        std::chrono::year_month_day_last{std::chrono::year{1970}, std::chrono::month_day_last{std::chrono::August}});
  check(loc,
        SV("%b=' 9'\t%B='9月'\t%h=' 9'\t%m='09'\t%Om='09'\t%d='30'\t%e='30'\t%Od='30'\t%Oe='30'\n"),
        lfmt,
        std::chrono::year_month_day_last{std::chrono::year{1970}, std::chrono::month_day_last{std::chrono::September}});
  check(loc,
        SV("%b='10'\t%B='10月'\t%h='10'\t%m='10'\t%Om='10'\t%d='31'\t%e='31'\t%Od='31'\t%Oe='31'\n"),
        lfmt,
        std::chrono::year_month_day_last{std::chrono::year{1970}, std::chrono::month_day_last{std::chrono::October}});
  check(loc,
        SV("%b='11'\t%B='11月'\t%h='11'\t%m='11'\t%Om='11'\t%d='30'\t%e='30'\t%Od='30'\t%Oe='30'\n"),
        lfmt,
        std::chrono::year_month_day_last{std::chrono::year{1970}, std::chrono::month_day_last{std::chrono::November}});
  check(loc,
        SV("%b='12'\t%B='12月'\t%h='12'\t%m='12'\t%Om='12'\t%d='31'\t%e='31'\t%Od='31'\t%Oe='31'\n"),
        lfmt,
        std::chrono::year_month_day_last{std::chrono::year{1970}, std::chrono::month_day_last{std::chrono::December}});
#else                    // defined(_WIN32)
  check(loc,
        SV("%b=' 1月'\t%B='1月'\t%h=' 1月'\t%m='01'\t%Om='一'\t%d='31'\t%e='31'\t%Od='三十一'\t%Oe='三十一'\n"),
        lfmt,
        std::chrono::year_month_day_last{std::chrono::year{1970}, std::chrono::month_day_last{std::chrono::January}});
  check(loc,
        SV("%b=' 2月'\t%B='2月'\t%h=' 2月'\t%m='02'\t%Om='二'\t%d='28'\t%e='28'\t%Od='二十八'\t%Oe='二十八'\n"),
        lfmt,
        std::chrono::year_month_day_last{std::chrono::year{1970}, std::chrono::month_day_last{std::chrono::February}});
  check(loc,
        SV("%b=' 3月'\t%B='3月'\t%h=' 3月'\t%m='03'\t%Om='三'\t%d='31'\t%e='31'\t%Od='三十一'\t%Oe='三十一'\n"),
        lfmt,
        std::chrono::year_month_day_last{std::chrono::year{1970}, std::chrono::month_day_last{std::chrono::March}});
  check(loc,
        SV("%b=' 4月'\t%B='4月'\t%h=' 4月'\t%m='04'\t%Om='四'\t%d='30'\t%e='30'\t%Od='三十'\t%Oe='三十'\n"),
        lfmt,
        std::chrono::year_month_day_last{std::chrono::year{1970}, std::chrono::month_day_last{std::chrono::April}});
  check(loc,
        SV("%b=' 5月'\t%B='5月'\t%h=' 5月'\t%m='05'\t%Om='五'\t%d='31'\t%e='31'\t%Od='三十一'\t%Oe='三十一'\n"),
        lfmt,
        std::chrono::year_month_day_last{std::chrono::year{1970}, std::chrono::month_day_last{std::chrono::May}});
  check(loc,
        SV("%b=' 6月'\t%B='6月'\t%h=' 6月'\t%m='06'\t%Om='六'\t%d='30'\t%e='30'\t%Od='三十'\t%Oe='三十'\n"),
        lfmt,
        std::chrono::year_month_day_last{std::chrono::year{1970}, std::chrono::month_day_last{std::chrono::June}});
  check(loc,
        SV("%b=' 7月'\t%B='7月'\t%h=' 7月'\t%m='07'\t%Om='七'\t%d='31'\t%e='31'\t%Od='三十一'\t%Oe='三十一'\n"),
        lfmt,
        std::chrono::year_month_day_last{std::chrono::year{1970}, std::chrono::month_day_last{std::chrono::July}});
  check(loc,
        SV("%b=' 8月'\t%B='8月'\t%h=' 8月'\t%m='08'\t%Om='八'\t%d='31'\t%e='31'\t%Od='三十一'\t%Oe='三十一'\n"),
        lfmt,
        std::chrono::year_month_day_last{std::chrono::year{1970}, std::chrono::month_day_last{std::chrono::August}});
  check(loc,
        SV("%b=' 9月'\t%B='9月'\t%h=' 9月'\t%m='09'\t%Om='九'\t%d='30'\t%e='30'\t%Od='三十'\t%Oe='三十'\n"),
        lfmt,
        std::chrono::year_month_day_last{std::chrono::year{1970}, std::chrono::month_day_last{std::chrono::September}});
  check(loc,
        SV("%b='10月'\t%B='10月'\t%h='10月'\t%m='10'\t%Om='十'\t%d='31'\t%e='31'\t%Od='三十一'\t%Oe='三十一'\n"),
        lfmt,
        std::chrono::year_month_day_last{std::chrono::year{1970}, std::chrono::month_day_last{std::chrono::October}});
  check(loc,
        SV("%b='11月'\t%B='11月'\t%h='11月'\t%m='11'\t%Om='十一'\t%d='30'\t%e='30'\t%Od='三十'\t%Oe='三十'\n"),
        lfmt,
        std::chrono::year_month_day_last{std::chrono::year{1970}, std::chrono::month_day_last{std::chrono::November}});
  check(loc,
        SV("%b='12月'\t%B='12月'\t%h='12月'\t%m='12'\t%Om='十二'\t%d='31'\t%e='31'\t%Od='三十一'\t%Oe='三十一'\n"),
        lfmt,
        std::chrono::year_month_day_last{std::chrono::year{1970}, std::chrono::month_day_last{std::chrono::December}});
#endif                   //  defined(_WIN32)

  std::locale::global(std::locale::classic());
}

template <class CharT>
static void test_valid_ymd_values() {
  constexpr std::basic_string_view<CharT> fmt = SV(
      "{:"
      "%%C='%C'%t"
      "%%D='%D'%t"
      "%%F='%F'%t"
      "%%j='%j'%t"
      "%%g='%g'%t"
      "%%G='%G'%t"
      "%%u='%u'%t"
      "%%U='%U'%t"
      "%%V='%V'%t"
      "%%w='%w'%t"
      "%%W='%W'%t"
      "%%x='%x'%t"
      "%%y='%y'%t"
      "%%Y='%Y'%t"
      "%%Ex='%Ex'%t"
      "%%EC='%EC'%t"
      "%%Ey='%Ey'%t"
      "%%EY='%EY'%t"
      "%%Ou='%Ou'%t"
      "%%OU='%OU'%t"
      "%%OV='%OV'%t"
      "%%Ow='%Ow'%t"
      "%%OW='%OW'%t"
      "%%Oy='%Oy'%t"
      "%n}");

  constexpr std::basic_string_view<CharT> lfmt = SV(
      "{:L"
      "%%C='%C'%t"
      "%%D='%D'%t"
      "%%F='%F'%t"
      "%%j='%j'%t"
      "%%g='%g'%t"
      "%%G='%G'%t"
      "%%u='%u'%t"
      "%%U='%U'%t"
      "%%V='%V'%t"
      "%%w='%w'%t"
      "%%W='%W'%t"
      "%%x='%x'%t"
      "%%y='%y'%t"
      "%%Y='%Y'%t"
      "%%Ex='%Ex'%t"
      "%%EC='%EC'%t"
      "%%Ey='%Ey'%t"
      "%%EY='%EY'%t"
      "%%Ou='%Ou'%t"
      "%%OU='%OU'%t"
      "%%OV='%OV'%t"
      "%%Ow='%Ow'%t"
      "%%OW='%OW'%t"
      "%%Oy='%Oy'%t"
      "%n}");

  const std::locale loc(LOCALE_ja_JP_UTF_8);
  std::locale::global(std::locale(LOCALE_fr_FR_UTF_8));

  // Non localized output using C-locale
  check(
      SV("%C='19'\t"
         "%D='01/31/70'\t"
         "%F='1970-01-31'\t"
         "%j='031'\t"
         "%g='70'\t"
         "%G='1970'\t"
         "%u='6'\t"
         "%U='04'\t"
         "%V='05'\t"
         "%w='6'\t"
         "%W='04'\t"
         "%x='01/31/70'\t"
         "%y='70'\t"
         "%Y='1970'\t"
         "%Ex='01/31/70'\t"
         "%EC='19'\t"
         "%Ey='70'\t"
         "%EY='1970'\t"
         "%Ou='6'\t"
         "%OU='04'\t"
         "%OV='05'\t"
         "%Ow='6'\t"
         "%OW='04'\t"
         "%Oy='70'\t"
         "\n"),
      fmt,
      std::chrono::year_month_day_last{std::chrono::year{1970}, std::chrono::month_day_last{std::chrono::January}});

  check(
      SV("%C='20'\t"
         "%D='05/31/04'\t"
         "%F='2004-05-31'\t"
         "%j='152'\t"
         "%g='04'\t"
         "%G='2004'\t"
         "%u='1'\t"
         "%U='22'\t"
         "%V='23'\t"
         "%w='1'\t"
         "%W='22'\t"
         "%x='05/31/04'\t"
         "%y='04'\t"
         "%Y='2004'\t"
         "%Ex='05/31/04'\t"
         "%EC='20'\t"
         "%Ey='04'\t"
         "%EY='2004'\t"
         "%Ou='1'\t"
         "%OU='22'\t"
         "%OV='23'\t"
         "%Ow='1'\t"
         "%OW='22'\t"
         "%Oy='04'\t"
         "\n"),
      fmt,
      std::chrono::year_month_day_last{std::chrono::year{2004}, std::chrono::month_day_last{std::chrono::May}});

  // Use the global locale (fr_FR)
  check(
      SV("%C='19'\t"
         "%D='01/31/70'\t"
         "%F='1970-01-31'\t"
         "%j='031'\t"
         "%g='70'\t"
         "%G='1970'\t"
         "%u='6'\t"
         "%U='04'\t"
         "%V='05'\t"
         "%w='6'\t"
         "%W='04'\t"
#if defined(__APPLE__)
         "%x='31.01.1970'\t"
#else
         "%x='31/01/1970'\t"
#endif
         "%y='70'\t"
         "%Y='1970'\t"
#if defined(__APPLE__)
         "%Ex='31.01.1970'\t"
#else
         "%Ex='31/01/1970'\t"
#endif
         "%EC='19'\t"
         "%Ey='70'\t"
         "%EY='1970'\t"
         "%Ou='6'\t"
         "%OU='04'\t"
         "%OV='05'\t"
         "%Ow='6'\t"
         "%OW='04'\t"
         "%Oy='70'\t"
         "\n"),
      lfmt,
      std::chrono::year_month_day_last{std::chrono::year{1970}, std::chrono::month_day_last{std::chrono::January}});

  check(
      SV("%C='20'\t"
         "%D='05/31/04'\t"
         "%F='2004-05-31'\t"
         "%j='152'\t"
         "%g='04'\t"
         "%G='2004'\t"
         "%u='1'\t"
         "%U='22'\t"
         "%V='23'\t"
         "%w='1'\t"
         "%W='22'\t"
#if defined(__APPLE__)
         "%x='31.05.2004'\t"
#else
         "%x='31/05/2004'\t"
#endif
         "%y='04'\t"
         "%Y='2004'\t"
#if defined(__APPLE__)
         "%Ex='31.05.2004'\t"
#else
         "%Ex='31/05/2004'\t"
#endif
         "%EC='20'\t"
         "%Ey='04'\t"
         "%EY='2004'\t"
         "%Ou='1'\t"
         "%OU='22'\t"
         "%OV='23'\t"
         "%Ow='1'\t"
         "%OW='22'\t"
         "%Oy='04'\t"
         "\n"),
      lfmt,
      std::chrono::year_month_day_last{std::chrono::year{2004}, std::chrono::month_day_last{std::chrono::May}});

  // Use supplied locale (ja_JP)
  check(
      loc,
      SV("%C='19'\t"
         "%D='01/31/70'\t"
         "%F='1970-01-31'\t"
         "%j='031'\t"
         "%g='70'\t"
         "%G='1970'\t"
         "%u='6'\t"
         "%U='04'\t"
         "%V='05'\t"
         "%w='6'\t"
         "%W='04'\t"
#if defined(__APPLE__) || defined(_AIX) || defined(_WIN32)
         "%x='1970/01/31'\t"
#else  // defined(__APPLE__) || defined(_AIX) || defined(_WIN32)
         "%x='1970年01月31日'\t"
#endif // defined(__APPLE__) || defined(_AIX) || defined(_WIN32)
         "%y='70'\t"
         "%Y='1970'\t"
#if defined(__APPLE__) || defined(_AIX) || defined(_WIN32)
         "%Ex='1970/01/31'\t"
         "%EC='19'\t"
         "%Ey='70'\t"
         "%EY='1970'\t"
         "%Ou='6'\t"
         "%OU='04'\t"
         "%OV='05'\t"
         "%Ow='6'\t"
         "%OW='04'\t"
         "%Oy='70'\t"
#else  // defined(__APPLE__) || defined(_AIX) || defined(_WIN32)
         "%Ex='昭和45年01月31日'\t"
         "%EC='昭和'\t"
         "%Ey='45'\t"
         "%EY='昭和45年'\t"
         "%Ou='六'\t"
         "%OU='四'\t"
         "%OV='五'\t"
         "%Ow='六'\t"
         "%OW='四'\t"
         "%Oy='七十'\t"
#endif // defined(__APPLE__) || defined(_AIX) || defined(_WIN32)
         "\n"),
      lfmt,
      std::chrono::year_month_day_last{std::chrono::year{1970}, std::chrono::month_day_last{std::chrono::January}});

  check(
      loc,
      SV("%C='20'\t"
         "%D='05/31/04'\t"
         "%F='2004-05-31'\t"
         "%j='152'\t"
         "%g='04'\t"
         "%G='2004'\t"
         "%u='1'\t"
         "%U='22'\t"
         "%V='23'\t"
         "%w='1'\t"
         "%W='22'\t"
#if defined(__APPLE__) || defined(_AIX) || defined(_WIN32)
         "%x='2004/05/31'\t"
#else  // defined(__APPLE__) || defined(_AIX) || defined(_WIN32)
         "%x='2004年05月31日'\t"
#endif // defined(__APPLE__) || defined(_AIX) || defined(_WIN32)
         "%y='04'\t"
         "%Y='2004'\t"
#if defined(__APPLE__) || defined(_AIX) || defined(_WIN32)
         "%Ex='2004/05/31'\t"
         "%EC='20'\t"
         "%Ey='04'\t"
         "%EY='2004'\t"
         "%Ou='1'\t"
         "%OU='22'\t"
         "%OV='23'\t"
         "%Ow='1'\t"
         "%OW='22'\t"
         "%Oy='04'\t"
#else  // defined(__APPLE__) || defined(_AIX) || defined(_WIN32)
         "%Ex='平成16年05月31日'\t"
         "%EC='平成'\t"
         "%Ey='16'\t"
         "%EY='平成16年'\t"
         "%Ou='一'\t"
         "%OU='二十二'\t"
         "%OV='二十三'\t"
         "%Ow='一'\t"
         "%OW='二十二'\t"
         "%Oy='四'\t"
#endif // defined(__APPLE__) || defined(_AIX) || defined(_WIN32)
         "\n"),
      lfmt,
      std::chrono::year_month_day_last{std::chrono::year{2004}, std::chrono::month_day_last{std::chrono::May}});

  std::locale::global(std::locale::classic());
}

template <class CharT>
static void test_valid_values() {
  // Fields only using month and day.
  test_valid_md_values<CharT>();
  // Fields only using year, month, and day.
  test_valid_ymd_values<CharT>();
}

template <class CharT>
static void test() {
  test_no_chrono_specs<CharT>();
  test_invalid_values<CharT>();
  test_valid_values<CharT>();
  check_invalid_types<CharT>(
      {SV("a"),  SV("A"),  SV("b"),  SV("B"),  SV("C"),  SV("d"),  SV("D"),  SV("e"),  SV("EC"),
       SV("Ex"), SV("Ey"), SV("EY"), SV("F"),  SV("g"),  SV("G"),  SV("h"),  SV("j"),  SV("m"),
       SV("Od"), SV("Oe"), SV("Om"), SV("Ou"), SV("OU"), SV("OV"), SV("Ow"), SV("OW"), SV("Oy"),
       SV("u"),  SV("U"),  SV("V"),  SV("w"),  SV("W"),  SV("x"),  SV("y"),  SV("Y")},
      std::chrono::year_month_day_last{std::chrono::year{1970}, std::chrono::month_day_last{std::chrono::January}});

  check_exception(
      "Expected '%' or '}' in the chrono format-string",
      SV("{:A"),
      std::chrono::year_month_day_last{std::chrono::year{1970}, std::chrono::month_day_last{std::chrono::January}});
  check_exception(
      "The chrono-specs contains a '{'",
      SV("{:%%{"),
      std::chrono::year_month_day_last{std::chrono::year{1970}, std::chrono::month_day_last{std::chrono::January}});
  check_exception(
      "End of input while parsing the modifier chrono conversion-spec",
      SV("{:%"),
      std::chrono::year_month_day_last{std::chrono::year{1970}, std::chrono::month_day_last{std::chrono::January}});
  check_exception(
      "End of input while parsing the modifier E",
      SV("{:%E"),
      std::chrono::year_month_day_last{std::chrono::year{1970}, std::chrono::month_day_last{std::chrono::January}});
  check_exception(
      "End of input while parsing the modifier O",
      SV("{:%O"),
      std::chrono::year_month_day_last{std::chrono::year{1970}, std::chrono::month_day_last{std::chrono::January}});

  // Precision not allowed
  check_exception(
      "Expected '%' or '}' in the chrono format-string",
      SV("{:.3}"),
      std::chrono::year_month_day_last{std::chrono::year{1970}, std::chrono::month_day_last{std::chrono::January}});
}

int main(int, char**) {
  test<char>();

#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  test<wchar_t>();
#endif

  return 0;
}
