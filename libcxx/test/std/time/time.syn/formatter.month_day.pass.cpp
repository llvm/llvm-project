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

// TODO FMT Investigate Windows issues.
// UNSUPPORTED: msvc, target={{.+}}-windows-gnu

// TODO FMT It seems GCC uses too much memory in the CI and fails.
// UNSUPPORTED: gcc-12

// REQUIRES: locale.fr_FR.UTF-8
// REQUIRES: locale.ja_JP.UTF-8

// <chrono>

// template<class charT> struct formatter<chrono::month_day, charT>;

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
  // Valid month "valid" day
  check(SV("Feb/31"), SV("{}"), std::chrono::month_day{std::chrono::month{2}, std::chrono::day{31}});
  check(SV("*Feb/31*"), SV("{:*^8}"), std::chrono::month_day{std::chrono::month{2}, std::chrono::day{31}});
  check(SV("*Feb/31"), SV("{:*>7}"), std::chrono::month_day{std::chrono::month{2}, std::chrono::day{31}});

  // Invalid month "valid" day
  check(SV("0 is not a valid month/31"), SV("{}"), std::chrono::month_day{std::chrono::month{0}, std::chrono::day{31}});
  check(SV("*0 is not a valid month/31*"),
        SV("{:*^27}"),
        std::chrono::month_day{std::chrono::month{0}, std::chrono::day{31}});

  // Valid month invalid day
  check(SV("Feb/32 is not a valid day"), SV("{}"), std::chrono::month_day{std::chrono::month{2}, std::chrono::day{32}});
  check(SV("*Feb/32 is not a valid day*"),
        SV("{:*^27}"),
        std::chrono::month_day{std::chrono::month{2}, std::chrono::day{32}});
  check(SV("*Feb/32 is not a valid day"),
        SV("{:*>26}"),
        std::chrono::month_day{std::chrono::month{2}, std::chrono::day{32}});

  // Invalid month invalid day
  check(SV("0 is not a valid month/32 is not a valid day"),
        SV("{}"),
        std::chrono::month_day{std::chrono::month{0}, std::chrono::day{32}});
  check(SV("*0 is not a valid month/32 is not a valid day*"),
        SV("{:*^46}"),
        std::chrono::month_day{std::chrono::month{0}, std::chrono::day{32}});
}

template <class CharT>
static void test_valid_values() {
  // Test that %b, %h, and %B throw an exception.
  check_exception("formatting a month name from an invalid month number",
                  SV("{:%b}"),
                  std::chrono::month_day{std::chrono::month{200}, std::chrono::day{31}});
  check_exception("formatting a month name from an invalid month number",
                  SV("{:%b}"),
                  std::chrono::month_day{std::chrono::month{13}, std::chrono::day{31}});
  check_exception("formatting a month name from an invalid month number",
                  SV("{:%b}"),
                  std::chrono::month_day{std::chrono::month{255}, std::chrono::day{31}});

  check_exception("formatting a month name from an invalid month number",
                  SV("{:%h}"),
                  std::chrono::month_day{std::chrono::month{0}, std::chrono::day{31}});
  check_exception("formatting a month name from an invalid month number",
                  SV("{:%h}"),
                  std::chrono::month_day{std::chrono::month{13}, std::chrono::day{31}});
  check_exception("formatting a month name from an invalid month number",
                  SV("{:%h}"),
                  std::chrono::month_day{std::chrono::month{255}, std::chrono::day{31}});

  check_exception("formatting a month name from an invalid month number",
                  SV("{:%B}"),
                  std::chrono::month_day{std::chrono::month{0}, std::chrono::day{31}});
  check_exception("formatting a month name from an invalid month number",
                  SV("{:%B}"),
                  std::chrono::month_day{std::chrono::month{13}, std::chrono::day{31}});
  check_exception("formatting a month name from an invalid month number",
                  SV("{:%B}"),
                  std::chrono::month_day{std::chrono::month{255}, std::chrono::day{31}});

  constexpr std::basic_string_view<CharT> fmt =
      SV("{:%%b='%b'%t%%B='%B'%t%%h='%h'%t%%m='%m'%t%%Om='%Om'%t%%d='%d'%t%%e='%e'%t%%Od='%Od'%t%%Oe='%Oe'%n}");
  constexpr std::basic_string_view<CharT> lfmt =
      SV("{:L%%b='%b'%t%%B='%B'%t%%h='%h'%t%%m='%m'%t%%Om='%Om'%t%%d='%d'%t%%e='%e'%t%%Od='%Od'%t%%Oe='%Oe'%n}");

  const std::locale loc(LOCALE_ja_JP_UTF_8);
  std::locale::global(std::locale(LOCALE_fr_FR_UTF_8));

  // Non localized output using C-locale
  check(SV("%b='Jan'\t%B='January'\t%h='Jan'\t%m='01'\t%Om='01'\t%d='00'\t%e=' 0'\t%Od='00'\t%Oe=' 0'\n"),
        fmt,
        std::chrono::month_day{std::chrono::January, std::chrono::day{0}});
  check(SV("%b='Feb'\t%B='February'\t%h='Feb'\t%m='02'\t%Om='02'\t%d='01'\t%e=' 1'\t%Od='01'\t%Oe=' 1'\n"),
        fmt,
        std::chrono::month_day{std::chrono::February, std::chrono::day{1}});
  check(SV("%b='Mar'\t%B='March'\t%h='Mar'\t%m='03'\t%Om='03'\t%d='09'\t%e=' 9'\t%Od='09'\t%Oe=' 9'\n"),
        fmt,
        std::chrono::month_day{std::chrono::March, std::chrono::day{9}});
  check(SV("%b='Apr'\t%B='April'\t%h='Apr'\t%m='04'\t%Om='04'\t%d='10'\t%e='10'\t%Od='10'\t%Oe='10'\n"),
        fmt,
        std::chrono::month_day{std::chrono::April, std::chrono::day{10}});
  check(SV("%b='May'\t%B='May'\t%h='May'\t%m='05'\t%Om='05'\t%d='28'\t%e='28'\t%Od='28'\t%Oe='28'\n"),
        fmt,
        std::chrono::month_day{std::chrono::May, std::chrono::day{28}});
  check(SV("%b='Jun'\t%B='June'\t%h='Jun'\t%m='06'\t%Om='06'\t%d='29'\t%e='29'\t%Od='29'\t%Oe='29'\n"),
        fmt,
        std::chrono::month_day{std::chrono::June, std::chrono::day{29}});
  check(SV("%b='Jul'\t%B='July'\t%h='Jul'\t%m='07'\t%Om='07'\t%d='30'\t%e='30'\t%Od='30'\t%Oe='30'\n"),
        fmt,
        std::chrono::month_day{std::chrono::July, std::chrono::day{30}});
  check(SV("%b='Aug'\t%B='August'\t%h='Aug'\t%m='08'\t%Om='08'\t%d='31'\t%e='31'\t%Od='31'\t%Oe='31'\n"),
        fmt,
        std::chrono::month_day{std::chrono::August, std::chrono::day{31}});
  check(SV("%b='Sep'\t%B='September'\t%h='Sep'\t%m='09'\t%Om='09'\t%d='32'\t%e='32'\t%Od='32'\t%Oe='32'\n"),
        fmt,
        std::chrono::month_day{std::chrono::September, std::chrono::day{32}});
  check(SV("%b='Oct'\t%B='October'\t%h='Oct'\t%m='10'\t%Om='10'\t%d='99'\t%e='99'\t%Od='99'\t%Oe='99'\n"),
        fmt,
        std::chrono::month_day{std::chrono::October, std::chrono::day{99}});
#if defined(_AIX)
  check(SV("%b='Nov'\t%B='November'\t%h='Nov'\t%m='11'\t%Om='11'\t%d='00'\t%e=' 0'\t%Od='00'\t%Oe=' 0'\n"),
        fmt,
        std::chrono::month_day{std::chrono::November, std::chrono::day{100}});
  check(SV("%b='Dec'\t%B='December'\t%h='Dec'\t%m='12'\t%Om='12'\t%d='55'\t%e='55'\t%Od='55'\t%Oe='55'\n"),
        fmt,
        std::chrono::month_day{std::chrono::December, std::chrono::day{255}});
#else  //  defined(_AIX)
  check(SV("%b='Nov'\t%B='November'\t%h='Nov'\t%m='11'\t%Om='11'\t%d='100'\t%e='100'\t%Od='100'\t%Oe='100'\n"),
        fmt,
        std::chrono::month_day{std::chrono::November, std::chrono::day{100}});
  check(SV("%b='Dec'\t%B='December'\t%h='Dec'\t%m='12'\t%Om='12'\t%d='255'\t%e='255'\t%Od='255'\t%Oe='255'\n"),
        fmt,
        std::chrono::month_day{std::chrono::December, std::chrono::day{255}});
#endif //  defined(_AIX)

  // Use the global locale (fr_FR)
#if defined(__APPLE__)
  check(SV("%b='jan'\t%B='janvier'\t%h='jan'\t%m='01'\t%Om='01'\t%d='00'\t%e=' 0'\t%Od='00'\t%Oe=' 0'\n"),
        lfmt,
        std::chrono::month_day{std::chrono::January, std::chrono::day{0}});
  check(SV("%b='fév'\t%B='février'\t%h='fév'\t%m='02'\t%Om='02'\t%d='01'\t%e=' 1'\t%Od='01'\t%Oe=' 1'\n"),
        lfmt,
        std::chrono::month_day{std::chrono::February, std::chrono::day{1}});
  check(SV("%b='mar'\t%B='mars'\t%h='mar'\t%m='03'\t%Om='03'\t%d='09'\t%e=' 9'\t%Od='09'\t%Oe=' 9'\n"),
        lfmt,
        std::chrono::month_day{std::chrono::March, std::chrono::day{9}});
  check(SV("%b='avr'\t%B='avril'\t%h='avr'\t%m='04'\t%Om='04'\t%d='10'\t%e='10'\t%Od='10'\t%Oe='10'\n"),
        lfmt,
        std::chrono::month_day{std::chrono::April, std::chrono::day{10}});
  check(SV("%b='mai'\t%B='mai'\t%h='mai'\t%m='05'\t%Om='05'\t%d='28'\t%e='28'\t%Od='28'\t%Oe='28'\n"),
        lfmt,
        std::chrono::month_day{std::chrono::May, std::chrono::day{28}});
  check(SV("%b='jui'\t%B='juin'\t%h='jui'\t%m='06'\t%Om='06'\t%d='29'\t%e='29'\t%Od='29'\t%Oe='29'\n"),
        lfmt,
        std::chrono::month_day{std::chrono::June, std::chrono::day{29}});
  check(SV("%b='jul'\t%B='juillet'\t%h='jul'\t%m='07'\t%Om='07'\t%d='30'\t%e='30'\t%Od='30'\t%Oe='30'\n"),
        lfmt,
        std::chrono::month_day{std::chrono::July, std::chrono::day{30}});
  check(SV("%b='aoû'\t%B='août'\t%h='aoû'\t%m='08'\t%Om='08'\t%d='31'\t%e='31'\t%Od='31'\t%Oe='31'\n"),
        lfmt,
        std::chrono::month_day{std::chrono::August, std::chrono::day{31}});
  check(SV("%b='sep'\t%B='septembre'\t%h='sep'\t%m='09'\t%Om='09'\t%d='32'\t%e='32'\t%Od='32'\t%Oe='32'\n"),
        lfmt,
        std::chrono::month_day{std::chrono::September, std::chrono::day{32}});
  check(SV("%b='oct'\t%B='octobre'\t%h='oct'\t%m='10'\t%Om='10'\t%d='99'\t%e='99'\t%Od='99'\t%Oe='99'\n"),
        lfmt,
        std::chrono::month_day{std::chrono::October, std::chrono::day{99}});
  check(SV("%b='nov'\t%B='novembre'\t%h='nov'\t%m='11'\t%Om='11'\t%d='100'\t%e='100'\t%Od='100'\t%Oe='100'\n"),
        lfmt,
        std::chrono::month_day{std::chrono::November, std::chrono::day{100}});
  check(SV("%b='déc'\t%B='décembre'\t%h='déc'\t%m='12'\t%Om='12'\t%d='255'\t%e='255'\t%Od='255'\t%Oe='255'\n"),
        lfmt,
        std::chrono::month_day{std::chrono::December, std::chrono::day{255}});
#else    // defined(__APPLE__)
  check(SV("%b='janv.'\t%B='janvier'\t%h='janv.'\t%m='01'\t%Om='01'\t%d='00'\t%e=' 0'\t%Od='00'\t%Oe=' 0'\n"),
        lfmt,
        std::chrono::month_day{std::chrono::January, std::chrono::day{0}});
  check(SV("%b='févr.'\t%B='février'\t%h='févr.'\t%m='02'\t%Om='02'\t%d='01'\t%e=' 1'\t%Od='01'\t%Oe=' 1'\n"),
        lfmt,
        std::chrono::month_day{std::chrono::February, std::chrono::day{1}});
  check(SV("%b='mars'\t%B='mars'\t%h='mars'\t%m='03'\t%Om='03'\t%d='09'\t%e=' 9'\t%Od='09'\t%Oe=' 9'\n"),
        lfmt,
        std::chrono::month_day{std::chrono::March, std::chrono::day{9}});
  check(
#  if defined(_WIN32) || defined(_AIX)
      SV("%b='avr.'\t%B='avril'\t%h='avr.'\t%m='04'\t%Om='04'\t%d='10'\t%e='10'\t%Od='10'\t%Oe='10'\n"),
#  else  // defined(_WIN32) || defined(_AIX)
      SV("%b='avril'\t%B='avril'\t%h='avril'\t%m='04'\t%Om='04'\t%d='10'\t%e='10'\t%Od='10'\t%Oe='10'\n"),
#  endif // defined(_WIN32) || defined(_AIX)
      lfmt,
      std::chrono::month_day{std::chrono::April, std::chrono::day{10}});
  check(SV("%b='mai'\t%B='mai'\t%h='mai'\t%m='05'\t%Om='05'\t%d='28'\t%e='28'\t%Od='28'\t%Oe='28'\n"),
        lfmt,
        std::chrono::month_day{std::chrono::May, std::chrono::day{28}});
  check(SV("%b='juin'\t%B='juin'\t%h='juin'\t%m='06'\t%Om='06'\t%d='29'\t%e='29'\t%Od='29'\t%Oe='29'\n"),
        lfmt,
        std::chrono::month_day{std::chrono::June, std::chrono::day{29}});
  check(SV("%b='juil.'\t%B='juillet'\t%h='juil.'\t%m='07'\t%Om='07'\t%d='30'\t%e='30'\t%Od='30'\t%Oe='30'\n"),
        lfmt,
        std::chrono::month_day{std::chrono::July, std::chrono::day{30}});
  check(SV("%b='août'\t%B='août'\t%h='août'\t%m='08'\t%Om='08'\t%d='31'\t%e='31'\t%Od='31'\t%Oe='31'\n"),
        lfmt,
        std::chrono::month_day{std::chrono::August, std::chrono::day{31}});
  check(SV("%b='sept.'\t%B='septembre'\t%h='sept.'\t%m='09'\t%Om='09'\t%d='32'\t%e='32'\t%Od='32'\t%Oe='32'\n"),
        lfmt,
        std::chrono::month_day{std::chrono::September, std::chrono::day{32}});
  check(SV("%b='oct.'\t%B='octobre'\t%h='oct.'\t%m='10'\t%Om='10'\t%d='99'\t%e='99'\t%Od='99'\t%Oe='99'\n"),
        lfmt,
        std::chrono::month_day{std::chrono::October, std::chrono::day{99}});
#  if defined(_AIX)
  check(SV("%b='nov.'\t%B='novembre'\t%h='nov.'\t%m='11'\t%Om='11'\t%d='00'\t%e=' 0'\t%Od='00'\t%Oe=' 0'\n"),
        lfmt,
        std::chrono::month_day{std::chrono::November, std::chrono::day{100}});
  check(SV("%b='déc.'\t%B='décembre'\t%h='déc.'\t%m='12'\t%Om='12'\t%d='55'\t%e='55'\t%Od='55'\t%Oe='55'\n"),
        lfmt,
        std::chrono::month_day{std::chrono::December, std::chrono::day{255}});
#  else  //   defined(_AIX)
  check(SV("%b='nov.'\t%B='novembre'\t%h='nov.'\t%m='11'\t%Om='11'\t%d='100'\t%e='100'\t%Od='100'\t%Oe='100'\n"),
        lfmt,
        std::chrono::month_day{std::chrono::November, std::chrono::day{100}});
  check(SV("%b='déc.'\t%B='décembre'\t%h='déc.'\t%m='12'\t%Om='12'\t%d='255'\t%e='255'\t%Od='255'\t%Oe='255'\n"),
        lfmt,
        std::chrono::month_day{std::chrono::December, std::chrono::day{255}});
#  endif //   defined(_AIX)
#endif   // defined(__APPLE__)

  // Use supplied locale (ja_JP)
#if defined(_WIN32)
  check(loc,
        SV("%b='1'\t%B='1月'\t%h='1'\t%m='01'\t%Om='01'\t%d='00'\t%e=' 0'\t%Od='〇'\t%Oe='〇'\n"),
        lfmt,
        std::chrono::month_day{std::chrono::January, std::chrono::day{0}});
  check(loc,
        SV("%b='2'\t%B='2月'\t%h='2'\t%m='02'\t%Om='02'\t%d='01'\t%e=' 1'\t%Od='一'\t%Oe='一'\n"),
        lfmt,
        std::chrono::month_day{std::chrono::February, std::chrono::day{1}});
  check(loc,
        SV("%b='3'\t%B='3月'\t%h='3'\t%m='03'\t%Om='03'\t%d='09'\t%e=' 9'\t%Od='九'\t%Oe='九'\n"),
        lfmt,
        std::chrono::month_day{std::chrono::March, std::chrono::day{9}});
  check(loc,
        SV("%b='4'\t%B='4月'\t%h='4'\t%m='04'\t%Om='04'\t%d='10'\t%e='10'\t%Od='十'\t%Oe='十'\n"),
        lfmt,
        std::chrono::month_day{std::chrono::April, std::chrono::day{10}});
  check(loc,
        SV("%b='5'\t%B='5月'\t%h='5'\t%m='05'\t%Om='05'\t%d='28'\t%e='28'\t%Od='二十八'\t%Oe='二十八'\n"),
        lfmt,
        std::chrono::month_day{std::chrono::May, std::chrono::day{28}});
  check(loc,
        SV("%b='6'\t%B='6月'\t%h='6'\t%m='06'\t%Om='06'\t%d='29'\t%e='29'\t%Od='二十九'\t%Oe='二十九'\n"),
        lfmt,
        std::chrono::month_day{std::chrono::June, std::chrono::day{29}});
  check(loc,
        SV("%b='7'\t%B='7月'\t%h='7'\t%m='07'\t%Om='07'\t%d='30'\t%e='30'\t%Od='三十'\t%Oe='三十'\n"),
        lfmt,
        std::chrono::month_day{std::chrono::July, std::chrono::day{30}});
  check(loc,
        SV("%b='8'\t%B='8月'\t%h='8'\t%m='08'\t%Om='08'\t%d='31'\t%e='31'\t%Od='三十一'\t%Oe='三十一'\n"),
        lfmt,
        std::chrono::month_day{std::chrono::August, std::chrono::day{31}});
  check(loc,
        SV("%b='9'\t%B='9月'\t%h='9'\t%m='09'\t%Om='09'\t%d='32'\t%e='32'\t%Od='三十二'\t%Oe='三十二'\n"),
        lfmt,
        std::chrono::month_day{std::chrono::September, std::chrono::day{32}});
  check(loc,
        SV("%b='10'\t%B='10月'\t%h='10'\t%m='10'\t%Om='10'\t%d='99'\t%e='99'\t%Od='九十九'\t%Oe='九十九'\n"),
        lfmt,
        std::chrono::month_day{std::chrono::October, std::chrono::day{99}});
  check(loc,
        SV("%b='11'\t%B='11月'\t%h='11'\t%m='11'\t%Om='11'\t%d='100'\t%e='100'\t%Od='100'\t%Oe='100'\n"),
        lfmt,
        std::chrono::month_day{std::chrono::November, std::chrono::day{100}});
  check(loc,
        SV("%b='12'\t%B='12月'\t%h='12'\t%m='12'\t%Om='12'\t%d='255'\t%e='255'\t%Od='255'\t%Oe='255'\n"),
        lfmt,
        std::chrono::month_day{std::chrono::December, std::chrono::day{255}});
#elif defined(_AIX)      // defined(_WIN32)
  check(loc,
        SV("%b='1月'\t%B='1月'\t%h='1月'\t%m='01'\t%Om='01'\t%d='00'\t%e=' 0'\t%Od='00'\t%Oe=' 0'\n"),
        lfmt,
        std::chrono::month_day{std::chrono::January, std::chrono::day{0}});
  check(loc,
        SV("%b='2月'\t%B='2月'\t%h='2月'\t%m='02'\t%Om='02'\t%d='01'\t%e=' 1'\t%Od='01'\t%Oe=' 1'\n"),
        lfmt,
        std::chrono::month_day{std::chrono::February, std::chrono::day{1}});
  check(loc,
        SV("%b='3月'\t%B='3月'\t%h='3月'\t%m='03'\t%Om='03'\t%d='09'\t%e=' 9'\t%Od='09'\t%Oe=' 9'\n"),
        lfmt,
        std::chrono::month_day{std::chrono::March, std::chrono::day{9}});
  check(loc,
        SV("%b='4月'\t%B='4月'\t%h='4月'\t%m='04'\t%Om='04'\t%d='10'\t%e='10'\t%Od='10'\t%Oe='10'\n"),
        lfmt,
        std::chrono::month_day{std::chrono::April, std::chrono::day{10}});
  check(loc,
        SV("%b='5月'\t%B='5月'\t%h='5月'\t%m='05'\t%Om='05'\t%d='28'\t%e='28'\t%Od='28'\t%Oe='28'\n"),
        lfmt,
        std::chrono::month_day{std::chrono::May, std::chrono::day{28}});
  check(loc,
        SV("%b='6月'\t%B='6月'\t%h='6月'\t%m='06'\t%Om='06'\t%d='29'\t%e='29'\t%Od='29'\t%Oe='29'\n"),
        lfmt,
        std::chrono::month_day{std::chrono::June, std::chrono::day{29}});
  check(loc,
        SV("%b='7月'\t%B='7月'\t%h='7月'\t%m='07'\t%Om='07'\t%d='30'\t%e='30'\t%Od='30'\t%Oe='30'\n"),
        lfmt,
        std::chrono::month_day{std::chrono::July, std::chrono::day{30}});
  check(loc,
        SV("%b='8月'\t%B='8月'\t%h='8月'\t%m='08'\t%Om='08'\t%d='31'\t%e='31'\t%Od='31'\t%Oe='31'\n"),
        lfmt,
        std::chrono::month_day{std::chrono::August, std::chrono::day{31}});
  check(loc,
        SV("%b='9月'\t%B='9月'\t%h='9月'\t%m='09'\t%Om='09'\t%d='32'\t%e='32'\t%Od='32'\t%Oe='32'\n"),
        lfmt,
        std::chrono::month_day{std::chrono::September, std::chrono::day{32}});
  check(loc,
        SV("%b='10月'\t%B='10月'\t%h='10月'\t%m='10'\t%Om='10'\t%d='99'\t%e='99'\t%Od='99'\t%Oe='99'\n"),
        lfmt,
        std::chrono::month_day{std::chrono::October, std::chrono::day{99}});
  check(loc,
        SV("%b='11月'\t%B='11月'\t%h='11月'\t%m='11'\t%Om='11'\t%d='00'\t%e=' 0'\t%Od='00'\t%Oe=' 0'\n"),
        lfmt,
        std::chrono::month_day{std::chrono::November, std::chrono::day{100}});
  check(loc,
        SV("%b='12月'\t%B='12月'\t%h='12月'\t%m='12'\t%Om='12'\t%d='55'\t%e='55'\t%Od='55'\t%Oe='55'\n"),
        lfmt,
        std::chrono::month_day{std::chrono::December, std::chrono::day{255}});
#elif defined(__APPLE__) // defined(_WIN32)
  check(loc,
        SV("%b=' 1'\t%B='1月'\t%h=' 1'\t%m='01'\t%Om='01'\t%d='00'\t%e=' 0'\t%Od='00'\t%Oe=' 0'\n"),
        lfmt,
        std::chrono::month_day{std::chrono::January, std::chrono::day{0}});
  check(loc,
        SV("%b=' 2'\t%B='2月'\t%h=' 2'\t%m='02'\t%Om='02'\t%d='01'\t%e=' 1'\t%Od='01'\t%Oe=' 1'\n"),
        lfmt,
        std::chrono::month_day{std::chrono::February, std::chrono::day{1}});
  check(loc,
        SV("%b=' 3'\t%B='3月'\t%h=' 3'\t%m='03'\t%Om='03'\t%d='09'\t%e=' 9'\t%Od='09'\t%Oe=' 9'\n"),
        lfmt,
        std::chrono::month_day{std::chrono::March, std::chrono::day{9}});
  check(loc,
        SV("%b=' 4'\t%B='4月'\t%h=' 4'\t%m='04'\t%Om='04'\t%d='10'\t%e='10'\t%Od='10'\t%Oe='10'\n"),
        lfmt,
        std::chrono::month_day{std::chrono::April, std::chrono::day{10}});
  check(loc,
        SV("%b=' 5'\t%B='5月'\t%h=' 5'\t%m='05'\t%Om='05'\t%d='28'\t%e='28'\t%Od='28'\t%Oe='28'\n"),
        lfmt,
        std::chrono::month_day{std::chrono::May, std::chrono::day{28}});
  check(loc,
        SV("%b=' 6'\t%B='6月'\t%h=' 6'\t%m='06'\t%Om='06'\t%d='29'\t%e='29'\t%Od='29'\t%Oe='29'\n"),
        lfmt,
        std::chrono::month_day{std::chrono::June, std::chrono::day{29}});
  check(loc,
        SV("%b=' 7'\t%B='7月'\t%h=' 7'\t%m='07'\t%Om='07'\t%d='30'\t%e='30'\t%Od='30'\t%Oe='30'\n"),
        lfmt,
        std::chrono::month_day{std::chrono::July, std::chrono::day{30}});
  check(loc,
        SV("%b=' 8'\t%B='8月'\t%h=' 8'\t%m='08'\t%Om='08'\t%d='31'\t%e='31'\t%Od='31'\t%Oe='31'\n"),
        lfmt,
        std::chrono::month_day{std::chrono::August, std::chrono::day{31}});
  check(loc,
        SV("%b=' 9'\t%B='9月'\t%h=' 9'\t%m='09'\t%Om='09'\t%d='32'\t%e='32'\t%Od='32'\t%Oe='32'\n"),
        lfmt,
        std::chrono::month_day{std::chrono::September, std::chrono::day{32}});
  check(loc,
        SV("%b='10'\t%B='10月'\t%h='10'\t%m='10'\t%Om='10'\t%d='99'\t%e='99'\t%Od='99'\t%Oe='99'\n"),
        lfmt,
        std::chrono::month_day{std::chrono::October, std::chrono::day{99}});
  check(loc,
        SV("%b='11'\t%B='11月'\t%h='11'\t%m='11'\t%Om='11'\t%d='100'\t%e='100'\t%Od='100'\t%Oe='100'\n"),
        lfmt,
        std::chrono::month_day{std::chrono::November, std::chrono::day{100}});
  check(loc,
        SV("%b='12'\t%B='12月'\t%h='12'\t%m='12'\t%Om='12'\t%d='255'\t%e='255'\t%Od='255'\t%Oe='255'\n"),
        lfmt,
        std::chrono::month_day{std::chrono::December, std::chrono::day{255}});
#else                    // defined(_WIN32)
  check(loc,
        SV("%b=' 1月'\t%B='1月'\t%h=' 1月'\t%m='01'\t%Om='一'\t%d='00'\t%e=' 0'\t%Od='〇'\t%Oe='〇'\n"),
        lfmt,
        std::chrono::month_day{std::chrono::January, std::chrono::day{0}});
  check(loc,
        SV("%b=' 2月'\t%B='2月'\t%h=' 2月'\t%m='02'\t%Om='二'\t%d='01'\t%e=' 1'\t%Od='一'\t%Oe='一'\n"),
        lfmt,
        std::chrono::month_day{std::chrono::February, std::chrono::day{1}});
  check(loc,
        SV("%b=' 3月'\t%B='3月'\t%h=' 3月'\t%m='03'\t%Om='三'\t%d='09'\t%e=' 9'\t%Od='九'\t%Oe='九'\n"),
        lfmt,
        std::chrono::month_day{std::chrono::March, std::chrono::day{9}});
  check(loc,
        SV("%b=' 4月'\t%B='4月'\t%h=' 4月'\t%m='04'\t%Om='四'\t%d='10'\t%e='10'\t%Od='十'\t%Oe='十'\n"),
        lfmt,
        std::chrono::month_day{std::chrono::April, std::chrono::day{10}});
  check(loc,
        SV("%b=' 5月'\t%B='5月'\t%h=' 5月'\t%m='05'\t%Om='五'\t%d='28'\t%e='28'\t%Od='二十八'\t%Oe='二十八'\n"),
        lfmt,
        std::chrono::month_day{std::chrono::May, std::chrono::day{28}});
  check(loc,
        SV("%b=' 6月'\t%B='6月'\t%h=' 6月'\t%m='06'\t%Om='六'\t%d='29'\t%e='29'\t%Od='二十九'\t%Oe='二十九'\n"),
        lfmt,
        std::chrono::month_day{std::chrono::June, std::chrono::day{29}});
  check(loc,
        SV("%b=' 7月'\t%B='7月'\t%h=' 7月'\t%m='07'\t%Om='七'\t%d='30'\t%e='30'\t%Od='三十'\t%Oe='三十'\n"),
        lfmt,
        std::chrono::month_day{std::chrono::July, std::chrono::day{30}});
  check(loc,
        SV("%b=' 8月'\t%B='8月'\t%h=' 8月'\t%m='08'\t%Om='八'\t%d='31'\t%e='31'\t%Od='三十一'\t%Oe='三十一'\n"),
        lfmt,
        std::chrono::month_day{std::chrono::August, std::chrono::day{31}});
  check(loc,
        SV("%b=' 9月'\t%B='9月'\t%h=' 9月'\t%m='09'\t%Om='九'\t%d='32'\t%e='32'\t%Od='三十二'\t%Oe='三十二'\n"),
        lfmt,
        std::chrono::month_day{std::chrono::September, std::chrono::day{32}});
  check(loc,
        SV("%b='10月'\t%B='10月'\t%h='10月'\t%m='10'\t%Om='十'\t%d='99'\t%e='99'\t%Od='九十九'\t%Oe='九十九'\n"),
        lfmt,
        std::chrono::month_day{std::chrono::October, std::chrono::day{99}});
  check(loc,
        SV("%b='11月'\t%B='11月'\t%h='11月'\t%m='11'\t%Om='十一'\t%d='100'\t%e='100'\t%Od='100'\t%Oe='100'\n"),
        lfmt,
        std::chrono::month_day{std::chrono::November, std::chrono::day{100}});
  check(loc,
        SV("%b='12月'\t%B='12月'\t%h='12月'\t%m='12'\t%Om='十二'\t%d='255'\t%e='255'\t%Od='255'\t%Oe='255'\n"),
        lfmt,
        std::chrono::month_day{std::chrono::December, std::chrono::day{255}});
#endif                   //  defined(_WIN32)

  std::locale::global(std::locale::classic());
}

template <class CharT>
static void test() {
  test_no_chrono_specs<CharT>();
  test_valid_values<CharT>();
  check_invalid_types<CharT>({SV("b"), SV("B"), SV("d"), SV("e"), SV("h"), SV("m"), SV("Od"), SV("Oe"), SV("Om")},
                             std::chrono::month_day{std::chrono::January, std::chrono::day{31}});

  check_exception("Expected '%' or '}' in the chrono format-string",
                  SV("{:A"),
                  std::chrono::month_day{std::chrono::January, std::chrono::day{31}});
  check_exception("The chrono-specs contains a '{'",
                  SV("{:%%{"),
                  std::chrono::month_day{std::chrono::January, std::chrono::day{31}});
  check_exception("End of input while parsing the modifier chrono conversion-spec",
                  SV("{:%"),
                  std::chrono::month_day{std::chrono::January, std::chrono::day{31}});
  check_exception("End of input while parsing the modifier E",
                  SV("{:%E"),
                  std::chrono::month_day{std::chrono::January, std::chrono::day{31}});
  check_exception("End of input while parsing the modifier O",
                  SV("{:%O"),
                  std::chrono::month_day{std::chrono::January, std::chrono::day{31}});

  // Precision not allowed
  check_exception("Expected '%' or '}' in the chrono format-string",
                  SV("{:.3}"),
                  std::chrono::month_day{std::chrono::January, std::chrono::day{31}});
}

int main(int, char**) {
  test<char>();

#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  test<wchar_t>();
#endif

  return 0;
}
