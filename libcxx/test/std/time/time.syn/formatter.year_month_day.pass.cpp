//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: no-localization
// UNSUPPORTED: GCC-ALWAYS_INLINE-FIXME

// TODO FMT This test should not require std::to_chars(floating-point)
// XFAIL: availability-fp_to_chars-missing

// TODO FMT Investigate Windows issues.
// XFAIL: msvc

// REQUIRES: locale.fr_FR.UTF-8
// REQUIRES: locale.ja_JP.UTF-8

// <chrono>

// template<class charT> struct formatter<chrono::year_month_day, charT>;

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
  // Valid year, valid month, valid day
  check(SV("1970-01-31"),
        SV("{}"),
        std::chrono::year_month_day{std::chrono::year{1970}, std::chrono::month{1}, std::chrono::day{31}});
  check(SV("*1970-01-31*"),
        SV("{:*^12}"),
        std::chrono::year_month_day{std::chrono::year{1970}, std::chrono::month{1}, std::chrono::day{31}});
  check(SV("*1970-01-31"),
        SV("{:*>11}"),
        std::chrono::year_month_day{std::chrono::year{1970}, std::chrono::month{1}, std::chrono::day{31}});

  // Valid year, valid month, invalid day
  check(SV("1970-02-31 is not a valid date"),
        SV("{}"),
        std::chrono::year_month_day{std::chrono::year{1970}, std::chrono::month{2}, std::chrono::day{31}});
  check(SV("*1970-02-31 is not a valid date*"),
        SV("{:*^32}"),
        std::chrono::year_month_day{std::chrono::year{1970}, std::chrono::month{2}, std::chrono::day{31}});
  check(SV("*1970-02-31 is not a valid date"),
        SV("{:*>31}"),
        std::chrono::year_month_day{std::chrono::year{1970}, std::chrono::month{2}, std::chrono::day{31}});

  // Valid year, invalid month, valid day
  check(SV("1970-00-31 is not a valid date"),
        SV("{}"),
        std::chrono::year_month_day{std::chrono::year{1970}, std::chrono::month{0}, std::chrono::day{31}});
  check(SV("*1970-00-31 is not a valid date*"),
        SV("{:*^32}"),
        std::chrono::year_month_day{std::chrono::year{1970}, std::chrono::month{0}, std::chrono::day{31}});
  check(SV("*1970-00-31 is not a valid date"),
        SV("{:*>31}"),
        std::chrono::year_month_day{std::chrono::year{1970}, std::chrono::month{0}, std::chrono::day{31}});

  // Valid year, invalid month, invalid day
  check(SV("1970-00-32 is not a valid date"),
        SV("{}"),
        std::chrono::year_month_day{std::chrono::year{1970}, std::chrono::month{0}, std::chrono::day{32}});
  check(SV("*1970-00-32 is not a valid date*"),
        SV("{:*^32}"),
        std::chrono::year_month_day{std::chrono::year{1970}, std::chrono::month{0}, std::chrono::day{32}});
  check(SV("*1970-00-32 is not a valid date"),
        SV("{:*>31}"),
        std::chrono::year_month_day{std::chrono::year{1970}, std::chrono::month{0}, std::chrono::day{32}});

  // Invalid year, valid month, valid day
  check(SV("-32768-01-31 is not a valid date"),
        SV("{}"),
        std::chrono::year_month_day{std::chrono::year{-32768}, std::chrono::month{1}, std::chrono::day{31}});
  check(SV("*-32768-01-31 is not a valid date*"),
        SV("{:*^34}"),
        std::chrono::year_month_day{std::chrono::year{-32768}, std::chrono::month{1}, std::chrono::day{31}});
  check(SV("*-32768-01-31 is not a valid date"),
        SV("{:*>33}"),
        std::chrono::year_month_day{std::chrono::year{-32768}, std::chrono::month{1}, std::chrono::day{31}});

  // Invalid year, valid month, invalid day
  check(SV("-32768-01-32 is not a valid date"),
        SV("{}"),
        std::chrono::year_month_day{std::chrono::year{-32768}, std::chrono::month{1}, std::chrono::day{32}});
  check(SV("*-32768-01-32 is not a valid date*"),
        SV("{:*^34}"),
        std::chrono::year_month_day{std::chrono::year{-32768}, std::chrono::month{1}, std::chrono::day{32}});
  check(SV("*-32768-01-32 is not a valid date"),
        SV("{:*>33}"),
        std::chrono::year_month_day{std::chrono::year{-32768}, std::chrono::month{1}, std::chrono::day{32}});

  // Invalid year, invalid month, valid day
  check(SV("-32768-00-31 is not a valid date"),
        SV("{}"),
        std::chrono::year_month_day{std::chrono::year{-32768}, std::chrono::month{0}, std::chrono::day{31}});
  check(SV("*-32768-00-31 is not a valid date*"),
        SV("{:*^34}"),
        std::chrono::year_month_day{std::chrono::year{-32768}, std::chrono::month{0}, std::chrono::day{31}});
  check(SV("*-32768-00-31 is not a valid date"),
        SV("{:*>33}"),
        std::chrono::year_month_day{std::chrono::year{-32768}, std::chrono::month{0}, std::chrono::day{31}});

  // Invalid year, invalid month, invalid day
  check(SV("-32768-00-32 is not a valid date"),
        SV("{}"),
        std::chrono::year_month_day{std::chrono::year{-32768}, std::chrono::month{0}, std::chrono::day{32}});
  check(SV("*-32768-00-32 is not a valid date*"),
        SV("{:*^34}"),
        std::chrono::year_month_day{std::chrono::year{-32768}, std::chrono::month{0}, std::chrono::day{32}});
  check(SV("*-32768-00-32 is not a valid date"),
        SV("{:*>33}"),
        std::chrono::year_month_day{std::chrono::year{-32768}, std::chrono::month{0}, std::chrono::day{32}});
}

// TODO FMT Should x throw?
template <class CharT>
static void test_invalid_values() {
  // Test that %a, %A, %b, %B, %h, %j, %u, %U, %V, %w, %W, %Ou, %OU, %OV, %Ow, and %OW throw an exception.
  check_exception("Formatting a weekday name needs a valid weekday",
                  SV("{:%A}"),
                  std::chrono::year_month_day{std::chrono::year{1970}, std::chrono::month{1}, std::chrono::day{0}});
  check_exception("Formatting a weekday name needs a valid weekday",
                  SV("{:%A}"),
                  std::chrono::year_month_day{std::chrono::year{1970}, std::chrono::month{1}, std::chrono::day{32}});
  check_exception("Formatting a weekday name needs a valid weekday",
                  SV("{:%A}"),
                  std::chrono::year_month_day{
                      std::chrono::year{1970}, std::chrono::month{2}, std::chrono::day{29}}); // not a leap year
  check_exception("Formatting a weekday name needs a valid weekday",
                  SV("{:%A}"),
                  std::chrono::year_month_day{std::chrono::year{1970}, std::chrono::month{0}, std::chrono::day{31}});
  check_exception("Formatting a weekday name needs a valid weekday",
                  SV("{:%A}"),
                  std::chrono::year_month_day{std::chrono::year{-32768}, std::chrono::month{1}, std::chrono::day{31}});

  check_exception("Formatting a weekday name needs a valid weekday",
                  SV("{:%a}"),
                  std::chrono::year_month_day{std::chrono::year{1970}, std::chrono::month{1}, std::chrono::day{0}});
  check_exception("Formatting a weekday name needs a valid weekday",
                  SV("{:%a}"),
                  std::chrono::year_month_day{std::chrono::year{1970}, std::chrono::month{1}, std::chrono::day{32}});
  check_exception("Formatting a weekday name needs a valid weekday",
                  SV("{:%a}"),
                  std::chrono::year_month_day{
                      std::chrono::year{1970}, std::chrono::month{2}, std::chrono::day{29}}); // not a leap year
  check_exception("Formatting a weekday name needs a valid weekday",
                  SV("{:%a}"),
                  std::chrono::year_month_day{std::chrono::year{1970}, std::chrono::month{0}, std::chrono::day{31}});
  check_exception("Formatting a weekday name needs a valid weekday",
                  SV("{:%a}"),
                  std::chrono::year_month_day{std::chrono::year{-32768}, std::chrono::month{1}, std::chrono::day{31}});

  check_exception("Formatting a month name from an invalid month number",
                  SV("{:%B}"),
                  std::chrono::year_month_day{std::chrono::year{1970}, std::chrono::month{0}, std::chrono::day{31}});
  check_exception("Formatting a month name from an invalid month number",
                  SV("{:%B}"),
                  std::chrono::year_month_day{std::chrono::year{1970}, std::chrono::month{13}, std::chrono::day{31}});
  check_exception("Formatting a month name from an invalid month number",
                  SV("{:%B}"),
                  std::chrono::year_month_day{std::chrono::year{1970}, std::chrono::month{255}, std::chrono::day{31}});

  check_exception("Formatting a month name from an invalid month number",
                  SV("{:%b}"),
                  std::chrono::year_month_day{std::chrono::year{1970}, std::chrono::month{200}, std::chrono::day{31}});
  check_exception("Formatting a month name from an invalid month number",
                  SV("{:%b}"),
                  std::chrono::year_month_day{std::chrono::year{1970}, std::chrono::month{13}, std::chrono::day{31}});
  check_exception("Formatting a month name from an invalid month number",
                  SV("{:%b}"),
                  std::chrono::year_month_day{std::chrono::year{1970}, std::chrono::month{255}, std::chrono::day{31}});

  check_exception("Formatting a month name from an invalid month number",
                  SV("{:%h}"),
                  std::chrono::year_month_day{std::chrono::year{1970}, std::chrono::month{0}, std::chrono::day{31}});
  check_exception("Formatting a month name from an invalid month number",
                  SV("{:%h}"),
                  std::chrono::year_month_day{std::chrono::year{1970}, std::chrono::month{13}, std::chrono::day{31}});
  check_exception("Formatting a month name from an invalid month number",
                  SV("{:%h}"),
                  std::chrono::year_month_day{std::chrono::year{1970}, std::chrono::month{255}, std::chrono::day{31}});

  check_exception("Formatting a day of year needs a valid date",
                  SV("{:%j}"),
                  std::chrono::year_month_day{std::chrono::year{1970}, std::chrono::month{1}, std::chrono::day{0}});
  check_exception("Formatting a day of year needs a valid date",
                  SV("{:%j}"),
                  std::chrono::year_month_day{std::chrono::year{1970}, std::chrono::month{1}, std::chrono::day{32}});
  check_exception("Formatting a day of year needs a valid date",
                  SV("{:%j}"),
                  std::chrono::year_month_day{
                      std::chrono::year{1970}, std::chrono::month{2}, std::chrono::day{29}}); // not a leap year
  check_exception("Formatting a day of year needs a valid date",
                  SV("{:%j}"),
                  std::chrono::year_month_day{std::chrono::year{1970}, std::chrono::month{0}, std::chrono::day{31}});
  check_exception("Formatting a day of year needs a valid date",
                  SV("{:%j}"),
                  std::chrono::year_month_day{std::chrono::year{-32768}, std::chrono::month{1}, std::chrono::day{31}});

  check_exception("Formatting a weekday needs a valid weekday",
                  SV("{:%u}"),
                  std::chrono::year_month_day{std::chrono::year{1970}, std::chrono::month{1}, std::chrono::day{0}});
  check_exception("Formatting a weekday needs a valid weekday",
                  SV("{:%u}"),
                  std::chrono::year_month_day{std::chrono::year{1970}, std::chrono::month{1}, std::chrono::day{32}});
  check_exception("Formatting a weekday needs a valid weekday",
                  SV("{:%u}"),
                  std::chrono::year_month_day{
                      std::chrono::year{1970}, std::chrono::month{2}, std::chrono::day{29}}); // not a leap year
  check_exception("Formatting a weekday needs a valid weekday",
                  SV("{:%u}"),
                  std::chrono::year_month_day{std::chrono::year{1970}, std::chrono::month{0}, std::chrono::day{31}});
  check_exception("Formatting a weekday needs a valid weekday",
                  SV("{:%u}"),
                  std::chrono::year_month_day{std::chrono::year{-32768}, std::chrono::month{1}, std::chrono::day{31}});

  check_exception("Formatting a week of year needs a valid date",
                  SV("{:%U}"),
                  std::chrono::year_month_day{std::chrono::year{1970}, std::chrono::month{1}, std::chrono::day{0}});
  check_exception("Formatting a week of year needs a valid date",
                  SV("{:%U}"),
                  std::chrono::year_month_day{std::chrono::year{1970}, std::chrono::month{1}, std::chrono::day{32}});
  check_exception("Formatting a week of year needs a valid date",
                  SV("{:%U}"),
                  std::chrono::year_month_day{
                      std::chrono::year{1970}, std::chrono::month{2}, std::chrono::day{29}}); // not a leap year
  check_exception("Formatting a week of year needs a valid date",
                  SV("{:%U}"),
                  std::chrono::year_month_day{std::chrono::year{1970}, std::chrono::month{0}, std::chrono::day{31}});
  check_exception("Formatting a week of year needs a valid date",
                  SV("{:%U}"),
                  std::chrono::year_month_day{std::chrono::year{-32768}, std::chrono::month{1}, std::chrono::day{31}});

  check_exception("Formatting a week of year needs a valid date",
                  SV("{:%V}"),
                  std::chrono::year_month_day{std::chrono::year{1970}, std::chrono::month{1}, std::chrono::day{0}});
  check_exception("Formatting a week of year needs a valid date",
                  SV("{:%V}"),
                  std::chrono::year_month_day{std::chrono::year{1970}, std::chrono::month{1}, std::chrono::day{32}});
  check_exception("Formatting a week of year needs a valid date",
                  SV("{:%V}"),
                  std::chrono::year_month_day{
                      std::chrono::year{1970}, std::chrono::month{2}, std::chrono::day{29}}); // not a leap year
  check_exception("Formatting a week of year needs a valid date",
                  SV("{:%V}"),
                  std::chrono::year_month_day{std::chrono::year{1970}, std::chrono::month{0}, std::chrono::day{31}});
  check_exception("Formatting a week of year needs a valid date",
                  SV("{:%V}"),
                  std::chrono::year_month_day{std::chrono::year{-32768}, std::chrono::month{1}, std::chrono::day{31}});

  check_exception("Formatting a weekday needs a valid weekday",
                  SV("{:%w}"),
                  std::chrono::year_month_day{std::chrono::year{1970}, std::chrono::month{1}, std::chrono::day{0}});
  check_exception("Formatting a weekday needs a valid weekday",
                  SV("{:%w}"),
                  std::chrono::year_month_day{std::chrono::year{1970}, std::chrono::month{1}, std::chrono::day{32}});
  check_exception("Formatting a weekday needs a valid weekday",
                  SV("{:%w}"),
                  std::chrono::year_month_day{
                      std::chrono::year{1970}, std::chrono::month{2}, std::chrono::day{29}}); // not a leap year
  check_exception("Formatting a weekday needs a valid weekday",
                  SV("{:%w}"),
                  std::chrono::year_month_day{std::chrono::year{1970}, std::chrono::month{0}, std::chrono::day{31}});
  check_exception("Formatting a weekday needs a valid weekday",
                  SV("{:%w}"),
                  std::chrono::year_month_day{std::chrono::year{-32768}, std::chrono::month{1}, std::chrono::day{31}});

  check_exception("Formatting a week of year needs a valid date",
                  SV("{:%W}"),
                  std::chrono::year_month_day{std::chrono::year{1970}, std::chrono::month{1}, std::chrono::day{0}});
  check_exception("Formatting a week of year needs a valid date",
                  SV("{:%W}"),
                  std::chrono::year_month_day{std::chrono::year{1970}, std::chrono::month{1}, std::chrono::day{32}});
  check_exception("Formatting a week of year needs a valid date",
                  SV("{:%W}"),
                  std::chrono::year_month_day{
                      std::chrono::year{1970}, std::chrono::month{2}, std::chrono::day{29}}); // not a leap year
  check_exception("Formatting a week of year needs a valid date",
                  SV("{:%W}"),
                  std::chrono::year_month_day{std::chrono::year{1970}, std::chrono::month{0}, std::chrono::day{31}});
  check_exception("Formatting a week of year needs a valid date",
                  SV("{:%W}"),
                  std::chrono::year_month_day{std::chrono::year{-32768}, std::chrono::month{1}, std::chrono::day{31}});

  check_exception("Formatting a weekday needs a valid weekday",
                  SV("{:%Ou}"),
                  std::chrono::year_month_day{std::chrono::year{1970}, std::chrono::month{1}, std::chrono::day{0}});
  check_exception("Formatting a weekday needs a valid weekday",
                  SV("{:%Ou}"),
                  std::chrono::year_month_day{std::chrono::year{1970}, std::chrono::month{1}, std::chrono::day{32}});
  check_exception("Formatting a weekday needs a valid weekday",
                  SV("{:%Ou}"),
                  std::chrono::year_month_day{
                      std::chrono::year{1970}, std::chrono::month{2}, std::chrono::day{29}}); // not a leap year
  check_exception("Formatting a weekday needs a valid weekday",
                  SV("{:%Ou}"),
                  std::chrono::year_month_day{std::chrono::year{1970}, std::chrono::month{0}, std::chrono::day{31}});
  check_exception("Formatting a weekday needs a valid weekday",
                  SV("{:%Ou}"),
                  std::chrono::year_month_day{std::chrono::year{-32768}, std::chrono::month{1}, std::chrono::day{31}});

  check_exception("Formatting a week of year needs a valid date",
                  SV("{:%OU}"),
                  std::chrono::year_month_day{std::chrono::year{1970}, std::chrono::month{1}, std::chrono::day{0}});
  check_exception("Formatting a week of year needs a valid date",
                  SV("{:%OU}"),
                  std::chrono::year_month_day{std::chrono::year{1970}, std::chrono::month{1}, std::chrono::day{32}});
  check_exception("Formatting a week of year needs a valid date",
                  SV("{:%OU}"),
                  std::chrono::year_month_day{
                      std::chrono::year{1970}, std::chrono::month{2}, std::chrono::day{29}}); // not a leap year
  check_exception("Formatting a week of year needs a valid date",
                  SV("{:%OU}"),
                  std::chrono::year_month_day{std::chrono::year{1970}, std::chrono::month{0}, std::chrono::day{31}});
  check_exception("Formatting a week of year needs a valid date",
                  SV("{:%OU}"),
                  std::chrono::year_month_day{std::chrono::year{-32768}, std::chrono::month{1}, std::chrono::day{31}});

  check_exception("Formatting a week of year needs a valid date",
                  SV("{:%OV}"),
                  std::chrono::year_month_day{std::chrono::year{1970}, std::chrono::month{1}, std::chrono::day{0}});
  check_exception("Formatting a week of year needs a valid date",
                  SV("{:%OV}"),
                  std::chrono::year_month_day{std::chrono::year{1970}, std::chrono::month{1}, std::chrono::day{32}});
  check_exception("Formatting a week of year needs a valid date",
                  SV("{:%OV}"),
                  std::chrono::year_month_day{
                      std::chrono::year{1970}, std::chrono::month{2}, std::chrono::day{29}}); // not a leap year
  check_exception("Formatting a week of year needs a valid date",
                  SV("{:%OV}"),
                  std::chrono::year_month_day{std::chrono::year{1970}, std::chrono::month{0}, std::chrono::day{31}});
  check_exception("Formatting a week of year needs a valid date",
                  SV("{:%OV}"),
                  std::chrono::year_month_day{std::chrono::year{-32768}, std::chrono::month{1}, std::chrono::day{31}});

  check_exception("Formatting a weekday needs a valid weekday",
                  SV("{:%Ow}"),
                  std::chrono::year_month_day{std::chrono::year{1970}, std::chrono::month{1}, std::chrono::day{0}});
  check_exception("Formatting a weekday needs a valid weekday",
                  SV("{:%Ow}"),
                  std::chrono::year_month_day{std::chrono::year{1970}, std::chrono::month{1}, std::chrono::day{32}});
  check_exception("Formatting a weekday needs a valid weekday",
                  SV("{:%Ow}"),
                  std::chrono::year_month_day{
                      std::chrono::year{1970}, std::chrono::month{2}, std::chrono::day{29}}); // not a leap year
  check_exception("Formatting a weekday needs a valid weekday",
                  SV("{:%Ow}"),
                  std::chrono::year_month_day{std::chrono::year{1970}, std::chrono::month{0}, std::chrono::day{31}});
  check_exception("Formatting a weekday needs a valid weekday",
                  SV("{:%Ow}"),
                  std::chrono::year_month_day{std::chrono::year{-32768}, std::chrono::month{1}, std::chrono::day{31}});

  check_exception("Formatting a week of year needs a valid date",
                  SV("{:%OW}"),
                  std::chrono::year_month_day{std::chrono::year{1970}, std::chrono::month{1}, std::chrono::day{0}});
  check_exception("Formatting a week of year needs a valid date",
                  SV("{:%OW}"),
                  std::chrono::year_month_day{std::chrono::year{1970}, std::chrono::month{1}, std::chrono::day{32}});
  check_exception("Formatting a week of year needs a valid date",
                  SV("{:%OW}"),
                  std::chrono::year_month_day{
                      std::chrono::year{1970}, std::chrono::month{2}, std::chrono::day{29}}); // not a leap year
  check_exception("Formatting a week of year needs a valid date",
                  SV("{:%OW}"),
                  std::chrono::year_month_day{std::chrono::year{1970}, std::chrono::month{0}, std::chrono::day{31}});
  check_exception("Formatting a week of year needs a valid date",
                  SV("{:%OW}"),
                  std::chrono::year_month_day{std::chrono::year{-32768}, std::chrono::month{1}, std::chrono::day{31}});
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
#ifdef _WIN32
  check(SV("%b='Jan'\t%B='January'\t%h='Jan'\t%m='01'\t%Om='01'\t%d=''\t%e=''\t%Od=''\t%Oe=''\n"),
        fmt,
        std::chrono::year_month_day{std::chrono::year{1970}, std::chrono::January, std::chrono::day{0}});
#else
  check(SV("%b='Jan'\t%B='January'\t%h='Jan'\t%m='01'\t%Om='01'\t%d='00'\t%e=' 0'\t%Od='00'\t%Oe=' 0'\n"),
        fmt,
        std::chrono::year_month_day{std::chrono::year{1970}, std::chrono::January, std::chrono::day{0}});
#endif
  check(SV("%b='Feb'\t%B='February'\t%h='Feb'\t%m='02'\t%Om='02'\t%d='01'\t%e=' 1'\t%Od='01'\t%Oe=' 1'\n"),
        fmt,
        std::chrono::year_month_day{std::chrono::year{1970}, std::chrono::February, std::chrono::day{1}});
  check(SV("%b='Mar'\t%B='March'\t%h='Mar'\t%m='03'\t%Om='03'\t%d='09'\t%e=' 9'\t%Od='09'\t%Oe=' 9'\n"),
        fmt,
        std::chrono::year_month_day{std::chrono::year{1970}, std::chrono::March, std::chrono::day{9}});
  check(SV("%b='Apr'\t%B='April'\t%h='Apr'\t%m='04'\t%Om='04'\t%d='10'\t%e='10'\t%Od='10'\t%Oe='10'\n"),
        fmt,
        std::chrono::year_month_day{std::chrono::year{1970}, std::chrono::April, std::chrono::day{10}});
  check(SV("%b='May'\t%B='May'\t%h='May'\t%m='05'\t%Om='05'\t%d='28'\t%e='28'\t%Od='28'\t%Oe='28'\n"),
        fmt,
        std::chrono::year_month_day{std::chrono::year{1970}, std::chrono::May, std::chrono::day{28}});
  check(SV("%b='Jun'\t%B='June'\t%h='Jun'\t%m='06'\t%Om='06'\t%d='29'\t%e='29'\t%Od='29'\t%Oe='29'\n"),
        fmt,
        std::chrono::year_month_day{std::chrono::year{1970}, std::chrono::June, std::chrono::day{29}});
  check(SV("%b='Jul'\t%B='July'\t%h='Jul'\t%m='07'\t%Om='07'\t%d='30'\t%e='30'\t%Od='30'\t%Oe='30'\n"),
        fmt,
        std::chrono::year_month_day{std::chrono::year{1970}, std::chrono::July, std::chrono::day{30}});
  check(SV("%b='Aug'\t%B='August'\t%h='Aug'\t%m='08'\t%Om='08'\t%d='31'\t%e='31'\t%Od='31'\t%Oe='31'\n"),
        fmt,
        std::chrono::year_month_day{std::chrono::year{1970}, std::chrono::August, std::chrono::day{31}});
#ifdef _WIN32
  check(SV("%b='Sep'\t%B='September'\t%h='Sep'\t%m='09'\t%Om='09'\t%d=''\t%e=''\t%Od=''\t%Oe=''\n"),
        fmt,
        std::chrono::year_month_day{std::chrono::year{1970}, std::chrono::September, std::chrono::day{32}});
  check(SV("%b='Oct'\t%B='October'\t%h='Oct'\t%m='10'\t%Om='10'\t%d=''\t%e=''\t%Od=''\t%Oe=''\n"),
        fmt,
        std::chrono::year_month_day{std::chrono::year{1970}, std::chrono::October, std::chrono::day{99}});
  check(SV("%b='Nov'\t%B='November'\t%h='Nov'\t%m='11'\t%Om='11'\t%d=''\t%e=''\t%Od=''\t%Oe=''\n"),
        fmt,
        std::chrono::year_month_day{std::chrono::year{1970}, std::chrono::November, std::chrono::day{100}});
  check(SV("%b='Dec'\t%B='December'\t%h='Dec'\t%m='12'\t%Om='12'\t%d=''\t%e=''\t%Od=''\t%Oe=''\n"),
        fmt,
        std::chrono::year_month_day{std::chrono::year{1970}, std::chrono::December, std::chrono::day{255}});
#else // _WIN32
  check(SV("%b='Sep'\t%B='September'\t%h='Sep'\t%m='09'\t%Om='09'\t%d='32'\t%e='32'\t%Od='32'\t%Oe='32'\n"),
        fmt,
        std::chrono::year_month_day{std::chrono::year{1970}, std::chrono::September, std::chrono::day{32}});
  check(SV("%b='Oct'\t%B='October'\t%h='Oct'\t%m='10'\t%Om='10'\t%d='99'\t%e='99'\t%Od='99'\t%Oe='99'\n"),
        fmt,
        std::chrono::year_month_day{std::chrono::year{1970}, std::chrono::October, std::chrono::day{99}});
#  if defined(_AIX)
  check(SV("%b='Nov'\t%B='November'\t%h='Nov'\t%m='11'\t%Om='11'\t%d='00'\t%e=' 0'\t%Od='00'\t%Oe=' 0'\n"),
        fmt,
        std::chrono::year_month_day{std::chrono::year{1970}, std::chrono::November, std::chrono::day{100}});
  check(SV("%b='Dec'\t%B='December'\t%h='Dec'\t%m='12'\t%Om='12'\t%d='55'\t%e='55'\t%Od='55'\t%Oe='55'\n"),
        fmt,
        std::chrono::year_month_day{std::chrono::year{1970}, std::chrono::December, std::chrono::day{255}});
#  else  //  defined(_AIX)
  check(SV("%b='Nov'\t%B='November'\t%h='Nov'\t%m='11'\t%Om='11'\t%d='100'\t%e='100'\t%Od='100'\t%Oe='100'\n"),
        fmt,
        std::chrono::year_month_day{std::chrono::year{1970}, std::chrono::November, std::chrono::day{100}});
  check(SV("%b='Dec'\t%B='December'\t%h='Dec'\t%m='12'\t%Om='12'\t%d='255'\t%e='255'\t%Od='255'\t%Oe='255'\n"),
        fmt,
        std::chrono::year_month_day{std::chrono::year{1970}, std::chrono::December, std::chrono::day{255}});
#  endif //  defined(_AIX)
#endif   // _WIN32

  // Use the global locale (fr_FR)
#if defined(__APPLE__)
  check(SV("%b='jan'\t%B='janvier'\t%h='jan'\t%m='01'\t%Om='01'\t%d='00'\t%e=' 0'\t%Od='00'\t%Oe=' 0'\n"),
        lfmt,
        std::chrono::year_month_day{std::chrono::year{1970}, std::chrono::January, std::chrono::day{0}});
  check(SV("%b='fév'\t%B='février'\t%h='fév'\t%m='02'\t%Om='02'\t%d='01'\t%e=' 1'\t%Od='01'\t%Oe=' 1'\n"),
        lfmt,
        std::chrono::year_month_day{std::chrono::year{1970}, std::chrono::February, std::chrono::day{1}});
  check(SV("%b='mar'\t%B='mars'\t%h='mar'\t%m='03'\t%Om='03'\t%d='09'\t%e=' 9'\t%Od='09'\t%Oe=' 9'\n"),
        lfmt,
        std::chrono::year_month_day{std::chrono::year{1970}, std::chrono::March, std::chrono::day{9}});
  check(SV("%b='avr'\t%B='avril'\t%h='avr'\t%m='04'\t%Om='04'\t%d='10'\t%e='10'\t%Od='10'\t%Oe='10'\n"),
        lfmt,
        std::chrono::year_month_day{std::chrono::year{1970}, std::chrono::April, std::chrono::day{10}});
  check(SV("%b='mai'\t%B='mai'\t%h='mai'\t%m='05'\t%Om='05'\t%d='28'\t%e='28'\t%Od='28'\t%Oe='28'\n"),
        lfmt,
        std::chrono::year_month_day{std::chrono::year{1970}, std::chrono::May, std::chrono::day{28}});
  check(SV("%b='jui'\t%B='juin'\t%h='jui'\t%m='06'\t%Om='06'\t%d='29'\t%e='29'\t%Od='29'\t%Oe='29'\n"),
        lfmt,
        std::chrono::year_month_day{std::chrono::year{1970}, std::chrono::June, std::chrono::day{29}});
  check(SV("%b='jul'\t%B='juillet'\t%h='jul'\t%m='07'\t%Om='07'\t%d='30'\t%e='30'\t%Od='30'\t%Oe='30'\n"),
        lfmt,
        std::chrono::year_month_day{std::chrono::year{1970}, std::chrono::July, std::chrono::day{30}});
  check(SV("%b='aoû'\t%B='août'\t%h='aoû'\t%m='08'\t%Om='08'\t%d='31'\t%e='31'\t%Od='31'\t%Oe='31'\n"),
        lfmt,
        std::chrono::year_month_day{std::chrono::year{1970}, std::chrono::August, std::chrono::day{31}});
  check(SV("%b='sep'\t%B='septembre'\t%h='sep'\t%m='09'\t%Om='09'\t%d='32'\t%e='32'\t%Od='32'\t%Oe='32'\n"),
        lfmt,
        std::chrono::year_month_day{std::chrono::year{1970}, std::chrono::September, std::chrono::day{32}});
  check(SV("%b='oct'\t%B='octobre'\t%h='oct'\t%m='10'\t%Om='10'\t%d='99'\t%e='99'\t%Od='99'\t%Oe='99'\n"),
        lfmt,
        std::chrono::year_month_day{std::chrono::year{1970}, std::chrono::October, std::chrono::day{99}});
  check(SV("%b='nov'\t%B='novembre'\t%h='nov'\t%m='11'\t%Om='11'\t%d='100'\t%e='100'\t%Od='100'\t%Oe='100'\n"),
        lfmt,
        std::chrono::year_month_day{std::chrono::year{1970}, std::chrono::November, std::chrono::day{100}});
  check(SV("%b='déc'\t%B='décembre'\t%h='déc'\t%m='12'\t%Om='12'\t%d='255'\t%e='255'\t%Od='255'\t%Oe='255'\n"),
        lfmt,
        std::chrono::year_month_day{std::chrono::year{1970}, std::chrono::December, std::chrono::day{255}});
#else // defined(__APPLE__)
#  ifdef _WIN32
  check(SV("%b='janv.'\t%B='janvier'\t%h='janv.'\t%m='01'\t%Om='01'\t%d=''\t%e=''\t%Od=''\t%Oe=''\n"),
        lfmt,
        std::chrono::year_month_day{std::chrono::year{1970}, std::chrono::January, std::chrono::day{0}});
#  else
  check(SV("%b='janv.'\t%B='janvier'\t%h='janv.'\t%m='01'\t%Om='01'\t%d='00'\t%e=' 0'\t%Od='00'\t%Oe=' 0'\n"),
        lfmt,
        std::chrono::year_month_day{std::chrono::year{1970}, std::chrono::January, std::chrono::day{0}});
#  endif
  check(SV("%b='févr.'\t%B='février'\t%h='févr.'\t%m='02'\t%Om='02'\t%d='01'\t%e=' 1'\t%Od='01'\t%Oe=' 1'\n"),
        lfmt,
        std::chrono::year_month_day{std::chrono::year{1970}, std::chrono::February, std::chrono::day{1}});
  check(SV("%b='mars'\t%B='mars'\t%h='mars'\t%m='03'\t%Om='03'\t%d='09'\t%e=' 9'\t%Od='09'\t%Oe=' 9'\n"),
        lfmt,
        std::chrono::year_month_day{std::chrono::year{1970}, std::chrono::March, std::chrono::day{9}});
  check(
#  if defined(_WIN32) || defined(_AIX) || defined(__FreeBSD__)
      SV("%b='avr.'\t%B='avril'\t%h='avr.'\t%m='04'\t%Om='04'\t%d='10'\t%e='10'\t%Od='10'\t%Oe='10'\n"),
#  else  // defined(_WIN32) || defined(_AIX) || defined(__FreeBSD__)
      SV("%b='avril'\t%B='avril'\t%h='avril'\t%m='04'\t%Om='04'\t%d='10'\t%e='10'\t%Od='10'\t%Oe='10'\n"),
#  endif // defined(_WIN32) || defined(_AIX) || defined(__FreeBSD__)
      lfmt,
      std::chrono::year_month_day{std::chrono::year{1970}, std::chrono::April, std::chrono::day{10}});
  check(SV("%b='mai'\t%B='mai'\t%h='mai'\t%m='05'\t%Om='05'\t%d='28'\t%e='28'\t%Od='28'\t%Oe='28'\n"),
        lfmt,
        std::chrono::year_month_day{std::chrono::year{1970}, std::chrono::May, std::chrono::day{28}});
  check(SV("%b='juin'\t%B='juin'\t%h='juin'\t%m='06'\t%Om='06'\t%d='29'\t%e='29'\t%Od='29'\t%Oe='29'\n"),
        lfmt,
        std::chrono::year_month_day{std::chrono::year{1970}, std::chrono::June, std::chrono::day{29}});
  check(SV("%b='juil.'\t%B='juillet'\t%h='juil.'\t%m='07'\t%Om='07'\t%d='30'\t%e='30'\t%Od='30'\t%Oe='30'\n"),
        lfmt,
        std::chrono::year_month_day{std::chrono::year{1970}, std::chrono::July, std::chrono::day{30}});
  check(SV("%b='août'\t%B='août'\t%h='août'\t%m='08'\t%Om='08'\t%d='31'\t%e='31'\t%Od='31'\t%Oe='31'\n"),
        lfmt,
        std::chrono::year_month_day{std::chrono::year{1970}, std::chrono::August, std::chrono::day{31}});
#  ifdef _WIN32
  check(SV("%b='sept.'\t%B='septembre'\t%h='sept.'\t%m='09'\t%Om='09'\t%d=''\t%e=''\t%Od=''\t%Oe=''\n"),
        lfmt,
        std::chrono::year_month_day{std::chrono::year{1970}, std::chrono::September, std::chrono::day{32}});
  check(SV("%b='oct.'\t%B='octobre'\t%h='oct.'\t%m='10'\t%Om='10'\t%d=''\t%e=''\t%Od=''\t%Oe=''\n"),
        lfmt,
        std::chrono::year_month_day{std::chrono::year{1970}, std::chrono::October, std::chrono::day{99}});
  check(SV("%b='nov.'\t%B='novembre'\t%h='nov.'\t%m='11'\t%Om='11'\t%d=''\t%e=''\t%Od=''\t%Oe=''\n"),
        lfmt,
        std::chrono::year_month_day{std::chrono::year{1970}, std::chrono::November, std::chrono::day{100}});
  check(SV("%b='déc.'\t%B='décembre'\t%h='déc.'\t%m='12'\t%Om='12'\t%d=''\t%e=''\t%Od=''\t%Oe=''\n"),
        lfmt,
        std::chrono::year_month_day{std::chrono::year{1970}, std::chrono::December, std::chrono::day{255}});
#  else // _WIN32
  check(SV("%b='sept.'\t%B='septembre'\t%h='sept.'\t%m='09'\t%Om='09'\t%d='32'\t%e='32'\t%Od='32'\t%Oe='32'\n"),
        lfmt,
        std::chrono::year_month_day{std::chrono::year{1970}, std::chrono::September, std::chrono::day{32}});
  check(SV("%b='oct.'\t%B='octobre'\t%h='oct.'\t%m='10'\t%Om='10'\t%d='99'\t%e='99'\t%Od='99'\t%Oe='99'\n"),
        lfmt,
        std::chrono::year_month_day{std::chrono::year{1970}, std::chrono::October, std::chrono::day{99}});
#    if defined(_AIX)
  check(SV("%b='nov.'\t%B='novembre'\t%h='nov.'\t%m='11'\t%Om='11'\t%d='00'\t%e=' 0'\t%Od='00'\t%Oe=' 0'\n"),
        lfmt,
        std::chrono::year_month_day{std::chrono::year{1970}, std::chrono::November, std::chrono::day{100}});
  check(SV("%b='déc.'\t%B='décembre'\t%h='déc.'\t%m='12'\t%Om='12'\t%d='55'\t%e='55'\t%Od='55'\t%Oe='55'\n"),
        lfmt,
        std::chrono::year_month_day{std::chrono::year{1970}, std::chrono::December, std::chrono::day{255}});
#    else  //   defined(_AIX)
  check(SV("%b='nov.'\t%B='novembre'\t%h='nov.'\t%m='11'\t%Om='11'\t%d='100'\t%e='100'\t%Od='100'\t%Oe='100'\n"),
        lfmt,
        std::chrono::year_month_day{std::chrono::year{1970}, std::chrono::November, std::chrono::day{100}});
  check(SV("%b='déc.'\t%B='décembre'\t%h='déc.'\t%m='12'\t%Om='12'\t%d='255'\t%e='255'\t%Od='255'\t%Oe='255'\n"),
        lfmt,
        std::chrono::year_month_day{std::chrono::year{1970}, std::chrono::December, std::chrono::day{255}});
#    endif //   defined(_AIX)
#  endif   // _WIN32
#endif     // defined(__APPLE__)

  // Use supplied locale (ja_JP)
#if defined(_WIN32)
  check(loc,
        SV("%b='1'\t%B='1月'\t%h='1'\t%m='01'\t%Om='01'\t%d=''\t%e=''\t%Od=''\t%Oe=''\n"),
        lfmt,
        std::chrono::year_month_day{std::chrono::year{1970}, std::chrono::January, std::chrono::day{0}});
  check(loc,
        SV("%b='2'\t%B='2月'\t%h='2'\t%m='02'\t%Om='02'\t%d='01'\t%e=' 1'\t%Od='01'\t%Oe=' 1'\n"),
        lfmt,
        std::chrono::year_month_day{std::chrono::year{1970}, std::chrono::February, std::chrono::day{1}});
  check(loc,
        SV("%b='3'\t%B='3月'\t%h='3'\t%m='03'\t%Om='03'\t%d='09'\t%e=' 9'\t%Od='09'\t%Oe=' 9'\n"),
        lfmt,
        std::chrono::year_month_day{std::chrono::year{1970}, std::chrono::March, std::chrono::day{9}});
  check(loc,
        SV("%b='4'\t%B='4月'\t%h='4'\t%m='04'\t%Om='04'\t%d='10'\t%e='10'\t%Od='10'\t%Oe='10'\n"),
        lfmt,
        std::chrono::year_month_day{std::chrono::year{1970}, std::chrono::April, std::chrono::day{10}});
  check(loc,
        SV("%b='5'\t%B='5月'\t%h='5'\t%m='05'\t%Om='05'\t%d='28'\t%e='28'\t%Od='28'\t%Oe='28'\n"),
        lfmt,
        std::chrono::year_month_day{std::chrono::year{1970}, std::chrono::May, std::chrono::day{28}});
  check(loc,
        SV("%b='6'\t%B='6月'\t%h='6'\t%m='06'\t%Om='06'\t%d='29'\t%e='29'\t%Od='29'\t%Oe='29'\n"),
        lfmt,
        std::chrono::year_month_day{std::chrono::year{1970}, std::chrono::June, std::chrono::day{29}});
  check(loc,
        SV("%b='7'\t%B='7月'\t%h='7'\t%m='07'\t%Om='07'\t%d='30'\t%e='30'\t%Od='30'\t%Oe='30'\n"),
        lfmt,
        std::chrono::year_month_day{std::chrono::year{1970}, std::chrono::July, std::chrono::day{30}});
  check(loc,
        SV("%b='8'\t%B='8月'\t%h='8'\t%m='08'\t%Om='08'\t%d='31'\t%e='31'\t%Od='31'\t%Oe='31'\n"),
        lfmt,
        std::chrono::year_month_day{std::chrono::year{1970}, std::chrono::August, std::chrono::day{31}});
  check(loc,
        SV("%b='9'\t%B='9月'\t%h='9'\t%m='09'\t%Om='09'\t%d=''\t%e=''\t%Od=''\t%Oe=''\n"),
        lfmt,
        std::chrono::year_month_day{std::chrono::year{1970}, std::chrono::September, std::chrono::day{32}});
  check(loc,
        SV("%b='10'\t%B='10月'\t%h='10'\t%m='10'\t%Om='10'\t%d=''\t%e=''\t%Od=''\t%Oe=''\n"),
        lfmt,
        std::chrono::year_month_day{std::chrono::year{1970}, std::chrono::October, std::chrono::day{99}});
  check(loc,
        SV("%b='11'\t%B='11月'\t%h='11'\t%m='11'\t%Om='11'\t%d=''\t%e=''\t%Od=''\t%Oe=''\n"),
        lfmt,
        std::chrono::year_month_day{std::chrono::year{1970}, std::chrono::November, std::chrono::day{100}});
  check(loc,
        SV("%b='12'\t%B='12月'\t%h='12'\t%m='12'\t%Om='12'\t%d=''\t%e=''\t%Od=''\t%Oe=''\n"),
        lfmt,
        std::chrono::year_month_day{std::chrono::year{1970}, std::chrono::December, std::chrono::day{255}});
#elif defined(_AIX)      // defined(_WIN32)
  check(loc,
        SV("%b='1月'\t%B='1月'\t%h='1月'\t%m='01'\t%Om='01'\t%d='00'\t%e=' 0'\t%Od='00'\t%Oe=' 0'\n"),
        lfmt,
        std::chrono::year_month_day{std::chrono::year{1970}, std::chrono::January, std::chrono::day{0}});
  check(loc,
        SV("%b='2月'\t%B='2月'\t%h='2月'\t%m='02'\t%Om='02'\t%d='01'\t%e=' 1'\t%Od='01'\t%Oe=' 1'\n"),
        lfmt,
        std::chrono::year_month_day{std::chrono::year{1970}, std::chrono::February, std::chrono::day{1}});
  check(loc,
        SV("%b='3月'\t%B='3月'\t%h='3月'\t%m='03'\t%Om='03'\t%d='09'\t%e=' 9'\t%Od='09'\t%Oe=' 9'\n"),
        lfmt,
        std::chrono::year_month_day{std::chrono::year{1970}, std::chrono::March, std::chrono::day{9}});
  check(loc,
        SV("%b='4月'\t%B='4月'\t%h='4月'\t%m='04'\t%Om='04'\t%d='10'\t%e='10'\t%Od='10'\t%Oe='10'\n"),
        lfmt,
        std::chrono::year_month_day{std::chrono::year{1970}, std::chrono::April, std::chrono::day{10}});
  check(loc,
        SV("%b='5月'\t%B='5月'\t%h='5月'\t%m='05'\t%Om='05'\t%d='28'\t%e='28'\t%Od='28'\t%Oe='28'\n"),
        lfmt,
        std::chrono::year_month_day{std::chrono::year{1970}, std::chrono::May, std::chrono::day{28}});
  check(loc,
        SV("%b='6月'\t%B='6月'\t%h='6月'\t%m='06'\t%Om='06'\t%d='29'\t%e='29'\t%Od='29'\t%Oe='29'\n"),
        lfmt,
        std::chrono::year_month_day{std::chrono::year{1970}, std::chrono::June, std::chrono::day{29}});
  check(loc,
        SV("%b='7月'\t%B='7月'\t%h='7月'\t%m='07'\t%Om='07'\t%d='30'\t%e='30'\t%Od='30'\t%Oe='30'\n"),
        lfmt,
        std::chrono::year_month_day{std::chrono::year{1970}, std::chrono::July, std::chrono::day{30}});
  check(loc,
        SV("%b='8月'\t%B='8月'\t%h='8月'\t%m='08'\t%Om='08'\t%d='31'\t%e='31'\t%Od='31'\t%Oe='31'\n"),
        lfmt,
        std::chrono::year_month_day{std::chrono::year{1970}, std::chrono::August, std::chrono::day{31}});
  check(loc,
        SV("%b='9月'\t%B='9月'\t%h='9月'\t%m='09'\t%Om='09'\t%d='32'\t%e='32'\t%Od='32'\t%Oe='32'\n"),
        lfmt,
        std::chrono::year_month_day{std::chrono::year{1970}, std::chrono::September, std::chrono::day{32}});
  check(loc,
        SV("%b='10月'\t%B='10月'\t%h='10月'\t%m='10'\t%Om='10'\t%d='99'\t%e='99'\t%Od='99'\t%Oe='99'\n"),
        lfmt,
        std::chrono::year_month_day{std::chrono::year{1970}, std::chrono::October, std::chrono::day{99}});
  check(loc,
        SV("%b='11月'\t%B='11月'\t%h='11月'\t%m='11'\t%Om='11'\t%d='00'\t%e=' 0'\t%Od='00'\t%Oe=' 0'\n"),
        lfmt,
        std::chrono::year_month_day{std::chrono::year{1970}, std::chrono::November, std::chrono::day{100}});
  check(loc,
        SV("%b='12月'\t%B='12月'\t%h='12月'\t%m='12'\t%Om='12'\t%d='55'\t%e='55'\t%Od='55'\t%Oe='55'\n"),
        lfmt,
        std::chrono::year_month_day{std::chrono::year{1970}, std::chrono::December, std::chrono::day{255}});
#elif defined(__FreeBSD__) // defined(_WIN32)
  check(loc,
        SV("%b=' 1月'\t%B='1月'\t%h=' 1月'\t%m='01'\t%Om='01'\t%d='00'\t%e=' 0'\t%Od='00'\t%Oe=' 0'\n"),
        lfmt,
        std::chrono::year_month_day{std::chrono::year{1970}, std::chrono::January, std::chrono::day{0}});
  check(loc,
        SV("%b=' 2月'\t%B='2月'\t%h=' 2月'\t%m='02'\t%Om='02'\t%d='01'\t%e=' 1'\t%Od='01'\t%Oe=' 1'\n"),
        lfmt,
        std::chrono::year_month_day{std::chrono::year{1970}, std::chrono::February, std::chrono::day{1}});
  check(loc,
        SV("%b=' 3月'\t%B='3月'\t%h=' 3月'\t%m='03'\t%Om='03'\t%d='09'\t%e=' 9'\t%Od='09'\t%Oe=' 9'\n"),
        lfmt,
        std::chrono::year_month_day{std::chrono::year{1970}, std::chrono::March, std::chrono::day{9}});
  check(loc,
        SV("%b=' 4月'\t%B='4月'\t%h=' 4月'\t%m='04'\t%Om='04'\t%d='10'\t%e='10'\t%Od='10'\t%Oe='10'\n"),
        lfmt,
        std::chrono::year_month_day{std::chrono::year{1970}, std::chrono::April, std::chrono::day{10}});
  check(loc,
        SV("%b=' 5月'\t%B='5月'\t%h=' 5月'\t%m='05'\t%Om='05'\t%d='28'\t%e='28'\t%Od='28'\t%Oe='28'\n"),
        lfmt,
        std::chrono::year_month_day{std::chrono::year{1970}, std::chrono::May, std::chrono::day{28}});
  check(loc,
        SV("%b=' 6月'\t%B='6月'\t%h=' 6月'\t%m='06'\t%Om='06'\t%d='29'\t%e='29'\t%Od='29'\t%Oe='29'\n"),
        lfmt,
        std::chrono::year_month_day{std::chrono::year{1970}, std::chrono::June, std::chrono::day{29}});
  check(loc,
        SV("%b=' 7月'\t%B='7月'\t%h=' 7月'\t%m='07'\t%Om='07'\t%d='30'\t%e='30'\t%Od='30'\t%Oe='30'\n"),
        lfmt,
        std::chrono::year_month_day{std::chrono::year{1970}, std::chrono::July, std::chrono::day{30}});
  check(loc,
        SV("%b=' 8月'\t%B='8月'\t%h=' 8月'\t%m='08'\t%Om='08'\t%d='31'\t%e='31'\t%Od='31'\t%Oe='31'\n"),
        lfmt,
        std::chrono::year_month_day{std::chrono::year{1970}, std::chrono::August, std::chrono::day{31}});
  check(loc,
        SV("%b=' 9月'\t%B='9月'\t%h=' 9月'\t%m='09'\t%Om='09'\t%d='32'\t%e='32'\t%Od='32'\t%Oe='32'\n"),
        lfmt,
        std::chrono::year_month_day{std::chrono::year{1970}, std::chrono::September, std::chrono::day{32}});
  check(loc,
        SV("%b='10月'\t%B='10月'\t%h='10月'\t%m='10'\t%Om='10'\t%d='99'\t%e='99'\t%Od='99'\t%Oe='99'\n"),
        lfmt,
        std::chrono::year_month_day{std::chrono::year{1970}, std::chrono::October, std::chrono::day{99}});
  check(loc,
        SV("%b='11月'\t%B='11月'\t%h='11月'\t%m='11'\t%Om='11'\t%d='100'\t%e='100'\t%Od='100'\t%Oe='100'\n"),
        lfmt,
        std::chrono::year_month_day{std::chrono::year{1970}, std::chrono::November, std::chrono::day{100}});
  check(loc,
        SV("%b='12月'\t%B='12月'\t%h='12月'\t%m='12'\t%Om='12'\t%d='255'\t%e='255'\t%Od='255'\t%Oe='255'\n"),
        lfmt,
        std::chrono::year_month_day{std::chrono::year{1970}, std::chrono::December, std::chrono::day{255}});
#elif defined(__APPLE__) // defined(_WIN32)
  check(loc,
        SV("%b=' 1'\t%B='1月'\t%h=' 1'\t%m='01'\t%Om='01'\t%d='00'\t%e=' 0'\t%Od='00'\t%Oe=' 0'\n"),
        lfmt,
        std::chrono::year_month_day{std::chrono::year{1970}, std::chrono::January, std::chrono::day{0}});
  check(loc,
        SV("%b=' 2'\t%B='2月'\t%h=' 2'\t%m='02'\t%Om='02'\t%d='01'\t%e=' 1'\t%Od='01'\t%Oe=' 1'\n"),
        lfmt,
        std::chrono::year_month_day{std::chrono::year{1970}, std::chrono::February, std::chrono::day{1}});
  check(loc,
        SV("%b=' 3'\t%B='3月'\t%h=' 3'\t%m='03'\t%Om='03'\t%d='09'\t%e=' 9'\t%Od='09'\t%Oe=' 9'\n"),
        lfmt,
        std::chrono::year_month_day{std::chrono::year{1970}, std::chrono::March, std::chrono::day{9}});
  check(loc,
        SV("%b=' 4'\t%B='4月'\t%h=' 4'\t%m='04'\t%Om='04'\t%d='10'\t%e='10'\t%Od='10'\t%Oe='10'\n"),
        lfmt,
        std::chrono::year_month_day{std::chrono::year{1970}, std::chrono::April, std::chrono::day{10}});
  check(loc,
        SV("%b=' 5'\t%B='5月'\t%h=' 5'\t%m='05'\t%Om='05'\t%d='28'\t%e='28'\t%Od='28'\t%Oe='28'\n"),
        lfmt,
        std::chrono::year_month_day{std::chrono::year{1970}, std::chrono::May, std::chrono::day{28}});
  check(loc,
        SV("%b=' 6'\t%B='6月'\t%h=' 6'\t%m='06'\t%Om='06'\t%d='29'\t%e='29'\t%Od='29'\t%Oe='29'\n"),
        lfmt,
        std::chrono::year_month_day{std::chrono::year{1970}, std::chrono::June, std::chrono::day{29}});
  check(loc,
        SV("%b=' 7'\t%B='7月'\t%h=' 7'\t%m='07'\t%Om='07'\t%d='30'\t%e='30'\t%Od='30'\t%Oe='30'\n"),
        lfmt,
        std::chrono::year_month_day{std::chrono::year{1970}, std::chrono::July, std::chrono::day{30}});
  check(loc,
        SV("%b=' 8'\t%B='8月'\t%h=' 8'\t%m='08'\t%Om='08'\t%d='31'\t%e='31'\t%Od='31'\t%Oe='31'\n"),
        lfmt,
        std::chrono::year_month_day{std::chrono::year{1970}, std::chrono::August, std::chrono::day{31}});
  check(loc,
        SV("%b=' 9'\t%B='9月'\t%h=' 9'\t%m='09'\t%Om='09'\t%d='32'\t%e='32'\t%Od='32'\t%Oe='32'\n"),
        lfmt,
        std::chrono::year_month_day{std::chrono::year{1970}, std::chrono::September, std::chrono::day{32}});
  check(loc,
        SV("%b='10'\t%B='10月'\t%h='10'\t%m='10'\t%Om='10'\t%d='99'\t%e='99'\t%Od='99'\t%Oe='99'\n"),
        lfmt,
        std::chrono::year_month_day{std::chrono::year{1970}, std::chrono::October, std::chrono::day{99}});
  check(loc,
        SV("%b='11'\t%B='11月'\t%h='11'\t%m='11'\t%Om='11'\t%d='100'\t%e='100'\t%Od='100'\t%Oe='100'\n"),
        lfmt,
        std::chrono::year_month_day{std::chrono::year{1970}, std::chrono::November, std::chrono::day{100}});
  check(loc,
        SV("%b='12'\t%B='12月'\t%h='12'\t%m='12'\t%Om='12'\t%d='255'\t%e='255'\t%Od='255'\t%Oe='255'\n"),
        lfmt,
        std::chrono::year_month_day{std::chrono::year{1970}, std::chrono::December, std::chrono::day{255}});
#else                    // defined(_WIN32)
  check(loc,
        SV("%b=' 1月'\t%B='1月'\t%h=' 1月'\t%m='01'\t%Om='一'\t%d='00'\t%e=' 0'\t%Od='〇'\t%Oe='〇'\n"),
        lfmt,
        std::chrono::year_month_day{std::chrono::year{1970}, std::chrono::January, std::chrono::day{0}});
  check(loc,
        SV("%b=' 2月'\t%B='2月'\t%h=' 2月'\t%m='02'\t%Om='二'\t%d='01'\t%e=' 1'\t%Od='一'\t%Oe='一'\n"),
        lfmt,
        std::chrono::year_month_day{std::chrono::year{1970}, std::chrono::February, std::chrono::day{1}});
  check(loc,
        SV("%b=' 3月'\t%B='3月'\t%h=' 3月'\t%m='03'\t%Om='三'\t%d='09'\t%e=' 9'\t%Od='九'\t%Oe='九'\n"),
        lfmt,
        std::chrono::year_month_day{std::chrono::year{1970}, std::chrono::March, std::chrono::day{9}});
  check(loc,
        SV("%b=' 4月'\t%B='4月'\t%h=' 4月'\t%m='04'\t%Om='四'\t%d='10'\t%e='10'\t%Od='十'\t%Oe='十'\n"),
        lfmt,
        std::chrono::year_month_day{std::chrono::year{1970}, std::chrono::April, std::chrono::day{10}});
  check(loc,
        SV("%b=' 5月'\t%B='5月'\t%h=' 5月'\t%m='05'\t%Om='五'\t%d='28'\t%e='28'\t%Od='二十八'\t%Oe='二十八'\n"),
        lfmt,
        std::chrono::year_month_day{std::chrono::year{1970}, std::chrono::May, std::chrono::day{28}});
  check(loc,
        SV("%b=' 6月'\t%B='6月'\t%h=' 6月'\t%m='06'\t%Om='六'\t%d='29'\t%e='29'\t%Od='二十九'\t%Oe='二十九'\n"),
        lfmt,
        std::chrono::year_month_day{std::chrono::year{1970}, std::chrono::June, std::chrono::day{29}});
  check(loc,
        SV("%b=' 7月'\t%B='7月'\t%h=' 7月'\t%m='07'\t%Om='七'\t%d='30'\t%e='30'\t%Od='三十'\t%Oe='三十'\n"),
        lfmt,
        std::chrono::year_month_day{std::chrono::year{1970}, std::chrono::July, std::chrono::day{30}});
  check(loc,
        SV("%b=' 8月'\t%B='8月'\t%h=' 8月'\t%m='08'\t%Om='八'\t%d='31'\t%e='31'\t%Od='三十一'\t%Oe='三十一'\n"),
        lfmt,
        std::chrono::year_month_day{std::chrono::year{1970}, std::chrono::August, std::chrono::day{31}});
  check(loc,
        SV("%b=' 9月'\t%B='9月'\t%h=' 9月'\t%m='09'\t%Om='九'\t%d='32'\t%e='32'\t%Od='三十二'\t%Oe='三十二'\n"),
        lfmt,
        std::chrono::year_month_day{std::chrono::year{1970}, std::chrono::September, std::chrono::day{32}});
  check(loc,
        SV("%b='10月'\t%B='10月'\t%h='10月'\t%m='10'\t%Om='十'\t%d='99'\t%e='99'\t%Od='九十九'\t%Oe='九十九'\n"),
        lfmt,
        std::chrono::year_month_day{std::chrono::year{1970}, std::chrono::October, std::chrono::day{99}});
  check(loc,
        SV("%b='11月'\t%B='11月'\t%h='11月'\t%m='11'\t%Om='十一'\t%d='100'\t%e='100'\t%Od='100'\t%Oe='100'\n"),
        lfmt,
        std::chrono::year_month_day{std::chrono::year{1970}, std::chrono::November, std::chrono::day{100}});
  check(loc,
        SV("%b='12月'\t%B='12月'\t%h='12月'\t%m='12'\t%Om='十二'\t%d='255'\t%e='255'\t%Od='255'\t%Oe='255'\n"),
        lfmt,
        std::chrono::year_month_day{std::chrono::year{1970}, std::chrono::December, std::chrono::day{255}});
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
         "%D='01/01/70'\t"
         "%F='1970-01-01'\t"
         "%j='001'\t"
         "%g='70'\t"
         "%G='1970'\t"
         "%u='4'\t"
         "%U='00'\t"
         "%V='01'\t"
         "%w='4'\t"
         "%W='00'\t"
         "%x='01/01/70'\t"
         "%y='70'\t"
         "%Y='1970'\t"
         "%Ex='01/01/70'\t"
         "%EC='19'\t"
         "%Ey='70'\t"
         "%EY='1970'\t"
         "%Ou='4'\t"
         "%OU='00'\t"
         "%OV='01'\t"
         "%Ow='4'\t"
         "%OW='00'\t"
         "%Oy='70'\t"
         "\n"),
      fmt,
      std::chrono::year_month_day{std::chrono::year{1970}, std::chrono::January, std::chrono::day{1}});

  check(
      SV("%C='20'\t"
         "%D='05/29/04'\t"
         "%F='2004-05-29'\t"
         "%j='150'\t"
         "%g='04'\t"
         "%G='2004'\t"
         "%u='6'\t"
         "%U='21'\t"
         "%V='22'\t"
         "%w='6'\t"
         "%W='21'\t"
         "%x='05/29/04'\t"
         "%y='04'\t"
         "%Y='2004'\t"
         "%Ex='05/29/04'\t"
         "%EC='20'\t"
         "%Ey='04'\t"
         "%EY='2004'\t"
         "%Ou='6'\t"
         "%OU='21'\t"
         "%OV='22'\t"
         "%Ow='6'\t"
         "%OW='21'\t"
         "%Oy='04'\t"
         "\n"),
      fmt,
      std::chrono::year_month_day{std::chrono::year{2004}, std::chrono::May, std::chrono::day{29}});

  // Use the global locale (fr_FR)
  check(
      SV("%C='19'\t"
         "%D='01/01/70'\t"
         "%F='1970-01-01'\t"
         "%j='001'\t"
         "%g='70'\t"
         "%G='1970'\t"
         "%u='4'\t"
         "%U='00'\t"
         "%V='01'\t"
         "%w='4'\t"
         "%W='00'\t"
#if defined(__APPLE__) || defined(__FreeBSD__)
         "%x='01.01.1970'\t"
#else
         "%x='01/01/1970'\t"
#endif
         "%y='70'\t"
         "%Y='1970'\t"
#if defined(__APPLE__) || defined(__FreeBSD__)
         "%Ex='01.01.1970'\t"
#else
         "%Ex='01/01/1970'\t"
#endif
         "%EC='19'\t"
         "%Ey='70'\t"
         "%EY='1970'\t"
         "%Ou='4'\t"
         "%OU='00'\t"
         "%OV='01'\t"
         "%Ow='4'\t"
         "%OW='00'\t"
         "%Oy='70'\t"
         "\n"),
      lfmt,
      std::chrono::year_month_day{std::chrono::year{1970}, std::chrono::January, std::chrono::day{1}});

  check(
      SV("%C='20'\t"
         "%D='05/29/04'\t"
         "%F='2004-05-29'\t"
         "%j='150'\t"
         "%g='04'\t"
         "%G='2004'\t"
         "%u='6'\t"
         "%U='21'\t"
         "%V='22'\t"
         "%w='6'\t"
         "%W='21'\t"
#if defined(__APPLE__) || defined(__FreeBSD__)
         "%x='29.05.2004'\t"
#else
         "%x='29/05/2004'\t"
#endif
         "%y='04'\t"
         "%Y='2004'\t"
#if defined(__APPLE__) || defined(__FreeBSD__)
         "%Ex='29.05.2004'\t"
#else
         "%Ex='29/05/2004'\t"
#endif
         "%EC='20'\t"
         "%Ey='04'\t"
         "%EY='2004'\t"
         "%Ou='6'\t"
         "%OU='21'\t"
         "%OV='22'\t"
         "%Ow='6'\t"
         "%OW='21'\t"
         "%Oy='04'\t"
         "\n"),
      lfmt,
      std::chrono::year_month_day{std::chrono::year{2004}, std::chrono::May, std::chrono::day{29}});

  // Use supplied locale (ja_JP)
  check(
      loc,
      SV("%C='19'\t"
         "%D='01/01/70'\t"
         "%F='1970-01-01'\t"
         "%j='001'\t"
         "%g='70'\t"
         "%G='1970'\t"
         "%u='4'\t"
         "%U='00'\t"
         "%V='01'\t"
         "%w='4'\t"
         "%W='00'\t"
#if defined(__APPLE__) || defined(_AIX) || defined(_WIN32) || defined(__FreeBSD__)
         "%x='1970/01/01'\t"
#else  // defined(__APPLE__) || defined(_AIX) || defined(_WIN32) || defined(__FreeBSD__)
         "%x='1970年01月01日'\t"
#endif // defined(__APPLE__) || defined(_AIX) || defined(_WIN32) || defined(__FreeBSD__)
         "%y='70'\t"
         "%Y='1970'\t"
#if defined(__APPLE__) || defined(_AIX) || defined(_WIN32) || defined(__FreeBSD__)
         "%Ex='1970/01/01'\t"
         "%EC='19'\t"
         "%Ey='70'\t"
         "%EY='1970'\t"
         "%Ou='4'\t"
         "%OU='00'\t"
         "%OV='01'\t"
         "%Ow='4'\t"
         "%OW='00'\t"
         "%Oy='70'\t"
#else  // defined(__APPLE__) || defined(_AIX) || defined(_WIN32) || defined(__FreeBSD__)
         "%Ex='昭和45年01月01日'\t"
         "%EC='昭和'\t"
         "%Ey='45'\t"
         "%EY='昭和45年'\t"
         "%Ou='四'\t"
         "%OU='〇'\t"
         "%OV='一'\t"
         "%Ow='四'\t"
         "%OW='〇'\t"
         "%Oy='七十'\t"
#endif // defined(__APPLE__) || defined(_AIX) || defined(_WIN32) || defined(__FreeBSD__)
         "\n"),
      lfmt,
      std::chrono::year_month_day{std::chrono::year{1970}, std::chrono::January, std::chrono::day{1}});

  check(
      loc,
      SV("%C='20'\t"
         "%D='05/29/04'\t"
         "%F='2004-05-29'\t"
         "%j='150'\t"
         "%g='04'\t"
         "%G='2004'\t"
         "%u='6'\t"
         "%U='21'\t"
         "%V='22'\t"
         "%w='6'\t"
         "%W='21'\t"
#if defined(__APPLE__) || defined(_AIX) || defined(_WIN32) || defined(__FreeBSD__)
         "%x='2004/05/29'\t"
#else  // defined(__APPLE__) || defined(_AIX) || defined(_WIN32) || defined(__FreeBSD__)
         "%x='2004年05月29日'\t"
#endif // defined(__APPLE__) || defined(_AIX) || defined(_WIN32) || defined(__FreeBSD__)
         "%y='04'\t"
         "%Y='2004'\t"
#if defined(__APPLE__) || defined(_AIX) || defined(_WIN32) || defined(__FreeBSD__)
         "%Ex='2004/05/29'\t"
         "%EC='20'\t"
         "%Ey='04'\t"
         "%EY='2004'\t"
         "%Ou='6'\t"
         "%OU='21'\t"
         "%OV='22'\t"
         "%Ow='6'\t"
         "%OW='21'\t"
         "%Oy='04'\t"
#else  // defined(__APPLE__) || defined(_AIX) || defined(_WIN32) || defined(__FreeBSD__)
         "%Ex='平成16年05月29日'\t"
         "%EC='平成'\t"
         "%Ey='16'\t"
         "%EY='平成16年'\t"
         "%Ou='六'\t"
         "%OU='二十一'\t"
         "%OV='二十二'\t"
         "%Ow='六'\t"
         "%OW='二十一'\t"
         "%Oy='四'\t"
#endif // defined(__APPLE__) || defined(_AIX) || defined(_WIN32) || defined(__FreeBSD__)
         "\n"),
      lfmt,
      std::chrono::year_month_day{std::chrono::year{2004}, std::chrono::May, std::chrono::day{29}});

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
      std::chrono::year_month_day{std::chrono::year{1970}, std::chrono::January, std::chrono::day{31}});

  check_exception("The format specifier expects a '%' or a '}'",
                  SV("{:A"),
                  std::chrono::year_month_day{std::chrono::year{1970}, std::chrono::January, std::chrono::day{31}});
  check_exception("The chrono specifiers contain a '{'",
                  SV("{:%%{"),
                  std::chrono::year_month_day{std::chrono::year{1970}, std::chrono::January, std::chrono::day{31}});
  check_exception("End of input while parsing a conversion specifier",
                  SV("{:%"),
                  std::chrono::year_month_day{std::chrono::year{1970}, std::chrono::January, std::chrono::day{31}});
  check_exception("End of input while parsing the modifier E",
                  SV("{:%E"),
                  std::chrono::year_month_day{std::chrono::year{1970}, std::chrono::January, std::chrono::day{31}});
  check_exception("End of input while parsing the modifier O",
                  SV("{:%O"),
                  std::chrono::year_month_day{std::chrono::year{1970}, std::chrono::January, std::chrono::day{31}});

  // Precision not allowed
  check_exception("The format specifier expects a '%' or a '}'",
                  SV("{:.3}"),
                  std::chrono::year_month_day{std::chrono::year{1970}, std::chrono::January, std::chrono::day{31}});
}

int main(int, char**) {
  test<char>();

#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  test<wchar_t>();
#endif

  return 0;
}
