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

// template<class charT> struct formatter<chrono::year_month_weekday, charT>;

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
  // Valid year, valid month, valid day, valid index
  check(SV("1970/Jan/Mon[1]"),
        SV("{}"),
        std::chrono::year_month_weekday{
            std::chrono::year{1970}, std::chrono::month{1}, std::chrono::weekday_indexed{std::chrono::weekday{1}, 1}});
  check(SV("*1970/Jan/Mon[1]"),
        SV("{:*>16}"),
        std::chrono::year_month_weekday{
            std::chrono::year{1970}, std::chrono::month{1}, std::chrono::weekday_indexed{std::chrono::weekday{1}, 1}});
  check(SV("*1970/Jan/Mon[1]*"),
        SV("{:*^17}"),
        std::chrono::year_month_weekday{
            std::chrono::year{1970}, std::chrono::month{1}, std::chrono::weekday_indexed{std::chrono::weekday{1}, 1}});

  // Valid year, valid month, valid day, invalid index
  check(SV("1970/Jan/Mon[7 is not a valid index]"),
        SV("{}"),
        std::chrono::year_month_weekday{
            std::chrono::year{1970}, std::chrono::month{1}, std::chrono::weekday_indexed{std::chrono::weekday{1}, 7}});

  // Valid year, valid month, invalid day, valid index
  check(SV("1970/Jan/13 is not a valid weekday[1]"),
        SV("{}"),
        std::chrono::year_month_weekday{
            std::chrono::year{1970}, std::chrono::month{1}, std::chrono::weekday_indexed{std::chrono::weekday{13}, 1}});

  // Valid year, valid month, invalid day, invalid index
  check(SV("1970/Jan/13 is not a valid weekday[7 is not a valid index]"),
        SV("{}"),
        std::chrono::year_month_weekday{
            std::chrono::year{1970}, std::chrono::month{1}, std::chrono::weekday_indexed{std::chrono::weekday{13}, 7}});

  // Valid year, invalid month, valid day, invalid index
  check(SV("1970/0 is not a valid month/Mon[7 is not a valid index]"),
        SV("{}"),
        std::chrono::year_month_weekday{
            std::chrono::year{1970}, std::chrono::month{0}, std::chrono::weekday_indexed{std::chrono::weekday{1}, 7}});

  // Valid year, invalid month, invalid day, valid index
  check(SV("1970/0 is not a valid month/13 is not a valid weekday[1]"),
        SV("{}"),
        std::chrono::year_month_weekday{
            std::chrono::year{1970}, std::chrono::month{0}, std::chrono::weekday_indexed{std::chrono::weekday{13}, 1}});

  // Valid year, invalid month, invalid day, invalid index
  check(SV("1970/0 is not a valid month/13 is not a valid weekday[7 is not a valid index]"),
        SV("{}"),
        std::chrono::year_month_weekday{
            std::chrono::year{1970}, std::chrono::month{0}, std::chrono::weekday_indexed{std::chrono::weekday{13}, 7}});

  // Invalid year, valid month, valid day, valid index
  check(
      SV("-32768 is not a valid year/Jan/Mon[1]"),
      SV("{}"),
      std::chrono::year_month_weekday{
          std::chrono::year{-32768}, std::chrono::month{1}, std::chrono::weekday_indexed{std::chrono::weekday{1}, 1}});

  // Invalid year, valid month, valid day, invalid index
  check(
      SV("-32768 is not a valid year/Jan/Mon[7 is not a valid index]"),
      SV("{}"),
      std::chrono::year_month_weekday{
          std::chrono::year{-32768}, std::chrono::month{1}, std::chrono::weekday_indexed{std::chrono::weekday{1}, 7}});

  // Invalid year, valid month, invalid day, valid index
  check(
      SV("-32768 is not a valid year/Jan/13 is not a valid weekday[1]"),
      SV("{}"),
      std::chrono::year_month_weekday{
          std::chrono::year{-32768}, std::chrono::month{1}, std::chrono::weekday_indexed{std::chrono::weekday{13}, 1}});

  // Invalid year, valid month, invalid day, invalid index
  check(
      SV("-32768 is not a valid year/Jan/13 is not a valid weekday[7 is not a valid index]"),
      SV("{}"),
      std::chrono::year_month_weekday{
          std::chrono::year{-32768}, std::chrono::month{1}, std::chrono::weekday_indexed{std::chrono::weekday{13}, 7}});

  // Invalid year, invalid month, valid day, invalid index
  check(
      SV("-32768 is not a valid year/0 is not a valid month/Mon[7 is not a valid index]"),
      SV("{}"),
      std::chrono::year_month_weekday{
          std::chrono::year{-32768}, std::chrono::month{0}, std::chrono::weekday_indexed{std::chrono::weekday{1}, 7}});

  // Invalid year, invalid month, invalid day, valid index
  check(
      SV("-32768 is not a valid year/0 is not a valid month/13 is not a valid weekday[1]"),
      SV("{}"),
      std::chrono::year_month_weekday{
          std::chrono::year{-32768}, std::chrono::month{0}, std::chrono::weekday_indexed{std::chrono::weekday{13}, 1}});

  // Invalid year, invalid month, invalid day, invalid index
  check(
      SV("-32768 is not a valid year/0 is not a valid month/13 is not a valid weekday[7 is not a valid index]"),
      SV("{}"),
      std::chrono::year_month_weekday{
          std::chrono::year{-32768}, std::chrono::month{0}, std::chrono::weekday_indexed{std::chrono::weekday{13}, 7}});
}

// TODO FMT Should x throw?
template <class CharT>
static void test_invalid_values() {
  // Test that %a, %A, %b, %B, %h, %j, %u, %U, %V, %w, %W, %Ou, %OU, %OV, %Ow, and %OW throw an exception.
  check_exception(
      "formatting a weekday name needs a valid weekday",
      SV("{:%a}"),
      std::chrono::year_month_weekday{
          std::chrono::year{1970}, std::chrono::month{1}, std::chrono::weekday_indexed{std::chrono::weekday{13}, 1}});

  check_exception(
      "formatting a weekday name needs a valid weekday",
      SV("{:%A}"),
      std::chrono::year_month_weekday{
          std::chrono::year{1970}, std::chrono::month{1}, std::chrono::weekday_indexed{std::chrono::weekday{13}, 1}});

  check_exception(
      "formatting a month name from an invalid month number",
      SV("{:%b}"),
      std::chrono::year_month_weekday{
          std::chrono::year{1970}, std::chrono::month{0}, std::chrono::weekday_indexed{std::chrono::weekday{1}, 1}});

  check_exception(
      "formatting a month name from an invalid month number",
      SV("{:%B}"),
      std::chrono::year_month_weekday{
          std::chrono::year{1970}, std::chrono::month{0}, std::chrono::weekday_indexed{std::chrono::weekday{1}, 1}});

  check_exception(
      "formatting a month name from an invalid month number",
      SV("{:%h}"),
      std::chrono::year_month_weekday{
          std::chrono::year{1970}, std::chrono::month{0}, std::chrono::weekday_indexed{std::chrono::weekday{1}, 1}});

  check_exception(
      "formatting a day of year needs a valid date",
      SV("{:%j}"),
      std::chrono::year_month_weekday{
          std::chrono::year{1970}, std::chrono::month{1}, std::chrono::weekday_indexed{std::chrono::weekday{1}, 7}});
  check_exception(
      "formatting a day of year needs a valid date",
      SV("{:%j}"),
      std::chrono::year_month_weekday{
          std::chrono::year{1970}, std::chrono::month{1}, std::chrono::weekday_indexed{std::chrono::weekday{13}, 1}});
  check_exception(
      "formatting a day of year needs a valid date",
      SV("{:%j}"),
      std::chrono::year_month_weekday{
          std::chrono::year{1970}, std::chrono::month{0}, std::chrono::weekday_indexed{std::chrono::weekday{1}, 1}});
  check_exception(
      "formatting a day of year needs a valid date",
      SV("{:%j}"),
      std::chrono::year_month_weekday{
          std::chrono::year{-32768}, std::chrono::month{1}, std::chrono::weekday_indexed{std::chrono::weekday{1}, 1}});

  check_exception(
      "formatting a weekday needs a valid weekday",
      SV("{:%u}"),
      std::chrono::year_month_weekday{
          std::chrono::year{1970}, std::chrono::month{1}, std::chrono::weekday_indexed{std::chrono::weekday{13}, 1}});

  check_exception(
      "formatting a week of year needs a valid date",
      SV("{:%U}"),
      std::chrono::year_month_weekday{
          std::chrono::year{1970}, std::chrono::month{1}, std::chrono::weekday_indexed{std::chrono::weekday{1}, 7}});
  check_exception(
      "formatting a week of year needs a valid date",
      SV("{:%U}"),
      std::chrono::year_month_weekday{
          std::chrono::year{1970}, std::chrono::month{1}, std::chrono::weekday_indexed{std::chrono::weekday{13}, 1}});
  check_exception(
      "formatting a week of year needs a valid date",
      SV("{:%U}"),
      std::chrono::year_month_weekday{
          std::chrono::year{1970}, std::chrono::month{0}, std::chrono::weekday_indexed{std::chrono::weekday{1}, 1}});
  check_exception(
      "formatting a week of year needs a valid date",
      SV("{:%U}"),
      std::chrono::year_month_weekday{
          std::chrono::year{-32768}, std::chrono::month{1}, std::chrono::weekday_indexed{std::chrono::weekday{1}, 1}});

  check_exception(
      "formatting a week of year needs a valid date",
      SV("{:%V}"),
      std::chrono::year_month_weekday{
          std::chrono::year{1970}, std::chrono::month{1}, std::chrono::weekday_indexed{std::chrono::weekday{1}, 7}});
  check_exception(
      "formatting a week of year needs a valid date",
      SV("{:%V}"),
      std::chrono::year_month_weekday{
          std::chrono::year{1970}, std::chrono::month{1}, std::chrono::weekday_indexed{std::chrono::weekday{13}, 1}});
  check_exception(
      "formatting a week of year needs a valid date",
      SV("{:%V}"),
      std::chrono::year_month_weekday{
          std::chrono::year{1970}, std::chrono::month{0}, std::chrono::weekday_indexed{std::chrono::weekday{1}, 1}});
  check_exception(
      "formatting a week of year needs a valid date",
      SV("{:%V}"),
      std::chrono::year_month_weekday{
          std::chrono::year{-32768}, std::chrono::month{1}, std::chrono::weekday_indexed{std::chrono::weekday{1}, 1}});

  check_exception(
      "formatting a weekday needs a valid weekday",
      SV("{:%w}"),
      std::chrono::year_month_weekday{
          std::chrono::year{1970}, std::chrono::month{1}, std::chrono::weekday_indexed{std::chrono::weekday{13}, 1}});

  check_exception(
      "formatting a week of year needs a valid date",
      SV("{:%W}"),
      std::chrono::year_month_weekday{
          std::chrono::year{1970}, std::chrono::month{1}, std::chrono::weekday_indexed{std::chrono::weekday{1}, 7}});
  check_exception(
      "formatting a week of year needs a valid date",
      SV("{:%W}"),
      std::chrono::year_month_weekday{
          std::chrono::year{1970}, std::chrono::month{1}, std::chrono::weekday_indexed{std::chrono::weekday{13}, 1}});
  check_exception(
      "formatting a week of year needs a valid date",
      SV("{:%W}"),
      std::chrono::year_month_weekday{
          std::chrono::year{1970}, std::chrono::month{0}, std::chrono::weekday_indexed{std::chrono::weekday{1}, 1}});
  check_exception(
      "formatting a week of year needs a valid date",
      SV("{:%W}"),
      std::chrono::year_month_weekday{
          std::chrono::year{-32768}, std::chrono::month{1}, std::chrono::weekday_indexed{std::chrono::weekday{1}, 1}});
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

  check(SV("%b='Jan'\t%B='January'\t%h='Jan'\t%m='01'\t%Om='01'\t%d='01'\t%e=' 1'\t%Od='01'\t%Oe=' 1'\n"),
        fmt,
        std::chrono::year_month_weekday{
            std::chrono::year{1970}, std::chrono::January, std::chrono::weekday_indexed{std::chrono::weekday{4}, 1}});
  check(SV("%b='Dec'\t%B='December'\t%h='Dec'\t%m='12'\t%Om='12'\t%d='20'\t%e='20'\t%Od='20'\t%Oe='20'\n"),
        fmt,
        std::chrono::year_month_weekday{
            std::chrono::year{1970}, std::chrono::December, std::chrono::weekday_indexed{std::chrono::weekday{7}, 3}});

  // Use the global locale (fr_FR)
#if defined(__APPLE__)
  check(SV("%b='jan'\t%B='janvier'\t%h='jan'\t%m='01'\t%Om='01'\t%d='01'\t%e=' 1'\t%Od='01'\t%Oe=' 1'\n"),
        lfmt,
        std::chrono::year_month_weekday{
            std::chrono::year{1970}, std::chrono::January, std::chrono::weekday_indexed{std::chrono::weekday{4}, 1}});

  check(SV("%b='déc'\t%B='décembre'\t%h='déc'\t%m='12'\t%Om='12'\t%d='20'\t%e='20'\t%Od='20'\t%Oe='20'\n"),
        lfmt,
        std::chrono::year_month_weekday{
            std::chrono::year{1970}, std::chrono::December, std::chrono::weekday_indexed{std::chrono::weekday{7}, 3}});

#else  // defined(__APPLE__)
  check(SV("%b='janv.'\t%B='janvier'\t%h='janv.'\t%m='01'\t%Om='01'\t%d='01'\t%e=' 1'\t%Od='01'\t%Oe=' 1'\n"),
        lfmt,
        std::chrono::year_month_weekday{
            std::chrono::year{1970}, std::chrono::January, std::chrono::weekday_indexed{std::chrono::weekday{4}, 1}});
  check(SV("%b='déc.'\t%B='décembre'\t%h='déc.'\t%m='12'\t%Om='12'\t%d='20'\t%e='20'\t%Od='20'\t%Oe='20'\n"),
        lfmt,
        std::chrono::year_month_weekday{
            std::chrono::year{1970}, std::chrono::December, std::chrono::weekday_indexed{std::chrono::weekday{7}, 3}});

#endif // defined(__APPLE__)

  // Use supplied locale (ja_JP)
#if defined(_WIN32)
  check(loc,
        SV("%b='1'\t%B='1月'\t%h='1'\t%m='01'\t%Om='01'\t%d='01'\t%e=' 1'\t%Od='01'\t%Oe=' 1'\n"),
        lfmt,
        std::chrono::year_month_weekday{
            std::chrono::year{1970}, std::chrono::January, std::chrono::weekday_indexed{std::chrono::weekday{4}, 1}});

  check(loc,
        SV("%b='12'\t%B='12月'\t%h='12'\t%m='12'\t%Om='12'\t%d='20'\t%e='20'\t%Od='20'\t%Oe='20'\n"),
        lfmt,
        std::chrono::year_month_weekday{
            std::chrono::year{1970}, std::chrono::December, std::chrono::weekday_indexed{std::chrono::weekday{7}, 3}});

#elif defined(_AIX)      // defined(_WIN32)
  check(loc,
        SV("%b='1月'\t%B='1月'\t%h='1月'\t%m='01'\t%Om='01'\t%d='01'\t%e=' 1'\t%Od='01'\t%Oe=' 1'\n"),
        lfmt,
        std::chrono::year_month_weekday{
            std::chrono::year{1970}, std::chrono::January, std::chrono::weekday_indexed{std::chrono::weekday{4}, 1}});

  check(loc,
        SV("%b='12月'\t%B='12月'\t%h='12月'\t%m='12'\t%Om='12'\t%d='20'\t%e='20'\t%Od='20'\t%Oe='20'\n"),
        lfmt,
        std::chrono::year_month_weekday{
            std::chrono::year{1970}, std::chrono::December, std::chrono::weekday_indexed{std::chrono::weekday{7}, 3}});

#elif defined(__APPLE__) // defined(_WIN32)
  check(loc,
        SV("%b=' 1'\t%B='1月'\t%h=' 1'\t%m='01'\t%Om='01'\t%d='01'\t%e=' 1'\t%Od='01'\t%Oe=' 1'\n"),
        lfmt,
        std::chrono::year_month_weekday{
            std::chrono::year{1970}, std::chrono::January, std::chrono::weekday_indexed{std::chrono::weekday{4}, 1}});

  check(loc,
        SV("%b='12'\t%B='12月'\t%h='12'\t%m='12'\t%Om='12'\t%d='20'\t%e='20'\t%Od='20'\t%Oe='20'\n"),
        lfmt,
        std::chrono::year_month_weekday{
            std::chrono::year{1970}, std::chrono::December, std::chrono::weekday_indexed{std::chrono::weekday{7}, 3}});

#else                    // defined(_WIN32)
  check(loc,
        SV("%b=' 1月'\t%B='1月'\t%h=' 1月'\t%m='01'\t%Om='一'\t%d='01'\t%e=' 1'\t%Od='一'\t%Oe='一'\n"),
        lfmt,
        std::chrono::year_month_weekday{
            std::chrono::year{1970}, std::chrono::January, std::chrono::weekday_indexed{std::chrono::weekday{4}, 1}});

  check(loc,
        SV("%b='12月'\t%B='12月'\t%h='12月'\t%m='12'\t%Om='十二'\t%d='20'\t%e='20'\t%Od='二十'\t%Oe='二十'\n"),
        lfmt,
        std::chrono::year_month_weekday{
            std::chrono::year{1970}, std::chrono::December, std::chrono::weekday_indexed{std::chrono::weekday{7}, 3}});

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
      std::chrono::year_month_weekday{
          std::chrono::year{1970}, std::chrono::January, std::chrono::weekday_indexed{std::chrono::weekday{4}, 1}});

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
      std::chrono::year_month_weekday{
          std::chrono::year{2004}, std::chrono::May, std::chrono::weekday_indexed{std::chrono::weekday{6}, 5}});

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
#if defined(__APPLE__)
         "%x='01.01.1970'\t"
#else
         "%x='01/01/1970'\t"
#endif
         "%y='70'\t"
         "%Y='1970'\t"
#if defined(__APPLE__)
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
      std::chrono::year_month_weekday{
          std::chrono::year{1970}, std::chrono::January, std::chrono::weekday_indexed{std::chrono::weekday{4}, 1}});

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
#if defined(__APPLE__)
         "%x='29.05.2004'\t"
#else
         "%x='29/05/2004'\t"
#endif
         "%y='04'\t"
         "%Y='2004'\t"
#if defined(__APPLE__)
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
      std::chrono::year_month_weekday{
          std::chrono::year{2004}, std::chrono::May, std::chrono::weekday_indexed{std::chrono::weekday{6}, 5}});

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
#if defined(__APPLE__) || defined(_AIX) || defined(_WIN32)
         "%x='1970/01/01'\t"
#else  // defined(__APPLE__) || defined(_AIX) || defined(_WIN32)
         "%x='1970年01月01日'\t"
#endif // defined(__APPLE__) || defined(_AIX) || defined(_WIN32)
         "%y='70'\t"
         "%Y='1970'\t"
#if defined(__APPLE__) || defined(_AIX) || defined(_WIN32)
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
#else  // defined(__APPLE__) || defined(_AIX) || defined(_WIN32)
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
#endif // defined(__APPLE__) || defined(_AIX) || defined(_WIN32)
         "\n"),
      lfmt,
      std::chrono::year_month_weekday{
          std::chrono::year{1970}, std::chrono::January, std::chrono::weekday_indexed{std::chrono::weekday{4}, 1}});

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
#if defined(__APPLE__) || defined(_AIX) || defined(_WIN32)
         "%x='2004/05/29'\t"
#else  // defined(__APPLE__) || defined(_AIX) || defined(_WIN32)
         "%x='2004年05月29日'\t"
#endif // defined(__APPLE__) || defined(_AIX) || defined(_WIN32)
         "%y='04'\t"
         "%Y='2004'\t"
#if defined(__APPLE__) || defined(_AIX) || defined(_WIN32)
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
#else  // defined(__APPLE__) || defined(_AIX) || defined(_WIN32)
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
#endif // defined(__APPLE__) || defined(_AIX) || defined(_WIN32)
         "\n"),
      lfmt,
      std::chrono::year_month_weekday{
          std::chrono::year{2004}, std::chrono::May, std::chrono::weekday_indexed{std::chrono::weekday{6}, 5}});

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
      std::chrono::year_month_weekday{
          std::chrono::year{1970}, std::chrono::January, std::chrono::weekday_indexed{std::chrono::weekday{1}, 1}});

  check_exception(
      "Expected '%' or '}' in the chrono format-string",
      SV("{:A"),
      std::chrono::year_month_weekday{
          std::chrono::year{1970}, std::chrono::January, std::chrono::weekday_indexed{std::chrono::weekday{1}, 1}});
  check_exception(
      "The chrono-specs contains a '{'",
      SV("{:%%{"),
      std::chrono::year_month_weekday{
          std::chrono::year{1970}, std::chrono::January, std::chrono::weekday_indexed{std::chrono::weekday{1}, 1}});
  check_exception(
      "End of input while parsing the modifier chrono conversion-spec",
      SV("{:%"),
      std::chrono::year_month_weekday{
          std::chrono::year{1970}, std::chrono::January, std::chrono::weekday_indexed{std::chrono::weekday{1}, 1}});
  check_exception(
      "End of input while parsing the modifier E",
      SV("{:%E"),
      std::chrono::year_month_weekday{
          std::chrono::year{1970}, std::chrono::January, std::chrono::weekday_indexed{std::chrono::weekday{1}, 1}});
  check_exception(
      "End of input while parsing the modifier O",
      SV("{:%O"),
      std::chrono::year_month_weekday{
          std::chrono::year{1970}, std::chrono::January, std::chrono::weekday_indexed{std::chrono::weekday{1}, 1}});

  // Precision not allowed
  check_exception(
      "Expected '%' or '}' in the chrono format-string",
      SV("{:.3}"),
      std::chrono::year_month_weekday{
          std::chrono::year{1970}, std::chrono::January, std::chrono::weekday_indexed{std::chrono::weekday{1}, 1}});
}

int main(int, char**) {
  test<char>();

#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  test<wchar_t>();
#endif

  return 0;
}
