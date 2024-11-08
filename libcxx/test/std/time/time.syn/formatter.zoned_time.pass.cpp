//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: no-filesystem, no-localization, no-tzdb

// TODO FMT This test should not require std::to_chars(floating-point)
// XFAIL: availability-fp_to_chars-missing

// XFAIL: libcpp-has-no-experimental-tzdb

// REQUIRES: locale.fr_FR.UTF-8
// REQUIRES: locale.ja_JP.UTF-8

// <chrono>
//
// template<class Duration, class TimeZonePtr, class charT>
// struct formatter<chrono::zoned_time<Duration, TimeZonePtr>, charT>

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
#include "test_macros.h"

template <class CharT>
static void test_no_chrono_specs() {
  using namespace std::literals::chrono_literals;

  check(SV("1970-01-01 01:00:00.000000042 +01"),
        SV("{}"),
        std::chrono::zoned_time("Etc/GMT-1", std::chrono::sys_time<std::chrono::nanoseconds>{42ns}));
  check(SV("1970-01-01 01:00:00.000042 +01"),
        SV("{}"),
        std::chrono::zoned_time("Etc/GMT-1", std::chrono::sys_time<std::chrono::microseconds>{42us}));
  check(SV("1970-01-01 01:00:00.042 +01"),
        SV("{}"),
        std::chrono::zoned_time("Etc/GMT-1", std::chrono::sys_time<std::chrono::milliseconds>{42ms}));
  check(SV("1970-01-01 01:00:42 +01"),
        SV("{}"),
        std::chrono::zoned_time("Etc/GMT-1", std::chrono::sys_time<std::chrono::seconds>{42s}));
  check(SV("1970-02-12 01:00:00 +01"),
        SV("{}"),
        std::chrono::zoned_time("Etc/GMT-1", std::chrono::sys_time<std::chrono::days>{std::chrono::days{42}}));
  check(SV("1970-10-22 01:00:00 +01"),
        SV("{}"),
        std::chrono::zoned_time("Etc/GMT-1", std::chrono::sys_time<std::chrono::weeks>{std::chrono::weeks{42}}));
}

template <class CharT>
static void test_valid_values_year() {
  using namespace std::literals::chrono_literals;

  constexpr std::basic_string_view<CharT> fmt =
      SV("{:%%C='%C'%t%%EC='%EC'%t%%y='%y'%t%%Oy='%Oy'%t%%Ey='%Ey'%t%%Y='%Y'%t%%EY='%EY'%n}");
  constexpr std::basic_string_view<CharT> lfmt =
      SV("{:L%%C='%C'%t%%EC='%EC'%t%%y='%y'%t%%Oy='%Oy'%t%%Ey='%Ey'%t%%Y='%Y'%t%%EY='%EY'%n}");

  const std::locale loc(LOCALE_ja_JP_UTF_8);
  std::locale::global(std::locale(LOCALE_fr_FR_UTF_8));

  // Non localized output using C-locale
  check(SV("%C='19'\t%EC='19'\t%y='70'\t%Oy='70'\t%Ey='70'\t%Y='1970'\t%EY='1970'\n"),
        fmt,
        std::chrono::zoned_time(std::chrono::sys_seconds{0s})); // 00:00:00 UTC Thursday, 1 January 1970

  check(SV("%C='20'\t%EC='20'\t%y='09'\t%Oy='09'\t%Ey='09'\t%Y='2009'\t%EY='2009'\n"),
        fmt,
        std::chrono::zoned_time(std::chrono::sys_seconds{1'234'567'890s})); // 23:31:30 UTC on Friday, 13 February 2009

  // Use the global locale (fr_FR)
  check(SV("%C='19'\t%EC='19'\t%y='70'\t%Oy='70'\t%Ey='70'\t%Y='1970'\t%EY='1970'\n"),
        lfmt,
        std::chrono::zoned_time(std::chrono::sys_seconds{0s})); // 00:00:00 UTC Thursday, 1 January 1970

  check(SV("%C='20'\t%EC='20'\t%y='09'\t%Oy='09'\t%Ey='09'\t%Y='2009'\t%EY='2009'\n"),
        lfmt,
        std::chrono::zoned_time(std::chrono::sys_seconds{1'234'567'890s})); // 23:31:30 UTC on Friday, 13 February 2009

  // Use supplied locale (ja_JP). This locale has a different alternate.
#if defined(_WIN32) || defined(__APPLE__) || defined(_AIX) || defined(__FreeBSD__)

  check(loc,
        SV("%C='19'\t%EC='19'\t%y='70'\t%Oy='70'\t%Ey='70'\t%Y='1970'\t%EY='1970'\n"),
        lfmt,
        std::chrono::zoned_time(std::chrono::sys_seconds{0s})); // 00:00:00 UTC Thursday, 1 January 1970

  check(loc,
        SV("%C='20'\t%EC='20'\t%y='09'\t%Oy='09'\t%Ey='09'\t%Y='2009'\t%EY='2009'\n"),
        lfmt,
        std::chrono::zoned_time(std::chrono::sys_seconds{1'234'567'890s})); // 23:31:30 UTC on Friday, 13 February 2009
#else  // defined(_WIN32) || defined(__APPLE__) || defined(_AIX)|| defined(__FreeBSD__)

  check(loc,
        SV("%C='19'\t%EC='昭和'\t%y='70'\t%Oy='七十'\t%Ey='45'\t%Y='1970'\t%EY='昭和45年'\n"),
        lfmt,
        std::chrono::zoned_time(std::chrono::sys_seconds{0s})); // 00:00:00 UTC Thursday, 1 January 1970

  check(loc,
        SV("%C='20'\t%EC='平成'\t%y='09'\t%Oy='九'\t%Ey='21'\t%Y='2009'\t%EY='平成21年'\n"),
        lfmt,
        std::chrono::zoned_time(std::chrono::sys_seconds{1'234'567'890s})); // 23:31:30 UTC on Friday, 13 February 2009
#endif // defined(_WIN32) || defined(__APPLE__) || defined(_AIX)|| defined(__FreeBSD__)

  std::locale::global(std::locale::classic());
}

template <class CharT>
static void test_valid_values_month() {
  using namespace std::literals::chrono_literals;

  constexpr std::basic_string_view<CharT> fmt  = SV("{:%%b='%b'%t%%h='%h'%t%%B='%B'%t%%m='%m'%t%%Om='%Om'%n}");
  constexpr std::basic_string_view<CharT> lfmt = SV("{:L%%b='%b'%t%%h='%h'%t%%B='%B'%t%%m='%m'%t%%Om='%Om'%n}");

  const std::locale loc(LOCALE_ja_JP_UTF_8);
  std::locale::global(std::locale(LOCALE_fr_FR_UTF_8));

  // Non localized output using C-locale
  check(SV("%b='Jan'\t%h='Jan'\t%B='January'\t%m='01'\t%Om='01'\n"),
        fmt,
        std::chrono::zoned_time(std::chrono::sys_seconds{0s})); // 00:00:00 UTC Thursday, 1 January 1970

  check(SV("%b='May'\t%h='May'\t%B='May'\t%m='05'\t%Om='05'\n"),
        fmt,
        std::chrono::zoned_time(std::chrono::sys_seconds{2'000'000'000s})); // 03:33:20 UTC on Wednesday, 18 May 2033

  // Use the global locale (fr_FR)
#if defined(__APPLE__)
  check(SV("%b='jan'\t%h='jan'\t%B='janvier'\t%m='01'\t%Om='01'\n"),
        lfmt,
        std::chrono::zoned_time(std::chrono::sys_seconds{0s})); // 00:00:00 UTC Thursday, 1 January 1970
#else
  check(SV("%b='janv.'\t%h='janv.'\t%B='janvier'\t%m='01'\t%Om='01'\n"),
        lfmt,
        std::chrono::zoned_time(std::chrono::sys_seconds{0s})); // 00:00:00 UTC Thursday, 1 January 1970
#endif

  check(SV("%b='mai'\t%h='mai'\t%B='mai'\t%m='05'\t%Om='05'\n"),
        lfmt,
        std::chrono::zoned_time(std::chrono::sys_seconds{2'000'000'000s})); // 03:33:20 UTC on Wednesday, 18 May 2033

  // Use supplied locale (ja_JP). This locale has a different alternate.
#ifdef _WIN32
  check(loc,
        SV("%b='1'\t%h='1'\t%B='1月'\t%m='01'\t%Om='01'\n"),
        lfmt,
        std::chrono::zoned_time(std::chrono::sys_seconds{0s})); // 00:00:00 UTC Thursday, 1 January 1970

  check(loc,
        SV("%b='5'\t%h='5'\t%B='5月'\t%m='05'\t%Om='05'\n"),
        lfmt,
        std::chrono::zoned_time(std::chrono::sys_seconds{2'000'000'000s})); // 03:33:20 UTC on Wednesday, 18 May 2033
#elif defined(_AIX)                                                         // _WIN32
  check(loc,
        SV("%b='1月'\t%h='1月'\t%B='1月'\t%m='01'\t%Om='01'\n"),
        lfmt,
        std::chrono::zoned_time(std::chrono::sys_seconds{0s})); // 00:00:00 UTC Thursday, 1 January 1970

  check(loc,
        SV("%b='5月'\t%h='5月'\t%B='5月'\t%m='05'\t%Om='05'\n"),
        lfmt,
        std::chrono::zoned_time(std::chrono::sys_seconds{2'000'000'000s})); // 03:33:20 UTC on Wednesday, 18 May 2033
#elif defined(__APPLE__)                                                    // _WIN32
  check(loc,
        SV("%b=' 1'\t%h=' 1'\t%B='1月'\t%m='01'\t%Om='01'\n"),
        lfmt,
        std::chrono::zoned_time(std::chrono::sys_seconds{0s})); // 00:00:00 UTC Thursday, 1 January 1970

  check(loc,
        SV("%b=' 5'\t%h=' 5'\t%B='5月'\t%m='05'\t%Om='05'\n"),
        lfmt,
        std::chrono::zoned_time(std::chrono::sys_seconds{2'000'000'000s})); // 03:33:20 UTC on Wednesday, 18 May 2033
#elif defined(__FreeBSD__)                                                  // _WIN32
  check(loc,
        SV("%b=' 1月'\t%h=' 1月'\t%B='1月'\t%m='01'\t%Om='01'\n"),
        lfmt,
        std::chrono::zoned_time(std::chrono::sys_seconds{0s})); // 00:00:00 UTC Thursday, 1 January 1970

  check(loc,
        SV("%b=' 5月'\t%h=' 5月'\t%B='5月'\t%m='05'\t%Om='05'\n"),
        lfmt,
        std::chrono::zoned_time(std::chrono::sys_seconds{2'000'000'000s})); // 03:33:20 UTC on Wednesday, 18 May 2033
#else                                                                       // _WIN32
  check(loc,
        SV("%b=' 1月'\t%h=' 1月'\t%B='1月'\t%m='01'\t%Om='一'\n"),
        lfmt,
        std::chrono::zoned_time(std::chrono::sys_seconds{0s})); // 00:00:00 UTC Thursday, 1 January 1970

  check(loc,
        SV("%b=' 5月'\t%h=' 5月'\t%B='5月'\t%m='05'\t%Om='五'\n"),
        lfmt,
        std::chrono::zoned_time(std::chrono::sys_seconds{2'000'000'000s})); // 03:33:20 UTC on Wednesday, 18 May 2033
#endif                                                                      // _WIN32

  std::locale::global(std::locale::classic());
}

template <class CharT>
static void test_valid_values_day() {
  using namespace std::literals::chrono_literals;

  constexpr std::basic_string_view<CharT> fmt  = SV("{:%%d='%d'%t%%Od='%Od'%t%%e='%e'%t%%Oe='%Oe'%n}");
  constexpr std::basic_string_view<CharT> lfmt = SV("{:L%%d='%d'%t%%Od='%Od'%t%%e='%e'%t%%Oe='%Oe'%n}");

  const std::locale loc(LOCALE_ja_JP_UTF_8);
  std::locale::global(std::locale(LOCALE_fr_FR_UTF_8));

  // Non localized output using C-locale
  check(SV("%d='01'\t%Od='01'\t%e=' 1'\t%Oe=' 1'\n"),
        fmt,
        std::chrono::zoned_time(std::chrono::sys_seconds{0s})); // 00:00:00 UTC Thursday, 1 January 1970

  check(SV("%d='13'\t%Od='13'\t%e='13'\t%Oe='13'\n"),
        fmt,
        std::chrono::zoned_time(std::chrono::sys_seconds{1'234'567'890s})); // 23:31:30 UTC on Friday, 13 February 2009

  // Use the global locale (fr_FR)
  check(SV("%d='01'\t%Od='01'\t%e=' 1'\t%Oe=' 1'\n"),
        lfmt,
        std::chrono::zoned_time(std::chrono::sys_seconds{0s})); // 00:00:00 UTC Thursday, 1 January 1970

  check(SV("%d='13'\t%Od='13'\t%e='13'\t%Oe='13'\n"),
        lfmt,
        std::chrono::zoned_time(std::chrono::sys_seconds{1'234'567'890s})); // 23:31:30 UTC on Friday, 13 February 2009

  // Use supplied locale (ja_JP). This locale has a different alternate.
#if defined(_WIN32) || defined(__APPLE__) || defined(_AIX) || defined(__FreeBSD__)
  check(loc,
        SV("%d='01'\t%Od='01'\t%e=' 1'\t%Oe=' 1'\n"),
        lfmt,
        std::chrono::zoned_time(std::chrono::sys_seconds{0s})); // 00:00:00 UTC Thursday, 1 January 1970

  check(loc,
        SV("%d='13'\t%Od='13'\t%e='13'\t%Oe='13'\n"),
        lfmt,
        std::chrono::zoned_time(std::chrono::sys_seconds{1'234'567'890s})); // 23:31:30 UTC on Friday, 13 February 2009
#else // defined(_WIN32) || defined(__APPLE__) || defined(_AIX) || defined(__FreeBSD__)
  check(loc,
        SV("%d='01'\t%Od='一'\t%e=' 1'\t%Oe='一'\n"),
        lfmt,
        std::chrono::zoned_time(std::chrono::sys_seconds{0s})); // 00:00:00 UTC Thursday, 1 January 1970

  check(loc,
        SV("%d='13'\t%Od='十三'\t%e='13'\t%Oe='十三'\n"),
        lfmt,
        std::chrono::zoned_time(std::chrono::sys_seconds{1'234'567'890s})); // 23:31:30 UTC on Friday, 13 February 2009

#endif // defined(_WIN32) || defined(__APPLE__) || defined(_AIX) || defined(__FreeBSD__)

  std::locale::global(std::locale::classic());
}

template <class CharT>
static void test_valid_values_weekday() {
  using namespace std::literals::chrono_literals;

  constexpr std::basic_string_view<CharT> fmt =
      SV("{:%%a='%a'%t%%A='%A'%t%%u='%u'%t%%Ou='%Ou'%t%%w='%w'%t%%Ow='%Ow'%n}");
  constexpr std::basic_string_view<CharT> lfmt =
      SV("{:L%%a='%a'%t%%A='%A'%t%%u='%u'%t%%Ou='%Ou'%t%%w='%w'%t%%Ow='%Ow'%n}");

  const std::locale loc(LOCALE_ja_JP_UTF_8);
  std::locale::global(std::locale(LOCALE_fr_FR_UTF_8));

  // Non localized output using C-locale
  check(SV("%a='Thu'\t%A='Thursday'\t%u='4'\t%Ou='4'\t%w='4'\t%Ow='4'\n"),
        fmt,
        std::chrono::zoned_time(std::chrono::sys_seconds{0s})); // 00:00:00 UTC Thursday, 1 January 1970

  check(SV("%a='Sun'\t%A='Sunday'\t%u='7'\t%Ou='7'\t%w='0'\t%Ow='0'\n"),
        fmt,
        std::chrono::zoned_time(std::chrono::sys_seconds{4'294'967'295s})); // 06:28:15 UTC on Sunday, 7 February 2106

  // Use the global locale (fr_FR)
#if defined(__APPLE__)
  check(SV("%a='Jeu'\t%A='Jeudi'\t%u='4'\t%Ou='4'\t%w='4'\t%Ow='4'\n"),
        lfmt,
        std::chrono::zoned_time(std::chrono::sys_seconds{0s})); // 00:00:00 UTC Thursday, 1 January 1970

  check(SV("%a='Dim'\t%A='Dimanche'\t%u='7'\t%Ou='7'\t%w='0'\t%Ow='0'\n"),
        lfmt,
        std::chrono::zoned_time(std::chrono::sys_seconds{4'294'967'295s})); // 06:28:15 UTC on Sunday, 7 February 2106
#else
  check(SV("%a='jeu.'\t%A='jeudi'\t%u='4'\t%Ou='4'\t%w='4'\t%Ow='4'\n"),
        lfmt,
        std::chrono::zoned_time(std::chrono::sys_seconds{0s})); // 00:00:00 UTC Thursday, 1 January 1970

  check(SV("%a='dim.'\t%A='dimanche'\t%u='7'\t%Ou='7'\t%w='0'\t%Ow='0'\n"),
        lfmt,
        std::chrono::zoned_time(std::chrono::sys_seconds{4'294'967'295s})); // 06:28:15 UTC on Sunday, 7 February 2106
#endif

  // Use supplied locale (ja_JP).
  // This locale has a different alternate, but not on all platforms
#if defined(_WIN32) || defined(__APPLE__) || defined(_AIX) || defined(__FreeBSD__)
  check(loc,
        SV("%a='木'\t%A='木曜日'\t%u='4'\t%Ou='4'\t%w='4'\t%Ow='4'\n"),
        lfmt,
        std::chrono::zoned_time(std::chrono::sys_seconds{0s})); // 00:00:00 UTC Thursday, 1 January 1970

  check(loc,
        SV("%a='日'\t%A='日曜日'\t%u='7'\t%Ou='7'\t%w='0'\t%Ow='0'\n"),
        lfmt,
        std::chrono::zoned_time(std::chrono::sys_seconds{4'294'967'295s})); // 06:28:15 UTC on Sunday, 7 February 2106
#else  // defined(_WIN32) || defined(__APPLE__) || defined(_AIX) || defined(__FreeBSD__)
  check(loc,
        SV("%a='木'\t%A='木曜日'\t%u='4'\t%Ou='四'\t%w='4'\t%Ow='四'\n"),
        lfmt,
        std::chrono::zoned_time(std::chrono::sys_seconds{0s})); // 00:00:00 UTC Thursday, 1 January 1970

  check(loc,
        SV("%a='日'\t%A='日曜日'\t%u='7'\t%Ou='七'\t%w='0'\t%Ow='〇'\n"),
        lfmt,
        std::chrono::zoned_time(std::chrono::sys_seconds{4'294'967'295s})); // 06:28:15 UTC on Sunday, 7 February 2106
#endif // defined(_WIN32) || defined(__APPLE__) || defined(_AIX) || defined(__FreeBSD__)

  std::locale::global(std::locale::classic());
}

template <class CharT>
static void test_valid_values_day_of_year() {
  using namespace std::literals::chrono_literals;

  constexpr std::basic_string_view<CharT> fmt  = SV("{:%%j='%j'%n}");
  constexpr std::basic_string_view<CharT> lfmt = SV("{:L%%j='%j'%n}");

  const std::locale loc(LOCALE_ja_JP_UTF_8);
  std::locale::global(std::locale(LOCALE_fr_FR_UTF_8));

  // Non localized output using C-locale
  check(SV("%j='001'\n"),
        fmt,
        std::chrono::zoned_time(std::chrono::sys_seconds{0s})); // 00:00:00 UTC Thursday, 1 January 1970
  check(SV("%j='138'\n"),
        fmt,
        std::chrono::zoned_time(std::chrono::sys_seconds{2'000'000'000s})); // 03:33:20 UTC on Wednesday, 18 May 2033

  // Use the global locale (fr_FR)
  check(SV("%j='001'\n"),
        lfmt,
        std::chrono::zoned_time(std::chrono::sys_seconds{0s})); // 00:00:00 UTC Thursday, 1 January 1970
  check(SV("%j='138'\n"),
        lfmt,
        std::chrono::zoned_time(std::chrono::sys_seconds{2'000'000'000s})); // 03:33:20 UTC on Wednesday, 18 May 2033

  // Use supplied locale (ja_JP). This locale has a different alternate.
  check(loc,
        SV("%j='001'\n"),
        lfmt,
        std::chrono::zoned_time(std::chrono::sys_seconds{0s})); // 00:00:00 UTC Thursday, 1 January 1970

  check(loc,
        SV("%j='138'\n"),
        lfmt,
        std::chrono::zoned_time(std::chrono::sys_seconds{2'000'000'000s})); // 03:33:20 UTC on Wednesday, 18 May 2033

  std::locale::global(std::locale::classic());
}

template <class CharT>
static void test_valid_values_week() {
  using namespace std::literals::chrono_literals;

  constexpr std::basic_string_view<CharT> fmt  = SV("{:%%U='%U'%t%%OU='%OU'%t%%W='%W'%t%%OW='%OW'%n}");
  constexpr std::basic_string_view<CharT> lfmt = SV("{:L%%U='%U'%t%%OU='%OU'%t%%W='%W'%t%%OW='%OW'%n}");

  const std::locale loc(LOCALE_ja_JP_UTF_8);
  std::locale::global(std::locale(LOCALE_fr_FR_UTF_8));

  // Non localized output using C-locale
  check(SV("%U='00'\t%OU='00'\t%W='00'\t%OW='00'\n"),
        fmt,
        std::chrono::zoned_time(std::chrono::sys_seconds{0s})); // 00:00:00 UTC Thursday, 1 January 1970

  check(SV("%U='20'\t%OU='20'\t%W='20'\t%OW='20'\n"),
        fmt,
        std::chrono::zoned_time(std::chrono::sys_seconds{2'000'000'000s})); // 03:33:20 UTC on Wednesday, 18 May 2033

  // Use the global locale (fr_FR)
  check(SV("%U='00'\t%OU='00'\t%W='00'\t%OW='00'\n"),
        lfmt,
        std::chrono::zoned_time(std::chrono::sys_seconds{0s})); // 00:00:00 UTC Thursday, 1 January 1970

  check(SV("%U='20'\t%OU='20'\t%W='20'\t%OW='20'\n"),
        lfmt,
        std::chrono::zoned_time(std::chrono::sys_seconds{2'000'000'000s})); // 03:33:20 UTC on Wednesday, 18 May 2033

  // Use supplied locale (ja_JP). This locale has a different alternate.
#if defined(_WIN32) || defined(__APPLE__) || defined(_AIX) || defined(__FreeBSD__)
  check(loc,
        SV("%U='00'\t%OU='00'\t%W='00'\t%OW='00'\n"),
        lfmt,
        std::chrono::zoned_time(std::chrono::sys_seconds{0s})); // 00:00:00 UTC Thursday, 1 January 1970

  check(loc,
        SV("%U='20'\t%OU='20'\t%W='20'\t%OW='20'\n"),
        lfmt,
        std::chrono::zoned_time(std::chrono::sys_seconds{2'000'000'000s})); // 03:33:20 UTC on Wednesday, 18 May 2033
#else  // defined(_WIN32) || defined(__APPLE__) || defined(_AIX) || defined(__FreeBSD__)
  check(loc,
        SV("%U='00'\t%OU='〇'\t%W='00'\t%OW='〇'\n"),
        lfmt,
        std::chrono::zoned_time(std::chrono::sys_seconds{0s})); // 00:00:00 UTC Thursday, 1 January 1970

  check(loc,
        SV("%U='20'\t%OU='二十'\t%W='20'\t%OW='二十'\n"),
        lfmt,
        std::chrono::zoned_time(std::chrono::sys_seconds{2'000'000'000s})); // 03:33:20 UTC on Wednesday, 18 May 2033
#endif // defined(_WIN32) || defined(__APPLE__) || defined(_AIX) || defined(__FreeBSD__)
  std::locale::global(std::locale::classic());
}

template <class CharT>
static void test_valid_values_iso_8601_week() {
  using namespace std::literals::chrono_literals;

  constexpr std::basic_string_view<CharT> fmt  = SV("{:%%g='%g'%t%%G='%G'%t%%V='%V'%t%%OV='%OV'%n}");
  constexpr std::basic_string_view<CharT> lfmt = SV("{:L%%g='%g'%t%%G='%G'%t%%V='%V'%t%%OV='%OV'%n}");

  const std::locale loc(LOCALE_ja_JP_UTF_8);
  std::locale::global(std::locale(LOCALE_fr_FR_UTF_8));

  // Non localized output using C-locale
  check(SV("%g='70'\t%G='1970'\t%V='01'\t%OV='01'\n"),
        fmt,
        std::chrono::zoned_time(std::chrono::sys_seconds{0s})); // 00:00:00 UTC Thursday, 1 January 1970

  check(SV("%g='09'\t%G='2009'\t%V='07'\t%OV='07'\n"),
        fmt,
        std::chrono::zoned_time(std::chrono::sys_seconds{1'234'567'890s})); // 23:31:30 UTC on Friday, 13 February 2009

  // Use the global locale (fr_FR)
  check(SV("%g='70'\t%G='1970'\t%V='01'\t%OV='01'\n"),
        lfmt,
        std::chrono::zoned_time(std::chrono::sys_seconds{0s})); // 00:00:00 UTC Thursday, 1 January 1970

  check(SV("%g='09'\t%G='2009'\t%V='07'\t%OV='07'\n"),
        lfmt,
        std::chrono::zoned_time(std::chrono::sys_seconds{1'234'567'890s})); // 23:31:30 UTC on Friday, 13 February 2009

  // Use supplied locale (ja_JP). This locale has a different alternate.
#if defined(_WIN32) || defined(__APPLE__) || defined(_AIX) || defined(__FreeBSD__)
  check(loc,
        SV("%g='70'\t%G='1970'\t%V='01'\t%OV='01'\n"),
        lfmt,
        std::chrono::zoned_time(std::chrono::sys_seconds{0s})); // 00:00:00 UTC Thursday, 1 January 1970

  check(loc,
        SV("%g='09'\t%G='2009'\t%V='07'\t%OV='07'\n"),
        lfmt,
        std::chrono::zoned_time(std::chrono::sys_seconds{1'234'567'890s})); // 23:31:30 UTC on Friday, 13 February 2009
#else  // defined(_WIN32) || defined(__APPLE__) || defined(_AIX) || defined(__FreeBSD__)
  check(loc,
        SV("%g='70'\t%G='1970'\t%V='01'\t%OV='一'\n"),
        lfmt,
        std::chrono::zoned_time(std::chrono::sys_seconds{0s})); // 00:00:00 UTC Thursday, 1 January 1970

  check(loc,
        SV("%g='09'\t%G='2009'\t%V='07'\t%OV='七'\n"),
        lfmt,
        std::chrono::zoned_time(std::chrono::sys_seconds{1'234'567'890s})); // 23:31:30 UTC on Friday, 13 February 2009
#endif // defined(_WIN32) || defined(__APPLE__) || defined(_AIX) || defined(__FreeBSD__)

  std::locale::global(std::locale::classic());
}

template <class CharT>
static void test_valid_values_date() {
  using namespace std::literals::chrono_literals;

  constexpr std::basic_string_view<CharT> fmt  = SV("{:%%D='%D'%t%%F='%F'%t%%x='%x'%t%%Ex='%Ex'%n}");
  constexpr std::basic_string_view<CharT> lfmt = SV("{:L%%D='%D'%t%%F='%F'%t%%x='%x'%t%%Ex='%Ex'%n}");

  const std::locale loc(LOCALE_ja_JP_UTF_8);
  std::locale::global(std::locale(LOCALE_fr_FR_UTF_8));

  // Non localized output using C-locale
  check(SV("%D='01/01/70'\t%F='1970-01-01'\t%x='01/01/70'\t%Ex='01/01/70'\n"),
        fmt,
        std::chrono::zoned_time(std::chrono::sys_seconds{0s})); // 00:00:00 UTC Thursday, 1 January 1970

  check(SV("%D='02/13/09'\t%F='2009-02-13'\t%x='02/13/09'\t%Ex='02/13/09'\n"),
        fmt,
        std::chrono::zoned_time(std::chrono::sys_seconds{1'234'567'890s})); // 23:31:30 UTC on Friday, 13 February 2009

  // Use the global locale (fr_FR)
#if defined(__APPLE__) || defined(__FreeBSD__)
  check(SV("%D='01/01/70'\t%F='1970-01-01'\t%x='01.01.1970'\t%Ex='01.01.1970'\n"),
        lfmt,
        std::chrono::zoned_time(std::chrono::sys_seconds{0s})); // 00:00:00 UTC Thursday, 1 January 1970

  check(SV("%D='02/13/09'\t%F='2009-02-13'\t%x='13.02.2009'\t%Ex='13.02.2009'\n"),
        lfmt,
        std::chrono::zoned_time(std::chrono::sys_seconds{1'234'567'890s})); // 23:31:30 UTC on Friday, 13 February 2009
#else
  check(SV("%D='01/01/70'\t%F='1970-01-01'\t%x='01/01/1970'\t%Ex='01/01/1970'\n"),
        lfmt,
        std::chrono::zoned_time(std::chrono::sys_seconds{0s})); // 00:00:00 UTC Thursday, 1 January 1970

  check(SV("%D='02/13/09'\t%F='2009-02-13'\t%x='13/02/2009'\t%Ex='13/02/2009'\n"),
        lfmt,
        std::chrono::zoned_time(std::chrono::sys_seconds{1'234'567'890s})); // 23:31:30 UTC on Friday, 13 February 2009
#endif

  // Use supplied locale (ja_JP). This locale has a different alternate.
#if defined(_WIN32) || defined(__APPLE__) || defined(_AIX) || defined(__FreeBSD__)
  check(loc,
        SV("%D='01/01/70'\t%F='1970-01-01'\t%x='1970/01/01'\t%Ex='1970/01/01'\n"),
        lfmt,
        std::chrono::zoned_time(std::chrono::sys_seconds{0s})); // 00:00:00 UTC Thursday, 1 January 1970

  check(loc,
        SV("%D='02/13/09'\t%F='2009-02-13'\t%x='2009/02/13'\t%Ex='2009/02/13'\n"),
        lfmt,
        std::chrono::zoned_time(std::chrono::sys_seconds{1'234'567'890s})); // 23:31:30 UTC on Friday, 13 February 2009
#else  // defined(_WIN32) || defined(__APPLE__) || defined(_AIX) || defined(__FreeBSD__)
  check(loc,
        SV("%D='01/01/70'\t%F='1970-01-01'\t%x='1970年01月01日'\t%Ex='昭和45年01月01日'\n"),
        lfmt,
        std::chrono::zoned_time(std::chrono::sys_seconds{0s})); // 00:00:00 UTC Thursday, 1 January 1970

  check(loc,
        SV("%D='02/13/09'\t%F='2009-02-13'\t%x='2009年02月13日'\t%Ex='平成21年02月13日'\n"),
        lfmt,
        std::chrono::zoned_time(std::chrono::sys_seconds{1'234'567'890s})); // 23:31:30 UTC on Friday, 13 February 2009
#endif // defined(_WIN32) || defined(__APPLE__) || defined(_AIX) || defined(__FreeBSD__)

  std::locale::global(std::locale::classic());
}

template <class CharT>
static void test_valid_values_time() {
  using namespace std::literals::chrono_literals;

  constexpr std::basic_string_view<CharT> fmt = SV(
      "{:"
      "%%H='%H'%t"
      "%%OH='%OH'%t"
      "%%I='%I'%t"
      "%%OI='%OI'%t"
      "%%M='%M'%t"
      "%%OM='%OM'%t"
      "%%S='%S'%t"
      "%%OS='%OS'%t"
      "%%p='%p'%t"
      "%%R='%R'%t"
      "%%T='%T'%t"
      "%%r='%r'%t"
      "%%X='%X'%t"
      "%%EX='%EX'%t"
      "%n}");
  constexpr std::basic_string_view<CharT> lfmt = SV(
      "{:L"
      "%%H='%H'%t"
      "%%OH='%OH'%t"
      "%%I='%I'%t"
      "%%OI='%OI'%t"
      "%%M='%M'%t"
      "%%OM='%OM'%t"
      "%%S='%S'%t"
      "%%OS='%OS'%t"
      "%%p='%p'%t"
      "%%R='%R'%t"
      "%%T='%T'%t"
      "%%r='%r'%t"
      "%%X='%X'%t"
      "%%EX='%EX'%t"
      "%n}");

  const std::locale loc(LOCALE_ja_JP_UTF_8);
  std::locale::global(std::locale(LOCALE_fr_FR_UTF_8));

  // Non localized output using C-locale
  check(SV("%H='00'\t"
           "%OH='00'\t"
           "%I='12'\t"
           "%OI='12'\t"
           "%M='00'\t"
           "%OM='00'\t"
           "%S='00'\t"
           "%OS='00'\t"
           "%p='AM'\t"
           "%R='00:00'\t"
           "%T='00:00:00'\t"
           "%r='12:00:00 AM'\t"
           "%X='00:00:00'\t"
           "%EX='00:00:00'\t"
           "\n"),
        fmt,
        std::chrono::zoned_time(std::chrono::sys_seconds{0s})); // 00:00:00 UTC Thursday, 1 January 1970

  check(SV("%H='23'\t"
           "%OH='23'\t"
           "%I='11'\t"
           "%OI='11'\t"
           "%M='31'\t"
           "%OM='31'\t"
           "%S='30.123'\t"
           "%OS='30.123'\t"
           "%p='PM'\t"
           "%R='23:31'\t"
           "%T='23:31:30.123'\t"
           "%r='11:31:30 PM'\t"
           "%X='23:31:30'\t"
           "%EX='23:31:30'\t"
           "\n"),
        fmt,
        std::chrono::sys_time<std::chrono::milliseconds>(
            1'234'567'890'123ms)); // 23:31:30 UTC on Friday, 13 February 2009
  // Use the global locale (fr_FR)
  check(SV("%H='00'\t"
           "%OH='00'\t"
           "%I='12'\t"
           "%OI='12'\t"
           "%M='00'\t"
           "%OM='00'\t"
           "%S='00'\t"
           "%OS='00'\t"
#if defined(_AIX)
           "%p='AM'\t"
#else
           "%p=''\t"
#endif
           "%R='00:00'\t"
           "%T='00:00:00'\t"
#ifdef _WIN32
           "%r='00:00:00'\t"
#elif defined(_AIX)
           "%r='12:00:00 AM'\t"
#elif defined(__APPLE__) || defined(__FreeBSD__)
           "%r=''\t"
#else
           "%r='12:00:00 '\t"
#endif
           "%X='00:00:00'\t"
           "%EX='00:00:00'\t"
           "\n"),
        lfmt,
        std::chrono::zoned_time(std::chrono::sys_seconds{0s})); // 00:00:00 UTC Thursday, 1 January 1970

  check(SV("%H='23'\t"
           "%OH='23'\t"
           "%I='11'\t"
           "%OI='11'\t"
           "%M='31'\t"
           "%OM='31'\t"
           "%S='30,123'\t"
           "%OS='30,123'\t"
#if defined(_AIX)
           "%p='PM'\t"
#else
           "%p=''\t"
#endif
           "%R='23:31'\t"
           "%T='23:31:30,123'\t"
#ifdef _WIN32
           "%r='23:31:30'\t"
#elif defined(_AIX)
           "%r='11:31:30 PM'\t"
#elif defined(__APPLE__) || defined(__FreeBSD__)
           "%r=''\t"
#elif defined(_WIN32)
           "%r='23:31:30 '\t"
#else
           "%r='11:31:30 '\t"
#endif
           "%X='23:31:30'\t"
           "%EX='23:31:30'\t"
           "\n"),
        lfmt,
        std::chrono::sys_time<std::chrono::milliseconds>(
            1'234'567'890'123ms)); // 23:31:30 UTC on Friday, 13 February 2009

  // Use supplied locale (ja_JP). This locale has a different alternate.a
#if defined(__APPLE__) || defined(_AIX) || defined(_WIN32) || defined(__FreeBSD__)
  check(loc,
        SV("%H='00'\t"
           "%OH='00'\t"
           "%I='12'\t"
           "%OI='12'\t"
           "%M='00'\t"
           "%OM='00'\t"
           "%S='00'\t"
           "%OS='00'\t"
#  if defined(__APPLE__)
           "%p='AM'\t"
#  else
           "%p='午前'\t"
#  endif
           "%R='00:00'\t"
           "%T='00:00:00'\t"
#  if defined(__APPLE__) || defined(__FreeBSD__)
#    if defined(__APPLE__)
           "%r='12:00:00 AM'\t"
#    else
           "%r='12:00:00 午前'\t"
#    endif
           "%X='00時00分00秒'\t"
           "%EX='00時00分00秒'\t"
#  elif defined(_WIN32)
           "%r='0:00:00'\t"
           "%X='0:00:00'\t"
           "%EX='0:00:00'\t"
#  else
           "%r='午前12:00:00'\t"
           "%X='00:00:00'\t"
           "%EX='00:00:00'\t"
#  endif
           "\n"),
        lfmt,
        std::chrono::hh_mm_ss(0s));

  check(loc,
        SV("%H='23'\t"
           "%OH='23'\t"
           "%I='11'\t"
           "%OI='11'\t"
           "%M='31'\t"
           "%OM='31'\t"
           "%S='30.123'\t"
           "%OS='30.123'\t"
#  if defined(__APPLE__)
           "%p='PM'\t"
#  else
           "%p='午後'\t"
#  endif
           "%R='23:31'\t"
           "%T='23:31:30.123'\t"
#  if defined(__APPLE__) || defined(__FreeBSD__)
#    if defined(__APPLE__)
           "%r='11:31:30 PM'\t"
#    else
           "%r='11:31:30 午後'\t"
#    endif
           "%X='23時31分30秒'\t"
           "%EX='23時31分30秒'\t"
#  elif defined(_WIN32)
           "%r='23:31:30'\t"
           "%X='23:31:30'\t"
           "%EX='23:31:30'\t"
#  else
           "%r='午後11:31:30'\t"
           "%X='23:31:30'\t"
           "%EX='23:31:30'\t"
#  endif
           "\n"),
        lfmt,
        std::chrono::hh_mm_ss(23h + 31min + 30s + 123ms));
#else  // defined(__APPLE__) || defined(_AIX) || defined(_WIN32) || defined(__FreeBSD__)
  check(loc,
        SV("%H='00'\t"
           "%OH='〇'\t"
           "%I='12'\t"
           "%OI='十二'\t"
           "%M='00'\t"
           "%OM='〇'\t"
           "%S='00'\t"
           "%OS='〇'\t"
           "%p='午前'\t"
           "%R='00:00'\t"
           "%T='00:00:00'\t"
           "%r='午前12時00分00秒'\t"
           "%X='00時00分00秒'\t"
           "%EX='00時00分00秒'\t"
           "\n"),
        lfmt,
        std::chrono::zoned_time(std::chrono::sys_seconds{0s})); // 00:00:00 UTC Thursday, 1 January 1970

  check(loc,
        SV("%H='23'\t"
           "%OH='二十三'\t"
           "%I='11'\t"
           "%OI='十一'\t"
           "%M='31'\t"
           "%OM='三十一'\t"
           "%S='30.123'\t"
           "%OS='三十.123'\t"
           "%p='午後'\t"
           "%R='23:31'\t"
           "%T='23:31:30.123'\t"
           "%r='午後11時31分30秒'\t"
           "%X='23時31分30秒'\t"
           "%EX='23時31分30秒'\t"
           "\n"),
        lfmt,
        std::chrono::sys_time<std::chrono::milliseconds>(
            1'234'567'890'123ms)); // 23:31:30 UTC on Friday, 13 February 2009
#endif // defined(__APPLE__) || defined(_AIX) || defined(_WIN32) || defined(__FreeBSD__)

  std::locale::global(std::locale::classic());
}

template <class CharT>
static void test_valid_values_date_time() {
  using namespace std::literals::chrono_literals;

  constexpr std::basic_string_view<CharT> fmt  = SV("{:%%c='%c'%t%%Ec='%Ec'%n}");
  constexpr std::basic_string_view<CharT> lfmt = SV("{:L%%c='%c'%t%%Ec='%Ec'%n}");

  const std::locale loc(LOCALE_ja_JP_UTF_8);
  std::locale::global(std::locale(LOCALE_fr_FR_UTF_8));

  // Non localized output using C-locale
  check(SV("%c='Thu Jan  1 00:00:00 1970'\t%Ec='Thu Jan  1 00:00:00 1970'\n"),
        fmt,
        std::chrono::zoned_time(std::chrono::sys_seconds{0s})); // 00:00:00 UTC Thursday, 1 January 1970

  check(SV("%c='Fri Feb 13 23:31:30 2009'\t%Ec='Fri Feb 13 23:31:30 2009'\n"),
        fmt,
        std::chrono::zoned_time(std::chrono::sys_seconds{1'234'567'890s})); // 23:31:30 UTC on Friday, 13 February 2009

  // Use the global locale (fr_FR)
  check(
// https://sourceware.org/bugzilla/show_bug.cgi?id=24054
#if defined(__powerpc__) && defined(__linux__)
      SV("%c='jeu. 01 janv. 1970 00:00:00 UTC'\t%Ec='jeu. 01 janv. 1970 00:00:00 UTC'\n"),
#elif defined(__GLIBC__) && __GLIBC__ <= 2 && __GLIBC_MINOR__ < 29
      SV("%c='jeu. 01 janv. 1970 00:00:00 GMT'\t%Ec='jeu. 01 janv. 1970 00:00:00 GMT'\n"),
#elif defined(_AIX)
      SV("%c=' 1 janvier 1970 à 00:00:00 UTC'\t%Ec=' 1 janvier 1970 à 00:00:00 UTC'\n"),
#elif defined(__APPLE__)
      SV("%c='Jeu  1 jan 00:00:00 1970'\t%Ec='Jeu  1 jan 00:00:00 1970'\n"),
#elif defined(_WIN32)
      SV("%c='01/01/1970 00:00:00'\t%Ec='01/01/1970 00:00:00'\n"),
#elif defined(__FreeBSD__)
      SV("%c='jeu.  1 janv. 00:00:00 1970'\t%Ec='jeu.  1 janv. 00:00:00 1970'\n"),
#else
      SV("%c='jeu. 01 janv. 1970 00:00:00'\t%Ec='jeu. 01 janv. 1970 00:00:00'\n"),
#endif
      lfmt,
      std::chrono::zoned_time(std::chrono::sys_seconds{0s})); // 00:00:00 UTC Thursday, 1 January 1970

  check(
// https://sourceware.org/bugzilla/show_bug.cgi?id=24054
#if defined(__powerpc__) && defined(__linux__)
      SV("%c='ven. 13 févr. 2009 23:31:30 UTC'\t%Ec='ven. 13 févr. 2009 23:31:30 UTC'\n"),
#elif defined(__GLIBC__) && __GLIBC__ <= 2 && __GLIBC_MINOR__ < 29
      SV("%c='ven. 13 févr. 2009 23:31:30 GMT'\t%Ec='ven. 13 févr. 2009 23:31:30 GMT'\n"),
#elif defined(_AIX)
      SV("%c='13 février 2009 à 23:31:30 UTC'\t%Ec='13 février 2009 à 23:31:30 UTC'\n"),
#elif defined(__APPLE__)
      SV("%c='Ven 13 fév 23:31:30 2009'\t%Ec='Ven 13 fév 23:31:30 2009'\n"),
#elif defined(_WIN32)
      SV("%c='13/02/2009 23:31:30'\t%Ec='13/02/2009 23:31:30'\n"),
#elif defined(__FreeBSD__)
      SV("%c='ven. 13 févr. 23:31:30 2009'\t%Ec='ven. 13 févr. 23:31:30 2009'\n"),
#else
      SV("%c='ven. 13 févr. 2009 23:31:30'\t%Ec='ven. 13 févr. 2009 23:31:30'\n"),
#endif
      lfmt,
      std::chrono::zoned_time(std::chrono::sys_seconds{1'234'567'890s})); // 23:31:30 UTC on Friday, 13 February 2009

  // Use supplied locale (ja_JP). This locale has a different alternate.a
#if defined(__APPLE__) || defined(__FreeBSD__)
  check(loc,
        SV("%c='木  1/ 1 00:00:00 1970'\t%Ec='木  1/ 1 00:00:00 1970'\n"),
        lfmt,
        std::chrono::zoned_time(std::chrono::sys_seconds{0s})); // 00:00:00 UTC Thursday, 1 January 1970
  check(loc,
        SV("%c='金  2/13 23:31:30 2009'\t%Ec='金  2/13 23:31:30 2009'\n"),
        lfmt,
        std::chrono::zoned_time(std::chrono::sys_seconds{1'234'567'890s})); // 23:31:30 UTC on Friday, 13 February 2009
#elif defined(_AIX)                                                         // defined(__APPLE__)|| defined(__FreeBSD__)
  check(loc,
        SV("%c='1970年01月 1日 00:00:00 UTC'\t%Ec='1970年01月 1日 00:00:00 UTC'\n"),
        lfmt,
        std::chrono::zoned_time(std::chrono::sys_seconds{0s})); // 00:00:00 UTC Thursday, 1 January 1970
  check(loc,
        SV("%c='2009年02月13日 23:31:30 UTC'\t%Ec='2009年02月13日 23:31:30 UTC'\n"),
        lfmt,
        std::chrono::zoned_time(std::chrono::sys_seconds{1'234'567'890s})); // 23:31:30 UTC on Friday, 13 February 2009
#elif defined(_WIN32)                                                       // defined(__APPLE__)|| defined(__FreeBSD__)
  check(loc,
        SV("%c='1970/01/01 0:00:00'\t%Ec='1970/01/01 0:00:00'\n"),
        lfmt,
        std::chrono::sys_seconds(0s)); // 00:00:00 UTC Thursday, 1 January 1970
  check(loc,
        SV("%c='2009/02/13 23:31:30'\t%Ec='2009/02/13 23:31:30'\n"),
        lfmt,
        std::chrono::sys_seconds(1'234'567'890s)); // 23:31:30 UTC on Friday, 13 February 2009
#else                                                                       // defined(__APPLE__)|| defined(__FreeBSD__)
  check(loc,
        SV("%c='1970年01月01日 00時00分00秒'\t%Ec='昭和45年01月01日 00時00分00秒'\n"),
        lfmt,
        std::chrono::zoned_time(std::chrono::sys_seconds{0s})); // 00:00:00 UTC Thursday, 1 January 1970

  check(loc,
        SV("%c='2009年02月13日 23時31分30秒'\t%Ec='平成21年02月13日 23時31分30秒'\n"),
        lfmt,
        std::chrono::zoned_time(std::chrono::sys_seconds{1'234'567'890s})); // 23:31:30 UTC on Friday, 13 February 2009
#endif                                                                      // defined(__APPLE__)|| defined(__FreeBSD__)

  std::locale::global(std::locale::classic());
}

template <class CharT>
static void test_valid_values_time_zone() {
  using namespace std::literals::chrono_literals;

  constexpr std::basic_string_view<CharT> fmt  = SV("{:%%z='%z'%t%%Ez='%Ez'%t%%Oz='%Oz'%t%%Z='%Z'%n}");
  constexpr std::basic_string_view<CharT> lfmt = SV("{:L%%z='%z'%t%%Ez='%Ez'%t%%Oz='%Oz'%t%%Z='%Z'%n}");

  const std::locale loc(LOCALE_ja_JP_UTF_8);
  std::locale::global(std::locale(LOCALE_fr_FR_UTF_8));

  // Non localized output using C-locale
  check(SV("%z='+0000'\t%Ez='+00:00'\t%Oz='+00:00'\t%Z='UTC'\n"),
        fmt,
        std::chrono::zoned_time(std::chrono::sys_seconds{0s})); // 00:00:00 UTC Thursday, 1 January 1970

  // Use the global locale (fr_FR)
  check(SV("%z='+0000'\t%Ez='+00:00'\t%Oz='+00:00'\t%Z='UTC'\n"),
        lfmt,
        std::chrono::zoned_time(std::chrono::sys_seconds{0s})); // 00:00:00 UTC Thursday, 1 January 1970

  // Use supplied locale (ja_JP).
  check(loc,
        SV("%z='+0000'\t%Ez='+00:00'\t%Oz='+00:00'\t%Z='UTC'\n"),
        lfmt,
        std::chrono::zoned_time(std::chrono::sys_seconds{0s})); // 00:00:00 UTC Thursday, 1 January 1970

  std::locale::global(std::locale::classic());
}

template <class CharT>
static void test_valid_values() {
  test_valid_values_year<CharT>();
  test_valid_values_month<CharT>();
  test_valid_values_day<CharT>();
  test_valid_values_weekday<CharT>();
  test_valid_values_day_of_year<CharT>();
  test_valid_values_week<CharT>();
  test_valid_values_iso_8601_week<CharT>();
  test_valid_values_date<CharT>();
  test_valid_values_time<CharT>();
  test_valid_values_date_time<CharT>();
  test_valid_values_time_zone<CharT>();
}

template <class CharT>
static void test() {
  test_no_chrono_specs<CharT>();
  test_valid_values<CharT>();

  check_invalid_types<CharT>(
      {SV("a"),  SV("A"),  SV("b"),  SV("B"),  SV("c"),  SV("C"),  SV("d"),  SV("D"),  SV("e"),  SV("F"),  SV("g"),
       SV("G"),  SV("h"),  SV("H"),  SV("I"),  SV("j"),  SV("m"),  SV("M"),  SV("p"),  SV("r"),  SV("R"),  SV("S"),
       SV("T"),  SV("u"),  SV("U"),  SV("V"),  SV("w"),  SV("W"),  SV("x"),  SV("X"),  SV("y"),  SV("Y"),  SV("z"),
       SV("Z"),  SV("Ec"), SV("EC"), SV("Ex"), SV("EX"), SV("Ey"), SV("EY"), SV("Ez"), SV("Od"), SV("Oe"), SV("OH"),
       SV("OI"), SV("Om"), SV("OM"), SV("OS"), SV("Ou"), SV("OU"), SV("OV"), SV("Ow"), SV("OW"), SV("Oy"), SV("Oz")},
      std::chrono::zoned_time{});
}

int main(int, char**) {
  test<char>();

#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  test<wchar_t>();
#endif

  return 0;
}
