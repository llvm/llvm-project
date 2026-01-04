//===----------------------------------------------------------------------===//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++20
// UNSUPPORTED: no-filesystem, no-localization, no-tzdb
// UNSUPPORTED: GCC-ALWAYS_INLINE-FIXME

// TODO FMT This test should not require std::to_chars(floating-point)
// XFAIL: availability-fp_to_chars-missing

// XFAIL: libcpp-has-no-experimental-tzdb
// XFAIL: availability-tzdb-missing

// REQUIRES: locale.fr_FR.UTF-8
// REQUIRES: locale.ja_JP.UTF-8

// <chrono>

// template<class Duration, class charT>
//   struct formatter<chrono::tai_time<Duration>, charT>;

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
  namespace cr = std::chrono;

  std::locale::global(std::locale(LOCALE_fr_FR_UTF_8));

  // Non localized output

  // [time.syn]
  //   using nanoseconds  = duration<signed integer type of at least 64 bits, nano>;
  //   using microseconds = duration<signed integer type of at least 55 bits, micro>;
  //   using milliseconds = duration<signed integer type of at least 45 bits, milli>;
  //   using seconds      = duration<signed integer type of at least 35 bits>;
  //   using minutes      = duration<signed integer type of at least 29 bits, ratio<  60>>;
  //   using hours        = duration<signed integer type of at least 23 bits, ratio<3600>>;
  check(SV("1413-08-04 22:06:56"), SV("{}"), cr::tai_seconds(-17'179'869'184s)); // Minimum value for 35 bits.
  check(SV("1889-12-12 20:45:52"), SV("{}"), cr::tai_seconds(-2'147'483'648s));

  check(SV("1957-12-31 00:00:00"), SV("{}"), cr::tai_seconds(-24h));
  check(SV("1957-12-31 06:00:00"), SV("{}"), cr::tai_seconds(-18h));
  check(SV("1957-12-31 12:00:00"), SV("{}"), cr::tai_seconds(-12h));
  check(SV("1957-12-31 18:00:00"), SV("{}"), cr::tai_seconds(-6h));
  check(SV("1957-12-31 23:59:59"), SV("{}"), cr::tai_seconds(-1s));

  check(SV("1958-01-01 00:00:00"), SV("{}"), cr::tai_seconds(0s));
  check(SV("1988-01-01 00:00:00"), SV("{}"), cr::tai_seconds(946'684'800s));
  check(SV("1988-01-01 01:02:03"), SV("{}"), cr::tai_seconds(946'688'523s));

  check(SV("2026-01-19 03:14:07"), SV("{}"), cr::tai_seconds(2'147'483'647s));
  check(SV("2502-05-30 01:53:03"), SV("{}"), cr::tai_seconds(17'179'869'183s)); // Maximum value for 35 bits.

  check(SV("1988-01-01 01:02:03.123"), SV("{}"), cr::tai_time<cr::milliseconds>(946'688'523'123ms));

  std::locale::global(std::locale::classic());
}

template <class CharT>
static void test_valid_values_year() {
  using namespace std::literals::chrono_literals;
  namespace cr = std::chrono;

  constexpr std::basic_string_view<CharT> fmt =
      SV("{:%%C='%C'%t%%EC='%EC'%t%%y='%y'%t%%Oy='%Oy'%t%%Ey='%Ey'%t%%Y='%Y'%t%%EY='%EY'%n}");
  constexpr std::basic_string_view<CharT> lfmt =
      SV("{:L%%C='%C'%t%%EC='%EC'%t%%y='%y'%t%%Oy='%Oy'%t%%Ey='%Ey'%t%%Y='%Y'%t%%EY='%EY'%n}");

  const std::locale loc(LOCALE_ja_JP_UTF_8);
  std::locale::global(std::locale(LOCALE_fr_FR_UTF_8));

  // Non localized output using C-locale
  check(SV("%C='19'\t%EC='19'\t%y='58'\t%Oy='58'\t%Ey='58'\t%Y='1958'\t%EY='1958'\n"),
        fmt,
        cr::tai_seconds(0s)); // 00:00:00 TAI Wednesday, 1 January 1958

  check(SV("%C='20'\t%EC='20'\t%y='09'\t%Oy='09'\t%Ey='09'\t%Y='2009'\t%EY='2009'\n"),
        fmt,
        cr::tai_seconds(1'613'259'090s)); // 23:31:30 TAI Friday, 13 February 2009

  // Use the global locale (fr_FR)
  check(SV("%C='19'\t%EC='19'\t%y='58'\t%Oy='58'\t%Ey='58'\t%Y='1958'\t%EY='1958'\n"),
        lfmt,
        cr::tai_seconds(0s)); // 00:00:00 TAI Wednesday, 1 January 1958

  check(SV("%C='20'\t%EC='20'\t%y='09'\t%Oy='09'\t%Ey='09'\t%Y='2009'\t%EY='2009'\n"),
        lfmt,
        cr::tai_seconds(1'613'259'090s)); // 23:31:30 TAI Friday, 13 February 2009

  // Use supplied locale (ja_JP). This locale has a different alternate.
#if defined(_WIN32) || defined(__APPLE__) || defined(_AIX) || defined(__FreeBSD__)
  check(loc,
        SV("%C='19'\t%EC='19'\t%y='58'\t%Oy='58'\t%Ey='58'\t%Y='1958'\t%EY='1958'\n"),
        lfmt,
        cr::tai_seconds(0s)); // 00:00:00 TAI Wednesday, 1 January 1958

  check(loc,
        SV("%C='20'\t%EC='20'\t%y='09'\t%Oy='09'\t%Ey='09'\t%Y='2009'\t%EY='2009'\n"),
        lfmt,
        cr::tai_seconds(1'613'259'090s)); // 23:31:30 TAI Friday, 13 February 2009
#else  // defined(_WIN32) || defined(__APPLE__) || defined(_AIX)||defined(__FreeBSD__)
  check(loc,
        SV("%C='19'\t%EC='昭和'\t%y='58'\t%Oy='五十八'\t%Ey='33'\t%Y='1958'\t%EY='昭和33年'\n"),
        lfmt,
        cr::tai_seconds(0s)); // 00:00:00 TAI Wednesday, 1 January 1958

  check(loc,
        SV("%C='20'\t%EC='平成'\t%y='09'\t%Oy='九'\t%Ey='21'\t%Y='2009'\t%EY='平成21年'\n"),
        lfmt,
        cr::tai_seconds(1'613'259'090s)); // 23:31:30 TAI Friday, 13 February 2009
#endif // defined(_WIN32) || defined(__APPLE__) || defined(_AIX)||defined(__FreeBSD__)

  std::locale::global(std::locale::classic());
}

template <class CharT>
static void test_valid_values_month() {
  using namespace std::literals::chrono_literals;
  namespace cr = std::chrono;

  constexpr std::basic_string_view<CharT> fmt  = SV("{:%%b='%b'%t%%h='%h'%t%%B='%B'%t%%m='%m'%t%%Om='%Om'%n}");
  constexpr std::basic_string_view<CharT> lfmt = SV("{:L%%b='%b'%t%%h='%h'%t%%B='%B'%t%%m='%m'%t%%Om='%Om'%n}");

  const std::locale loc(LOCALE_ja_JP_UTF_8);
  std::locale::global(std::locale(LOCALE_fr_FR_UTF_8));

  // Non localized output using C-locale
  check(SV("%b='Jan'\t%h='Jan'\t%B='January'\t%m='01'\t%Om='01'\n"),
        fmt,
        cr::tai_seconds(0s)); // 00:00:00 TAI Wednesday, 1 January 1958

  check(SV("%b='May'\t%h='May'\t%B='May'\t%m='05'\t%Om='05'\n"),
        fmt,
        cr::tai_seconds(2'378'691'200s)); // 03:33:20 TAI on Wednesday, 18 May 2033

  // Use the global locale (fr_FR)
#if defined(__APPLE__)
  check(SV("%b='jan'\t%h='jan'\t%B='janvier'\t%m='01'\t%Om='01'\n"),
        lfmt,
        cr::tai_seconds(0s)); // 00:00:00 TAI Wednesday, 1 January 1958
#else
  check(SV("%b='janv.'\t%h='janv.'\t%B='janvier'\t%m='01'\t%Om='01'\n"),
        lfmt,
        cr::tai_seconds(0s)); // 00:00:00 TAI Wednesday, 1 January 1958
#endif

  check(SV("%b='mai'\t%h='mai'\t%B='mai'\t%m='05'\t%Om='05'\n"),
        lfmt,
        cr::tai_seconds(2'378'691'200s)); // 03:33:20 TAI on Wednesday, 18 May 2033

  // Use supplied locale (ja_JP). This locale has a different alternate.
#ifdef _WIN32
  check(loc,
        SV("%b='1'\t%h='1'\t%B='1月'\t%m='01'\t%Om='01'\n"),
        lfmt,
        cr::tai_seconds(0s)); // 00:00:00 TAI Wednesday, 1 January 1958

  check(loc,
        SV("%b='5'\t%h='5'\t%B='5月'\t%m='05'\t%Om='05'\n"),
        lfmt,
        cr::tai_seconds(2'378'691'200s)); // 03:33:20 TAI Wednesday, 18 May 2033
#elif defined(_AIX)                       // _WIN32
  check(loc,
        SV("%b='1月'\t%h='1月'\t%B='1月'\t%m='01'\t%Om='01'\n"),
        lfmt,
        cr::tai_seconds(0s)); // 00:00:00 TAI Wednesday, 1 January 1958

  check(loc,
        SV("%b='5月'\t%h='5月'\t%B='5月'\t%m='05'\t%Om='05'\n"),
        lfmt,
        cr::tai_seconds(2'378'691'200s)); // 03:33:20 TAI Wednesday, 18 May 2033
#elif defined(__APPLE__)                  // _WIN32
  check(loc,
        SV("%b=' 1'\t%h=' 1'\t%B='1月'\t%m='01'\t%Om='01'\n"),
        lfmt,
        cr::tai_seconds(0s)); // 00:00:00 TAI Wednesday, 1 January 1958

  check(loc,
        SV("%b=' 5'\t%h=' 5'\t%B='5月'\t%m='05'\t%Om='05'\n"),
        lfmt,
        cr::tai_seconds(2'378'691'200s)); // 03:33:20 TAI Wednesday, 18 May 2033
#elif defined(__FreeBSD__)                // _WIN32
  check(loc,
        SV("%b=' 1月'\t%h=' 1月'\t%B='1月'\t%m='01'\t%Om='01'\n"),
        lfmt,
        cr::tai_seconds(0s)); // 00:00:00 TAI Wednesday, 1 January 1958

  check(loc,
        SV("%b=' 5月'\t%h=' 5月'\t%B='5月'\t%m='05'\t%Om='05'\n"),
        lfmt,
        cr::tai_seconds(2'378'691'200s)); // 03:33:20 TAI Wednesday, 18 May 2033
#else                                     // _WIN32
  check(loc,
        SV("%b=' 1月'\t%h=' 1月'\t%B='1月'\t%m='01'\t%Om='一'\n"),
        lfmt,
        cr::tai_seconds(0s)); // 00:00:00 TAI Wednesday, 1 January 1958

  check(loc,
        SV("%b=' 5月'\t%h=' 5月'\t%B='5月'\t%m='05'\t%Om='五'\n"),
        lfmt,
        cr::tai_seconds(2'378'691'200s)); // 03:33:20 TAI Wednesday, 18 May 2033
#endif                                    // _WIN32

  std::locale::global(std::locale::classic());
}

template <class CharT>
static void test_valid_values_day() {
  using namespace std::literals::chrono_literals;
  namespace cr = std::chrono;

  constexpr std::basic_string_view<CharT> fmt  = SV("{:%%d='%d'%t%%Od='%Od'%t%%e='%e'%t%%Oe='%Oe'%n}");
  constexpr std::basic_string_view<CharT> lfmt = SV("{:L%%d='%d'%t%%Od='%Od'%t%%e='%e'%t%%Oe='%Oe'%n}");

  const std::locale loc(LOCALE_ja_JP_UTF_8);
  std::locale::global(std::locale(LOCALE_fr_FR_UTF_8));

  // Non localized output using C-locale
  check(SV("%d='01'\t%Od='01'\t%e=' 1'\t%Oe=' 1'\n"),
        fmt,
        cr::tai_seconds(0s)); // 00:00:00 TAI Wednesday, 1 January 1958

  check(SV("%d='13'\t%Od='13'\t%e='13'\t%Oe='13'\n"),
        fmt,
        cr::tai_seconds(1'613'259'090s)); // 23:31:30 TAI Friday, 13 February 2009

  // Use the global locale (fr_FR)
  check(SV("%d='01'\t%Od='01'\t%e=' 1'\t%Oe=' 1'\n"),
        lfmt,
        cr::tai_seconds(0s)); // 00:00:00 TAI Wednesday, 1 January 1958

  check(SV("%d='13'\t%Od='13'\t%e='13'\t%Oe='13'\n"),
        lfmt,
        cr::tai_seconds(1'613'259'090s)); // 23:31:30 TAI Friday, 13 February 2009

  // Use the global locale (fr_FR)
  check(SV("%d='01'\t%Od='01'\t%e=' 1'\t%Oe=' 1'\n"),
        lfmt,
        cr::tai_seconds(0s)); // 00:00:00 TAI Wednesday, 1 January 1958

  // Use supplied locale (ja_JP). This locale has a different alternate.
#if defined(_WIN32) || defined(__APPLE__) || defined(_AIX) || defined(__FreeBSD__)
  check(loc,
        SV("%d='01'\t%Od='01'\t%e=' 1'\t%Oe=' 1'\n"),
        lfmt,
        cr::tai_seconds(0s)); // 00:00:00 TAI Wednesday, 1 January 1958

  check(loc,
        SV("%d='13'\t%Od='13'\t%e='13'\t%Oe='13'\n"),
        lfmt,
        cr::tai_seconds(1'613'259'090s)); // 23:31:30 TAI Friday, 13 February 2009

#else // defined(_WIN32) || defined(__APPLE__) || defined(_AIX) || defined(__FreeBSD__)
  check(loc,
        SV("%d='01'\t%Od='一'\t%e=' 1'\t%Oe='一'\n"),
        lfmt,
        cr::tai_seconds(0s)); // 00:00:00 TAI Wednesday, 1 January 1958

  check(loc,
        SV("%d='13'\t%Od='十三'\t%e='13'\t%Oe='十三'\n"),
        lfmt,
        cr::tai_seconds(1'613'259'090s)); // 23:31:30 TAI Friday, 13 February 2009

#endif // defined(_WIN32) || defined(__APPLE__) || defined(_AIX) || defined(__FreeBSD__)

  std::locale::global(std::locale::classic());
}

template <class CharT>
static void test_valid_values_weekday() {
  using namespace std::literals::chrono_literals;
  namespace cr = std::chrono;

  constexpr std::basic_string_view<CharT> fmt =
      SV("{:%%a='%a'%t%%A='%A'%t%%u='%u'%t%%Ou='%Ou'%t%%w='%w'%t%%Ow='%Ow'%n}");
  constexpr std::basic_string_view<CharT> lfmt =
      SV("{:L%%a='%a'%t%%A='%A'%t%%u='%u'%t%%Ou='%Ou'%t%%w='%w'%t%%Ow='%Ow'%n}");

  const std::locale loc(LOCALE_ja_JP_UTF_8);
  std::locale::global(std::locale(LOCALE_fr_FR_UTF_8));

  // Non localized output using C-locale
  check(SV("%a='Wed'\t%A='Wednesday'\t%u='3'\t%Ou='3'\t%w='3'\t%Ow='3'\n"),
        fmt,
        cr::tai_seconds(0s)); // 00:00:00 TAI Wednesday, 1 January 1958

  check(SV("%a='Sun'\t%A='Sunday'\t%u='7'\t%Ou='7'\t%w='0'\t%Ow='0'\n"),
        fmt,
        cr::tai_seconds(4'673'658'495s)); // 06:28:15 TAI on Sunday, 7 February 2106

  // Use the global locale (fr_FR)
#if defined(__APPLE__)
  check(SV("%a='mer'\t%A='Mercredi'\t%u='3'\t%Ou='3'\t%w='3'\t%Ow='3'\n"),
        lfmt,
        cr::tai_seconds(0s)); // 00:00:00 TAI Wednesday, 1 January 1958

  check(SV("%a='Dim'\t%A='Dimanche'\t%u='7'\t%Ou='7'\t%w='0'\t%Ow='0'\n"),
        lfmt,
        cr::tai_seconds(4'673'658'495s)); // 06:28:15 TAI on Sunday, 7 February 2106
#else
  check(SV("%a='mer.'\t%A='mercredi'\t%u='3'\t%Ou='3'\t%w='3'\t%Ow='3'\n"),
        lfmt,
        cr::tai_seconds(0s)); // 00:00:00 TAI Wednesday, 1 January 1958

  check(SV("%a='dim.'\t%A='dimanche'\t%u='7'\t%Ou='7'\t%w='0'\t%Ow='0'\n"),
        lfmt,
        cr::tai_seconds(4'673'658'495s)); // 06:28:15 TAI on Sunday, 7 February 2106
#endif

  // Use supplied locale (ja_JP).
  // This locale has a different alternate, but not on all platforms
#if defined(_WIN32) || defined(__APPLE__) || defined(_AIX) || defined(__FreeBSD__)
  check(loc,
        SV("%a='水'\t%A='水曜日'\t%u='3'\t%Ou='3'\t%w='3'\t%Ow='3'\n"),
        lfmt,
        cr::tai_seconds(0s)); // 00:00:00 TAI Wednesday, 1 January 1958

  check(loc,
        SV("%a='日'\t%A='日曜日'\t%u='7'\t%Ou='7'\t%w='0'\t%Ow='0'\n"),
        lfmt,
        cr::tai_seconds(4'673'658'495s)); // 06:28:15 TAI on Sunday, 7 February 2106
#else  // defined(_WIN32) || defined(__APPLE__) || defined(_AIX) || defined(__FreeBSD__)
  check(loc,
        SV("%a='水'\t%A='水曜日'\t%u='3'\t%Ou='三'\t%w='3'\t%Ow='三'\n"),
        lfmt,
        cr::tai_seconds(0s)); // 00:00:00 TAI Wednesday, 1 January 1958

  check(loc,
        SV("%a='日'\t%A='日曜日'\t%u='7'\t%Ou='七'\t%w='0'\t%Ow='〇'\n"),
        lfmt,
        cr::tai_seconds(4'673'658'495s)); // 06:28:15 TAI on Sunday, 7 February 2106
#endif // defined(_WIN32) || defined(__APPLE__) || defined(_AIX) || defined(__FreeBSD__)

  std::locale::global(std::locale::classic());
}

template <class CharT>
static void test_valid_values_day_of_year() {
  using namespace std::literals::chrono_literals;
  namespace cr = std::chrono;

  constexpr std::basic_string_view<CharT> fmt  = SV("{:%%j='%j'%n}");
  constexpr std::basic_string_view<CharT> lfmt = SV("{:L%%j='%j'%n}");

  const std::locale loc(LOCALE_ja_JP_UTF_8);
  std::locale::global(std::locale(LOCALE_fr_FR_UTF_8));

  // Non localized output using C-locale
  check(SV("%j='001'\n"), fmt, cr::tai_seconds(0s));             // 00:00:00 TAI Wednesday, 1 January 1958
  check(SV("%j='138'\n"), fmt, cr::tai_seconds(2'378'691'200s)); // 03:33:20 TAI Wednesday, 18 May 2033

  // Use the global locale (fr_FR)
  check(SV("%j='001'\n"), lfmt, cr::tai_seconds(0s));             // 00:00:00 TAI Wednesday, 1 January 1958
  check(SV("%j='138'\n"), lfmt, cr::tai_seconds(2'378'691'200s)); // 03:33:20 TAI Wednesday, 18 May 2033

  // Use supplied locale (ja_JP). This locale has a different alternate.
  check(loc, SV("%j='001'\n"), lfmt, cr::tai_seconds(0s));             // 00:00:00 TAI Wednesday, 1 January 1958
  check(loc, SV("%j='138'\n"), lfmt, cr::tai_seconds(2'378'691'200s)); // 03:33:20 TAI Wednesday, 18 May 2033

  std::locale::global(std::locale::classic());
}

template <class CharT>
static void test_valid_values_week() {
  using namespace std::literals::chrono_literals;
  namespace cr = std::chrono;

  constexpr std::basic_string_view<CharT> fmt  = SV("{:%%U='%U'%t%%OU='%OU'%t%%W='%W'%t%%OW='%OW'%n}");
  constexpr std::basic_string_view<CharT> lfmt = SV("{:L%%U='%U'%t%%OU='%OU'%t%%W='%W'%t%%OW='%OW'%n}");

  const std::locale loc(LOCALE_ja_JP_UTF_8);
  std::locale::global(std::locale(LOCALE_fr_FR_UTF_8));

  // Non localized output using C-locale
  check(SV("%U='00'\t%OU='00'\t%W='00'\t%OW='00'\n"),
        fmt,
        cr::tai_seconds(0s)); // 00:00:00 TAI Wednesday, 1 January 1958

  check(SV("%U='20'\t%OU='20'\t%W='20'\t%OW='20'\n"),
        fmt,
        cr::tai_seconds(2'378'691'200s)); // 03:33:20 TAI Wednesday, 18 May 2033

  // Use the global locale (fr_FR)
  check(SV("%U='00'\t%OU='00'\t%W='00'\t%OW='00'\n"),
        lfmt,
        cr::tai_seconds(0s)); // 00:00:00 TAI Wednesday, 1 January 1958

  check(SV("%U='20'\t%OU='20'\t%W='20'\t%OW='20'\n"),
        lfmt,
        cr::tai_seconds(2'378'691'200s)); // 03:33:20 TAI Wednesday, 18 May 2033

  // Use supplied locale (ja_JP). This locale has a different alternate.
#if defined(_WIN32) || defined(__APPLE__) || defined(_AIX) || defined(__FreeBSD__)
  check(loc,
        SV("%U='00'\t%OU='00'\t%W='00'\t%OW='00'\n"),
        lfmt,
        cr::tai_seconds(0s)); // 00:00:00 TAI Wednesday, 1 January 1958

  check(loc,
        SV("%U='20'\t%OU='20'\t%W='20'\t%OW='20'\n"),
        lfmt,
        cr::tai_seconds(2'378'691'200s)); // 03:33:20 TAI Wednesday, 18 May 2033
#else  // defined(_WIN32) || defined(__APPLE__) || defined(_AIX) || defined(__FreeBSD__)
  check(loc,
        SV("%U='00'\t%OU='〇'\t%W='00'\t%OW='〇'\n"),
        lfmt,
        cr::tai_seconds(0s)); // 00:00:00 TAI Wednesday, 1 January 1958

  check(loc,
        SV("%U='20'\t%OU='二十'\t%W='20'\t%OW='二十'\n"),
        lfmt,
        cr::tai_seconds(2'378'691'200s)); // 03:33:20 TAI Wednesday, 18 May 2033
#endif // defined(_WIN32) || defined(__APPLE__) || defined(_AIX) || defined(__FreeBSD__)
  std::locale::global(std::locale::classic());
}

template <class CharT>
static void test_valid_values_iso_8601_week() {
  using namespace std::literals::chrono_literals;
  namespace cr = std::chrono;

  constexpr std::basic_string_view<CharT> fmt  = SV("{:%%g='%g'%t%%G='%G'%t%%V='%V'%t%%OV='%OV'%n}");
  constexpr std::basic_string_view<CharT> lfmt = SV("{:L%%g='%g'%t%%G='%G'%t%%V='%V'%t%%OV='%OV'%n}");

  const std::locale loc(LOCALE_ja_JP_UTF_8);
  std::locale::global(std::locale(LOCALE_fr_FR_UTF_8));

  // Non localized output using C-locale
  check(SV("%g='58'\t%G='1958'\t%V='01'\t%OV='01'\n"),
        fmt,
        cr::tai_seconds(0s)); // 00:00:00 TAI Wednesday, 1 January 1958

  check(SV("%g='09'\t%G='2009'\t%V='07'\t%OV='07'\n"),
        fmt,
        cr::tai_seconds(1'613'259'090s)); // 23:31:30 TAI Friday, 13 February 2009

  // Use the global locale (fr_FR)
  check(SV("%g='58'\t%G='1958'\t%V='01'\t%OV='01'\n"),
        lfmt,
        cr::tai_seconds(0s)); // 00:00:00 TAI Wednesday, 1 January 1958

  check(SV("%g='09'\t%G='2009'\t%V='07'\t%OV='07'\n"),
        lfmt,
        cr::tai_seconds(1'613'259'090s)); // 23:31:30 TAI Friday, 13 February 2009

  // Use supplied locale (ja_JP). This locale has a different alternate.
#if defined(_WIN32) || defined(__APPLE__) || defined(_AIX) || defined(__FreeBSD__)
  check(loc,
        SV("%g='58'\t%G='1958'\t%V='01'\t%OV='01'\n"),
        lfmt,
        cr::tai_seconds(0s)); // 00:00:00 TAI Wednesday, 1 January 1958

  check(loc,
        SV("%g='09'\t%G='2009'\t%V='07'\t%OV='07'\n"),
        lfmt,
        cr::tai_seconds(1'613'259'090s)); // 23:31:30 TAI Friday, 13 February 2009
#else  // defined(_WIN32) || defined(__APPLE__) || defined(_AIX) || defined(__FreeBSD__)
  check(loc,
        SV("%g='58'\t%G='1958'\t%V='01'\t%OV='一'\n"),
        lfmt,
        cr::tai_seconds(0s)); // 00:00:00 TAI Wednesday, 1 January 1958

  check(loc,
        SV("%g='09'\t%G='2009'\t%V='07'\t%OV='七'\n"),
        lfmt,
        cr::tai_seconds(1'613'259'090s)); // 23:31:30 TAI Friday, 13 February 2009
#endif // defined(_WIN32) || defined(__APPLE__) || defined(_AIX) || defined(__FreeBSD__)

  std::locale::global(std::locale::classic());
}

template <class CharT>
static void test_valid_values_date() {
  using namespace std::literals::chrono_literals;
  namespace cr = std::chrono;

  constexpr std::basic_string_view<CharT> fmt  = SV("{:%%D='%D'%t%%F='%F'%t%%x='%x'%t%%Ex='%Ex'%n}");
  constexpr std::basic_string_view<CharT> lfmt = SV("{:L%%D='%D'%t%%F='%F'%t%%x='%x'%t%%Ex='%Ex'%n}");

  const std::locale loc(LOCALE_ja_JP_UTF_8);
  std::locale::global(std::locale(LOCALE_fr_FR_UTF_8));

  // Non localized output using C-locale
  check(SV("%D='01/01/58'\t%F='1958-01-01'\t%x='01/01/58'\t%Ex='01/01/58'\n"),
        fmt,
        cr::tai_seconds(0s)); // 00:00:00 TAI Wednesday, 1 January 1958

  check(SV("%D='02/13/09'\t%F='2009-02-13'\t%x='02/13/09'\t%Ex='02/13/09'\n"),
        fmt,
        cr::tai_seconds(1'613'259'090s)); // 23:31:30 TAI Friday, 13 February 2009

  // Use the global locale (fr_FR)
#if defined(__APPLE__) || defined(__FreeBSD__)
  check(SV("%D='01/01/58'\t%F='1958-01-01'\t%x='01.01.1958'\t%Ex='01.01.1958'\n"),
        lfmt,
        cr::tai_seconds(0s)); // 00:00:00 TAI Wednesday, 1 January 1958

  check(SV("%D='02/13/09'\t%F='2009-02-13'\t%x='13.02.2009'\t%Ex='13.02.2009'\n"),
        lfmt,
        cr::tai_seconds(1'613'259'090s)); // 23:31:30 TAI Friday, 13 February 2009
#else
  check(SV("%D='01/01/58'\t%F='1958-01-01'\t%x='01/01/1958'\t%Ex='01/01/1958'\n"),
        lfmt,
        cr::tai_seconds(0s)); // 00:00:00 TAI Wednesday, 1 January 1958

  check(SV("%D='02/13/09'\t%F='2009-02-13'\t%x='13/02/2009'\t%Ex='13/02/2009'\n"),
        lfmt,
        cr::tai_seconds(1'613'259'090s)); // 23:31:30 TAI Friday, 13 February 2009
#endif

  // Use supplied locale (ja_JP). This locale has a different alternate.
#if defined(_WIN32) || defined(__APPLE__) || defined(_AIX) || defined(__FreeBSD__)
  check(loc,
        SV("%D='01/01/58'\t%F='1958-01-01'\t%x='1958/01/01'\t%Ex='1958/01/01'\n"),
        lfmt,
        cr::tai_seconds(0s)); // 00:00:00 TAI Wednesday, 1 January 1958

  check(loc,
        SV("%D='02/13/09'\t%F='2009-02-13'\t%x='2009/02/13'\t%Ex='2009/02/13'\n"),
        lfmt,
        cr::tai_seconds(1'613'259'090s)); // 23:31:30 TAI Friday, 13 February 2009
#else  // defined(_WIN32) || defined(__APPLE__) || defined(_AIX) || defined(__FreeBSD__)
  check(loc,
        SV("%D='01/01/58'\t%F='1958-01-01'\t%x='1958年01月01日'\t%Ex='昭和33年01月01日'\n"),
        lfmt,
        cr::tai_seconds(0s)); // 00:00:00 TAI Wednesday, 1 January 1958

  check(loc,
        SV("%D='02/13/09'\t%F='2009-02-13'\t%x='2009年02月13日'\t%Ex='平成21年02月13日'\n"),
        lfmt,
        cr::tai_seconds(1'613'259'090s)); // 23:31:30 TAI Friday, 13 February 2009
#endif // defined(_WIN32) || defined(__APPLE__) || defined(_AIX) || defined(__FreeBSD__)

  std::locale::global(std::locale::classic());
}

template <class CharT>
static void test_valid_values_time() {
  using namespace std::literals::chrono_literals;
  namespace cr = std::chrono;

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
        cr::tai_seconds(0s)); // 00:00:00 TAI Wednesday, 1 January 1958

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
        cr::tai_time<cr::milliseconds>(1'613'259'090'123ms)); // 23:31:30 TAI Friday, 13 February 2009
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
        cr::tai_seconds(0s)); // 00:00:00 TAI Wednesday, 1 January 1958

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
#else
           "%r='11:31:30 '\t"
#endif
           "%X='23:31:30'\t"
           "%EX='23:31:30'\t"
           "\n"),
        lfmt,
        cr::tai_time<cr::milliseconds>(1'613'259'090'123ms)); // 23:31:30 TAI Friday, 13 February 2009

  // Use supplied locale (ja_JP). This locale has a different alternate.
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
        cr::hh_mm_ss(0s));

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
        cr::hh_mm_ss(23h + 31min + 30s + 123ms));
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
        cr::tai_seconds(0s)); // 00:00:00 TAI Wednesday, 1 January 1958

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
        cr::tai_time<cr::milliseconds>(1'613'259'090'123ms)); // 23:31:30 TAI Friday, 13 February 2009
#endif // defined(__APPLE__) || defined(_AIX) || defined(_WIN32) || defined(__FreeBSD__)

  std::locale::global(std::locale::classic());
}

template <class CharT>
static void test_valid_values_date_time() {
  using namespace std::literals::chrono_literals;
  namespace cr = std::chrono;

  constexpr std::basic_string_view<CharT> fmt  = SV("{:%%c='%c'%t%%Ec='%Ec'%n}");
  constexpr std::basic_string_view<CharT> lfmt = SV("{:L%%c='%c'%t%%Ec='%Ec'%n}");

  const std::locale loc(LOCALE_ja_JP_UTF_8);
  std::locale::global(std::locale(LOCALE_fr_FR_UTF_8));

  // Non localized output using C-locale
  check(SV("%c='Wed Jan  1 00:00:00 1958'\t%Ec='Wed Jan  1 00:00:00 1958'\n"),
        fmt,
        cr::tai_seconds(0s)); // 00:00:00 TAI Wednesday, 1 January 1958

  check(SV("%c='Fri Feb 13 23:31:30 2009'\t%Ec='Fri Feb 13 23:31:30 2009'\n"),
        fmt,
        cr::tai_seconds(1'613'259'090s)); // 23:31:30 TAI Friday, 13 February 2009

  // Use the global locale (fr_FR)
  check(
// https://sourceware.org/bugzilla/show_bug.cgi?id=24054
#if defined(__powerpc__) && defined(__linux__)
      SV("%c='mer. 01 janv. 1958 00:00:00 TAI'\t%Ec='mer. 01 janv. 1958 00:00:00 TAI'\n"),
#elif defined(__GLIBC__) && __GLIBC__ <= 2 && __GLIBC_MINOR__ < 29
      SV("%c='mer. 01 janv. 1958 00:00:00 GMT'\t%Ec='mer. 01 janv. 1958 00:00:00 GMT'\n"),
#elif defined(_AIX)
      SV("%c=' 1 janvier 1958 à 00:00:00 TAI'\t%Ec=' 1 janvier 1958 à 00:00:00 TAI'\n"),
#elif defined(__APPLE__)
      SV("%c='Jeu  1 jan 00:00:00 1958'\t%Ec='Jeu  1 jan 00:00:00 1958'\n"),
#elif defined(_WIN32)
      SV("%c='01/01/1958 00:00:00'\t%Ec='01/01/1958 00:00:00'\n"),
#elif defined(__FreeBSD__)
      SV("%c='mer.  1 janv. 00:00:00 1958'\t%Ec='mer.  1 janv. 00:00:00 1958'\n"),
#else
      SV("%c='mer. 01 janv. 1958 00:00:00'\t%Ec='mer. 01 janv. 1958 00:00:00'\n"),
#endif
      lfmt,
      cr::tai_seconds(0s)); // 00:00:00 TAI Wednesday, 1 January 1958

  check(
// https://sourceware.org/bugzilla/show_bug.cgi?id=24054
#if defined(__powerpc__) && defined(__linux__)
      SV("%c='ven. 13 févr. 2009 23:31:30 TAI'\t%Ec='ven. 13 févr. 2009 23:31:30 TAI'\n"),
#elif defined(__GLIBC__) && __GLIBC__ <= 2 && __GLIBC_MINOR__ < 29
      SV("%c='ven. 13 févr. 2009 23:31:30 GMT'\t%Ec='ven. 13 févr. 2009 23:31:30 GMT'\n"),
#elif defined(_AIX)
      SV("%c='13 février 2009 à 23:31:30 TAI'\t%Ec='13 février 2009 à 23:31:30 TAI'\n"),
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
      cr::tai_seconds(1'613'259'090s)); // 23:31:30 TAI Friday, 13 February 2009

  // Use supplied locale (ja_JP). This locale has a different alternate.a
#if defined(__APPLE__) || defined(__FreeBSD__)
  check(loc,
        SV("%c='水  1/ 1 00:00:00 1958'\t%Ec='水  1/ 1 00:00:00 1958'\n"),
        lfmt,
        cr::tai_seconds(0s)); // 00:00:00 TAI Wednesday, 1 January 1958
  check(loc,
        SV("%c='金  2/13 23:31:30 2009'\t%Ec='金  2/13 23:31:30 2009'\n"),
        lfmt,
        cr::tai_seconds(1'613'259'090s)); // 23:31:30 TAI Friday, 13 February 2009
#elif defined(_AIX)                       // defined(__APPLE__)|| defined(__FreeBSD__)
  check(loc,
        SV("%c='1958年01月 1日 00:00:00 TAI'\t%Ec='1958年01月 1日 00:00:00 TAI'\n"),
        lfmt,
        cr::tai_seconds(0s)); // 00:00:00 TAI Wednesday, 1 January 1958
  check(loc,
        SV("%c='2009年02月13日 23:31:30 TAI'\t%Ec='2009年02月13日 23:31:30 TAI'\n"),
        lfmt,
        cr::tai_seconds(1'613'259'090s)); // 23:31:30 TAI Friday, 13 February 2009
#elif defined(_WIN32)                     // defined(__APPLE__)|| defined(__FreeBSD__)
  check(loc,
        SV("%c='1958/01/01 0:00:00'\t%Ec='1958/01/01 0:00:00'\n"),
        lfmt,
        cr::tai_seconds(0s)); // 00:00:00 TAI Wednesday, 1 January 1958
  check(loc,
        SV("%c='2009/02/13 23:31:30'\t%Ec='2009/02/13 23:31:30'\n"),
        lfmt,
        cr::tai_seconds(1'613'259'090s)); // 23:31:30 TAI Friday, 13 February 2009
#else                                     // defined(__APPLE__)|| defined(__FreeBSD__)
  check(loc,
        SV("%c='1958年01月01日 00時00分00秒'\t%Ec='昭和33年01月01日 00時00分00秒'\n"),
        lfmt,
        cr::tai_seconds(0s)); // 00:00:00 TAI Wednesday, 1 January 1958

  check(loc,
        SV("%c='2009年02月13日 23時31分30秒'\t%Ec='平成21年02月13日 23時31分30秒'\n"),
        lfmt,
        cr::tai_seconds(1'613'259'090s)); // 23:31:30 TAI Friday, 13 February 2009
#endif                                    // defined(__APPLE__)|| defined(__FreeBSD__)

  std::locale::global(std::locale::classic());
}

template <class CharT>
static void test_valid_values_time_zone() {
  using namespace std::literals::chrono_literals;
  namespace cr = std::chrono;

  constexpr std::basic_string_view<CharT> fmt  = SV("{:%%z='%z'%t%%Ez='%Ez'%t%%Oz='%Oz'%t%%Z='%Z'%n}");
  constexpr std::basic_string_view<CharT> lfmt = SV("{:L%%z='%z'%t%%Ez='%Ez'%t%%Oz='%Oz'%t%%Z='%Z'%n}");

  const std::locale loc(LOCALE_ja_JP_UTF_8);
  std::locale::global(std::locale(LOCALE_fr_FR_UTF_8));

  // Non localized output using C-locale
  check(SV("%z='+0000'\t%Ez='+00:00'\t%Oz='+00:00'\t%Z='TAI'\n"),
        fmt,
        cr::tai_seconds(0s)); // 00:00:00 TAI Wednesday, 1 January 1958

  // Use the global locale (fr_FR)
  check(SV("%z='+0000'\t%Ez='+00:00'\t%Oz='+00:00'\t%Z='TAI'\n"),
        lfmt,
        cr::tai_seconds(0s)); // 00:00:00 TAI Wednesday, 1 January 1958

  // Use supplied locale (ja_JP).
  check(loc,
        SV("%z='+0000'\t%Ez='+00:00'\t%Oz='+00:00'\t%Z='TAI'\n"),
        lfmt,
        cr::tai_seconds(0s)); // 00:00:00 TAI Wednesday, 1 January 1958

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
  using namespace std::literals::chrono_literals;
  namespace cr = std::chrono;

  test_no_chrono_specs<CharT>();
  test_valid_values<CharT>();
  check_invalid_types<CharT>(
      {SV("a"),  SV("A"),  SV("b"),  SV("B"),  SV("c"),  SV("C"),  SV("d"),  SV("D"),  SV("e"),  SV("F"),  SV("g"),
       SV("G"),  SV("h"),  SV("H"),  SV("I"),  SV("j"),  SV("m"),  SV("M"),  SV("p"),  SV("r"),  SV("R"),  SV("S"),
       SV("T"),  SV("u"),  SV("U"),  SV("V"),  SV("w"),  SV("W"),  SV("x"),  SV("X"),  SV("y"),  SV("Y"),  SV("z"),
       SV("Z"),  SV("Ec"), SV("EC"), SV("Ex"), SV("EX"), SV("Ey"), SV("EY"), SV("Ez"), SV("Od"), SV("Oe"), SV("OH"),
       SV("OI"), SV("Om"), SV("OM"), SV("OS"), SV("Ou"), SV("OU"), SV("OV"), SV("Ow"), SV("OW"), SV("Oy"), SV("Oz")},
      cr::tai_seconds(0s));

  check_exception("The format specifier expects a '%' or a '}'", SV("{:A"), cr::tai_seconds(0s));
  check_exception("The chrono specifiers contain a '{'", SV("{:%%{"), cr::tai_seconds(0s));
  check_exception("End of input while parsing a conversion specifier", SV("{:%"), cr::tai_seconds(0s));
  check_exception("End of input while parsing the modifier E", SV("{:%E"), cr::tai_seconds(0s));
  check_exception("End of input while parsing the modifier O", SV("{:%O"), cr::tai_seconds(0s));

  // Precision not allowed
  check_exception("The format specifier expects a '%' or a '}'", SV("{:.3}"), cr::tai_seconds(0s));
}

int main(int, char**) {
  test<char>();

#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  test<wchar_t>();
#endif

  return 0;
}
