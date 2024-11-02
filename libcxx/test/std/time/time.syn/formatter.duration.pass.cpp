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

//  template<class Rep, class Period, class charT>
//    struct formatter<chrono::duration<Rep, Period>, charT>;

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

  check(SV("1as"), SV("{}"), std::chrono::duration<int, std::atto>(1));
  check(SV("1fs"), SV("{}"), std::chrono::duration<int, std::femto>(1));
  check(SV("1ps"), SV("{}"), std::chrono::duration<int, std::pico>(1));
  check(SV("1ns"), SV("{}"), 1ns);
#ifndef TEST_HAS_NO_UNICODE
  check(SV("1\u00b5s"), SV("{}"), 1us);
#else
  check(SV("1us"), SV("{}"), 1us);
#endif
  check(SV("1ms"), SV("{}"), 1ms);
  check(SV("1cs"), SV("{}"), std::chrono::duration<int, std::centi>(1));
  check(SV("1ds"), SV("{}"), std::chrono::duration<int, std::deci>(1));

  check(SV("1s"), SV("{}"), 1s);

  check(SV("1das"), SV("{}"), std::chrono::duration<int, std::deca>(1));
  check(SV("1hs"), SV("{}"), std::chrono::duration<int, std::hecto>(1));
  check(SV("1ks"), SV("{}"), std::chrono::duration<int, std::kilo>(1));
  check(SV("1Ms"), SV("{}"), std::chrono::duration<int, std::mega>(1));
  check(SV("1Gs"), SV("{}"), std::chrono::duration<int, std::giga>(1));
  check(SV("1Ts"), SV("{}"), std::chrono::duration<int, std::tera>(1));
  check(SV("1Ps"), SV("{}"), std::chrono::duration<int, std::peta>(1));
  check(SV("1Es"), SV("{}"), std::chrono::duration<int, std::exa>(1));

  check(SV("1min"), SV("{}"), 1min);
  check(SV("1h"), SV("{}"), 1h);
  check(SV("1d"), SV("{}"), std::chrono::duration<int, std::ratio<86400>>(1));

  check(SV("1[42]s"), SV("{}"), std::chrono::duration<int, std::ratio<42>>(1));
  check(SV("1[11]s"), SV("{}"), std::chrono::duration<int, std::ratio<33, 3>>(1));
  check(SV("1[11/9]s"), SV("{}"), std::chrono::duration<int, std::ratio<11, 9>>(1));
}

template <class CharT>
static void test_valid_positive_integral_values() {
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
      "%%j='%j'%t"
      "%%Q='%Q'%t"
      "%%q='%q'%t"
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
      "%%j='%j'%t"
      "%%Q='%Q'%t"
      "%%q='%q'%t"
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
           "%j='0'\t"
           "%Q='0'\t"
           "%q='s'\t"
           "\n"),
        fmt,
        0s);

  check(SV("%H='11'\t"
           "%OH='11'\t"
           "%I='11'\t"
           "%OI='11'\t"
           "%M='59'\t"
           "%OM='59'\t"
           "%S='59'\t"
           "%OS='59'\t"
           "%p='AM'\t"
           "%R='11:59'\t"
           "%T='11:59:59'\t"
           "%r='11:59:59 AM'\t"
           "%X='11:59:59'\t"
           "%EX='11:59:59'\t"
           "%j='0'\t"
           "%Q='43199'\t"
           "%q='s'\t"
           "\n"),
        fmt,
        11h + 59min + 59s);

  check(SV("%H='12'\t"
           "%OH='12'\t"
           "%I='12'\t"
           "%OI='12'\t"
           "%M='00'\t"
           "%OM='00'\t"
           "%S='00'\t"
           "%OS='00'\t"
           "%p='PM'\t"
           "%R='12:00'\t"
           "%T='12:00:00'\t"
           "%r='12:00:00 PM'\t"
           "%X='12:00:00'\t"
           "%EX='12:00:00'\t"
           "%j='0'\t"
           "%Q='12'\t"
           "%q='h'\t"
           "\n"),
        fmt,
        12h);

  check(SV("%H='23'\t"
           "%OH='23'\t"
           "%I='11'\t"
           "%OI='11'\t"
           "%M='59'\t"
           "%OM='59'\t"
           "%S='59'\t"
           "%OS='59'\t"
           "%p='PM'\t"
           "%R='23:59'\t"
           "%T='23:59:59'\t"
           "%r='11:59:59 PM'\t"
           "%X='23:59:59'\t"
           "%EX='23:59:59'\t"
           "%j='0'\t"
           "%Q='86399'\t"
           "%q='s'\t"
           "\n"),
        fmt,
        23h + 59min + 59s);

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
           "%j='7'\t"
           "%Q='7'\t"
           "%q='d'\t"
           "\n"),
        fmt,
        std::chrono::duration<int, std::ratio<86400>>(7));

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
           "%r='12:00:00'\t"
#elif defined(_AIX)
           "%r='12:00:00 AM'\t"
#elif defined(__APPLE__)
           "%r=''\t"
#else
           "%r='12:00:00 '\t"
#endif
           "%X='00:00:00'\t"
           "%EX='00:00:00'\t"
           "%j='0'\t"
           "%Q='0'\t"
           "%q='s'\t"
           "\n"),
        lfmt,
        0s);

  check(SV("%H='11'\t"
           "%OH='11'\t"
           "%I='11'\t"
           "%OI='11'\t"
           "%M='59'\t"
           "%OM='59'\t"
           "%S='59'\t"
           "%OS='59'\t"
#if defined(_AIX)
           "%p='AM'\t"
#else
           "%p=''\t"
#endif
           "%R='11:59'\t"
           "%T='11:59:59'\t"
#ifdef _WIN32
           "%r='11:59:59'\t"
#elif defined(_AIX)
           "%r='11:59:59 AM'\t"
#elif defined(__APPLE__)
           "%r=''\t"
#else
           "%r='11:59:59 '\t"
#endif
           "%X='11:59:59'\t"
           "%EX='11:59:59'\t"
           "%j='0'\t"
           "%Q='43199'\t"
           "%q='s'\t"
           "\n"),
        lfmt,
        11h + 59min + 59s);

  check(SV("%H='12'\t"
           "%OH='12'\t"
           "%I='12'\t"
           "%OI='12'\t"
           "%M='00'\t"
           "%OM='00'\t"
           "%S='00'\t"
           "%OS='00'\t"
#if defined(_AIX)
           "%p='PM'\t"
#else
           "%p=''\t"
#endif
           "%R='12:00'\t"
           "%T='12:00:00'\t"
#ifdef _WIN32
           "%r='00:00:00'\t"
#elif defined(_AIX)
           "%r='12:00:00 PM'\t"
#elif defined(__APPLE__)
           "%r=''\t"
#else
           "%r='12:00:00 '\t"
#endif
           "%X='12:00:00'\t"
           "%EX='12:00:00'\t"
           "%j='0'\t"
           "%Q='12'\t"
           "%q='h'\t"
           "\n"),
        lfmt,
        12h);

  check(SV("%H='23'\t"
           "%OH='23'\t"
           "%I='11'\t"
           "%OI='11'\t"
           "%M='59'\t"
           "%OM='59'\t"
           "%S='59'\t"
           "%OS='59'\t"
#if defined(_AIX)
           "%p='PM'\t"
#else
           "%p=''\t"
#endif
           "%R='23:59'\t"
           "%T='23:59:59'\t"
#if defined(_AIX)
           "%r='11:59:59 PM'\t"
#elif defined(__APPLE__)
           "%r=''\t"
#else
           "%r='11:59:59 '\t"
#endif
           "%X='23:59:59'\t"
           "%EX='23:59:59'\t"
           "%j='0'\t"
           "%Q='86399'\t"
           "%q='s'\t"
           "\n"),
        lfmt,
        23h + 59min + 59s);

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
           "%r='12:00:00'\t"
#elif defined(_AIX)
           "%r='12:00:00 AM'\t"
#elif defined(__APPLE__)
           "%r=''\t"
#else
           "%r='12:00:00 '\t"
#endif
           "%X='00:00:00'\t"
           "%EX='00:00:00'\t"
           "%j='7'\t"
           "%Q='7'\t"
           "%q='d'\t"
           "\n"),
        lfmt,
        std::chrono::duration<int, std::ratio<86400>>(7));

  // Use supplied locale (ja_JP). This locale has a different alternate.
#if defined(__APPLE__) || defined(_AIX)
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
#  if defined(__APPLE__)
           "%r='12:00:00 AM'\t"
           "%X='00時00分00秒'\t"
           "%EX='00時00分00秒'\t"
#  else
           "%r='午前12:00:00'\t"
           "%X='00:00:00'\t"
           "%EX='00:00:00'\t"
#  endif
           "%j='0'\t"
           "%Q='0'\t"
           "%q='s'\t"
           "\n"),
        lfmt,
        0s);

  check(loc,
        SV("%H='11'\t"
           "%OH='11'\t"
           "%I='11'\t"
           "%OI='11'\t"
           "%M='59'\t"
           "%OM='59'\t"
           "%S='59'\t"
           "%OS='59'\t"
#  if defined(__APPLE__)
           "%p='AM'\t"
#  else
           "%p='午前'\t"
#  endif
           "%R='11:59'\t"
           "%T='11:59:59'\t"
#  if defined(__APPLE__)
           "%r='11:59:59 AM'\t"
           "%X='11時59分59秒'\t"
           "%EX='11時59分59秒'\t"
#  else
           "%r='午前11:59:59'\t"
           "%X='11:59:59'\t"
           "%EX='11:59:59'\t"
#  endif
           "%j='0'\t"
           "%Q='43199'\t"
           "%q='s'\t"
           "\n"),
        lfmt,
        11h + 59min + 59s);

  check(loc,
        SV("%H='12'\t"
           "%OH='12'\t"
           "%I='12'\t"
           "%OI='12'\t"
           "%M='00'\t"
           "%OM='00'\t"
           "%S='00'\t"
           "%OS='00'\t"
#  if defined(__APPLE__)
           "%p='PM'\t"
#  else
           "%p='午後'\t"
#  endif
           "%R='12:00'\t"
           "%T='12:00:00'\t"
#  if defined(__APPLE__)
           "%r='12:00:00 PM'\t"
           "%X='12時00分00秒'\t"
           "%EX='12時00分00秒'\t"
#  else
           "%r='午後12:00:00'\t"
           "%X='12:00:00'\t"
           "%EX='12:00:00'\t"
#  endif
           "%j='0'\t"
           "%Q='12'\t"
           "%q='h'\t"
           "\n"),
        lfmt,
        12h);

  check(loc,
        SV("%H='23'\t"
           "%OH='23'\t"
           "%I='11'\t"
           "%OI='11'\t"
           "%M='59'\t"
           "%OM='59'\t"
           "%S='59'\t"
           "%OS='59'\t"
#  if defined(__APPLE__)
           "%p='PM'\t"
#  else
           "%p='午後'\t"
#  endif
           "%R='23:59'\t"
           "%T='23:59:59'\t"
#  if defined(__APPLE__)
           "%r='11:59:59 PM'\t"
           "%X='23時59分59秒'\t"
           "%EX='23時59分59秒'\t"
#  else
           "%r='午後11:59:59'\t"
           "%X='23:59:59'\t"
           "%EX='23:59:59'\t"
#  endif
           "%j='0'\t"
           "%Q='86399'\t"
           "%q='s'\t"
           "\n"),
        lfmt,
        23h + 59min + 59s);

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
#  if defined(__APPLE__)
           "%r='12:00:00 AM'\t"
           "%X='00時00分00秒'\t"
           "%EX='00時00分00秒'\t"
#  else
           "%r='午前12:00:00'\t"
           "%X='00:00:00'\t"
           "%EX='00:00:00'\t"
#  endif
           "%j='7'\t"
           "%Q='7'\t"
           "%q='d'\t"
           "\n"),
        lfmt,
        std::chrono::duration<int, std::ratio<86400>>(7));
#else // defined(__APPLE__) || defined(_AIX)
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
           "%j='0'\t"
           "%Q='0'\t"
           "%q='s'\t"
           "\n"),
        lfmt,
        0s);

  check(loc,
        SV("%H='11'\t"
           "%OH='十一'\t"
           "%I='11'\t"
           "%OI='十一'\t"
           "%M='59'\t"
           "%OM='五十九'\t"
           "%S='59'\t"
           "%OS='五十九'\t"
           "%p='午前'\t"
           "%R='11:59'\t"
           "%T='11:59:59'\t"
           "%r='午前11時59分59秒'\t"
           "%X='11時59分59秒'\t"
           "%EX='11時59分59秒'\t"
           "%j='0'\t"
           "%Q='43199'\t"
           "%q='s'\t"
           "\n"),
        lfmt,
        11h + 59min + 59s);

  check(loc,
        SV("%H='12'\t"
           "%OH='十二'\t"
           "%I='12'\t"
           "%OI='十二'\t"
           "%M='00'\t"
           "%OM='〇'\t"
           "%S='00'\t"
           "%OS='〇'\t"
           "%p='午後'\t"
           "%R='12:00'\t"
           "%T='12:00:00'\t"
           "%r='午後12時00分00秒'\t"
           "%X='12時00分00秒'\t"
           "%EX='12時00分00秒'\t"
           "%j='0'\t"
           "%Q='12'\t"
           "%q='h'\t"
           "\n"),
        lfmt,
        12h);

  check(loc,
        SV("%H='23'\t"
           "%OH='二十三'\t"
           "%I='11'\t"
           "%OI='十一'\t"
           "%M='59'\t"
           "%OM='五十九'\t"
           "%S='59'\t"
           "%OS='五十九'\t"
           "%p='午後'\t"
           "%R='23:59'\t"
           "%T='23:59:59'\t"
           "%r='午後11時59分59秒'\t"
           "%X='23時59分59秒'\t"
           "%EX='23時59分59秒'\t"
           "%j='0'\t"
           "%Q='86399'\t"
           "%q='s'\t"
           "\n"),
        lfmt,
        23h + 59min + 59s);

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
           "%j='7'\t"
           "%Q='7'\t"
           "%q='d'\t"
           "\n"),
        lfmt,
        std::chrono::duration<int, std::ratio<86400>>(7));

#endif // defined(__APPLE__) || defined(_AIX)
  std::locale::global(std::locale::classic());
}

template <class CharT>
static void test_valid_negative_integral_values() {
  // [time.format]/4 The result of formatting a std::chrono::duration instance
  // holding a negative value, or an hh_mm_ss object h for which
  // h.is_negative() is true, is equivalent to the output of the corresponding
  // positive value, with a STATICALLY-WIDEN<charT>("-") character sequence
  // placed before the replacement of the initial conversion specifier.
  //
  // Note in this case %% is the initial conversion specifier.
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
      "%%j='%j'%t"
      "%%Q='%Q'%t"
      "%%q='%q'%t"
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
      "%%j='%j'%t"
      "%%Q='%Q'%t"
      "%%q='%q'%t"
      "%n}");

  const std::locale loc(LOCALE_ja_JP_UTF_8);
  std::locale::global(std::locale(LOCALE_fr_FR_UTF_8));

  // Non localized output using C-locale
  check(SV("-%H='23'\t"
           "%OH='23'\t"
           "%I='11'\t"
           "%OI='11'\t"
           "%M='59'\t"
           "%OM='59'\t"
           "%S='59'\t"
           "%OS='59'\t"
           "%p='PM'\t"
           "%R='23:59'\t"
           "%T='23:59:59'\t"
           "%r='11:59:59 PM'\t"
           "%X='23:59:59'\t"
           "%EX='23:59:59'\t"
           "%j='0'\t"
           "%Q='86399'\t"
           "%q='s'\t"
           "\n"),
        fmt,
        -(23h + 59min + 59s));

  // Use the global locale (fr_FR)
  check(SV("-%H='23'\t"
           "%OH='23'\t"
           "%I='11'\t"
           "%OI='11'\t"
           "%M='59'\t"
           "%OM='59'\t"
           "%S='59'\t"
           "%OS='59'\t"
#if defined(_AIX)
           "%p='PM'\t"
#else
           "%p=''\t"
#endif
           "%R='23:59'\t"
           "%T='23:59:59'\t"
#if defined(_AIX)
           "%r='11:59:59 PM'\t"
#elif defined(__APPLE__)
           "%r=''\t"
#else
           "%r='11:59:59 '\t"
#endif
           "%X='23:59:59'\t"
           "%EX='23:59:59'\t"
           "%j='0'\t"
           "%Q='86399'\t"
           "%q='s'\t"
           "\n"),
        lfmt,
        -(23h + 59min + 59s));

  // Use supplied locale (ja_JP). This locale has a different alternate.
#if defined(__APPLE__) || defined(_AIX)
  check(loc,
        SV("-%H='23'\t"
           "%OH='23'\t"
           "%I='11'\t"
           "%OI='11'\t"
           "%M='59'\t"
           "%OM='59'\t"
           "%S='59'\t"
           "%OS='59'\t"
#  if defined(__APPLE__)
           "%p='PM'\t"
#  else
           "%p='午後'\t"
#  endif
           "%R='23:59'\t"
           "%T='23:59:59'\t"
#  if defined(__APPLE__)
           "%r='11:59:59 PM'\t"
           "%X='23時59分59秒'\t"
           "%EX='23時59分59秒'\t"
#  else
           "%r='午後11:59:59'\t"
           "%X='23:59:59'\t"
           "%EX='23:59:59'\t"
#  endif
           "%j='0'\t"
           "%Q='86399'\t"
           "%q='s'\t"
           "\n"),
        lfmt,
        -(23h + 59min + 59s));
#else  // defined(__APPLE__) || defined(_AIX)
  check(loc,
        SV("-%H='23'\t"
           "%OH='二十三'\t"
           "%I='11'\t"
           "%OI='十一'\t"
           "%M='59'\t"
           "%OM='五十九'\t"
           "%S='59'\t"
           "%OS='五十九'\t"
           "%p='午後'\t"
           "%R='23:59'\t"
           "%T='23:59:59'\t"
           "%r='午後11時59分59秒'\t"
           "%X='23時59分59秒'\t"
           "%EX='23時59分59秒'\t"
           "%j='0'\t"
           "%Q='86399'\t"
           "%q='s'\t"
           "\n"),
        lfmt,
        -(23h + 59min + 59s));
#endif // defined(__APPLE__) || defined(_AIX)
  std::locale::global(std::locale::classic());
}

template <class CharT>
static void test_valid_fractional_values() {
  using namespace std::literals::chrono_literals;

  const std::locale loc(LOCALE_ja_JP_UTF_8);
  std::locale::global(std::locale(LOCALE_fr_FR_UTF_8));

  // Non localized output using C-locale
  check(SV("00.000000001"), SV("{:%S}"), 1ns);
  check(SV("00.000000501"), SV("{:%S}"), 501ns);
  check(SV("00.000001000"), SV("{:%S}"), 1000ns);
  check(SV("00.000000000001"), SV("{:%S}"), std::chrono::duration<int, std::pico>(1));
  check(SV("00.000000000000001"), SV("{:%S}"), std::chrono::duration<int, std::femto>(1));
  check(SV("00.000000000000000001"), SV("{:%S}"), std::chrono::duration<int, std::atto>(1));

  check(SV("00.001"), SV("{:%S}"), 1ms);
  check(SV("00.01"), SV("{:%S}"), std::chrono::duration<int, std::centi>(1));
  check(SV("00.1"), SV("{:%S}"), std::chrono::duration<int, std::deci>(1));
  check(SV("01.1"), SV("{:%S}"), std::chrono::duration<int, std::deci>(11));

  check(SV("00.001"), SV("{:%S}"), std::chrono::duration<float, std::milli>(1.123456789));
  check(SV("00.011"), SV("{:%S}"), std::chrono::duration<double, std::milli>(11.123456789));
  check(SV("01"), SV("{:%S}"), std::chrono::duration<long double>(61.123456789));

  check(SV("00.000000001"), SV("{:%OS}"), 1ns);
  check(SV("00.000000501"), SV("{:%OS}"), 501ns);
  check(SV("00.000001000"), SV("{:%OS}"), 1000ns);
  check(SV("00.000000000001"), SV("{:%OS}"), std::chrono::duration<int, std::pico>(1));
  check(SV("00.000000000000001"), SV("{:%OS}"), std::chrono::duration<int, std::femto>(1));
  check(SV("00.000000000000000001"), SV("{:%OS}"), std::chrono::duration<int, std::atto>(1));

  check(SV("00.001"), SV("{:%OS}"), 1ms);
  check(SV("00.01"), SV("{:%OS}"), std::chrono::duration<int, std::centi>(1));
  check(SV("00.1"), SV("{:%OS}"), std::chrono::duration<int, std::deci>(1));
  check(SV("01.1"), SV("{:%OS}"), std::chrono::duration<int, std::deci>(11));

  check(SV("00.001"), SV("{:%OS}"), std::chrono::duration<float, std::milli>(1.123456789));
  check(SV("00.011"), SV("{:%OS}"), std::chrono::duration<double, std::milli>(11.123456789));
  check(SV("01"), SV("{:%OS}"), std::chrono::duration<long double>(61.123456789));

  check(SV("01:05:06.000000001"), SV("{:%T}"), 1h + 5min + 6s + 1ns);
  check(SV("01:05:06.000000501"), SV("{:%T}"), 1h + 5min + 6s + 501ns);
  check(SV("01:05:06.000001000"), SV("{:%T}"), 1h + 5min + 6s + 1000ns);

  check(SV("01:05:06.001"), SV("{:%T}"), 1h + 5min + 6s + 1ms);
  check(SV("01:05:06.01"), SV("{:%T}"), 1h + 5min + 6s + std::chrono::duration<int, std::centi>(1));
  check(SV("01:05:06.1"), SV("{:%T}"), 1h + 5min + 6s + std::chrono::duration<int, std::deci>(1));
  check(SV("01:05:07.1"), SV("{:%T}"), 1h + 5min + 6s + std::chrono::duration<int, std::deci>(11));

  check(
      SV("00:01:02"), SV("{:%T}"), std::chrono::duration<float, std::ratio<60>>(1) + std::chrono::duration<float>(2.5));
  check(SV("01:05:11"),
        SV("{:%T}"),
        std::chrono::duration<double, std::ratio<3600>>(1) + std::chrono::duration<double, std::ratio<60>>(5) +
            std::chrono::duration<double>(11.123456789));
  check(SV("01:06:01"),
        SV("{:%T}"),
        std::chrono::duration<long double, std::ratio<3600>>(1) +
            std::chrono::duration<long double, std::ratio<60>>(5) + std::chrono::duration<long double>(61.123456789));

  check(SV("0"), SV("{:%j}"), std::chrono::duration<float, std::milli>(1.));
  check(SV("1"), SV("{:%j}"), std::chrono::duration<double, std::milli>(86'400'000));
  check(SV("1"), SV("{:%j}"), std::chrono::duration<long double, std::ratio<7 * 24 * 3600>>(0.14285714286));

  check(SV("1000000"), SV("{:%Q}"), 1'000'000s);
  check(SV("1"), SV("{:%Q}"), std::chrono::duration<float, std::milli>(1.));
  check(SV("1.123456789"), SV("{:.6%Q}"), std::chrono::duration<double, std::milli>(1.123456789));
  check(SV("1.123456789"), SV("{:.9%Q}"), std::chrono::duration<long double, std::milli>(1.123456789));

  // Use the global locale (fr_FR)
  check(SV("00,000000001"), SV("{:L%S}"), 1ns);
  check(SV("00,000000501"), SV("{:L%S}"), 501ns);
  check(SV("00,000001000"), SV("{:L%S}"), 1000ns);
  check(SV("00,000000000001"), SV("{:L%S}"), std::chrono::duration<int, std::pico>(1));
  check(SV("00,000000000000001"), SV("{:L%S}"), std::chrono::duration<int, std::femto>(1));
  check(SV("00,000000000000000001"), SV("{:L%S}"), std::chrono::duration<int, std::atto>(1));

  check(SV("00,001"), SV("{:L%S}"), 1ms);
  check(SV("00,01"), SV("{:L%S}"), std::chrono::duration<int, std::centi>(1));
  check(SV("00,1"), SV("{:L%S}"), std::chrono::duration<int, std::deci>(1));
  check(SV("01,1"), SV("{:L%S}"), std::chrono::duration<int, std::deci>(11));

  check(SV("00,001"), SV("{:L%S}"), std::chrono::duration<float, std::milli>(1.123456789));
  check(SV("00,011"), SV("{:L%S}"), std::chrono::duration<double, std::milli>(11.123456789));
  check(SV("01"), SV("{:L%S}"), std::chrono::duration<long double>(61.123456789));

  check(SV("00,000000001"), SV("{:L%OS}"), 1ns);
  check(SV("00,000000501"), SV("{:L%OS}"), 501ns);
  check(SV("00,000001000"), SV("{:L%OS}"), 1000ns);
  check(SV("00,000000000001"), SV("{:L%OS}"), std::chrono::duration<int, std::pico>(1));
  check(SV("00,000000000000001"), SV("{:L%OS}"), std::chrono::duration<int, std::femto>(1));
  check(SV("00,000000000000000001"), SV("{:L%OS}"), std::chrono::duration<int, std::atto>(1));

  check(SV("00,001"), SV("{:L%OS}"), 1ms);
  check(SV("00,01"), SV("{:L%OS}"), std::chrono::duration<int, std::centi>(1));
  check(SV("00,1"), SV("{:L%OS}"), std::chrono::duration<int, std::deci>(1));
  check(SV("01,1"), SV("{:L%OS}"), std::chrono::duration<int, std::deci>(11));

  check(SV("00,001"), SV("{:L%OS}"), std::chrono::duration<float, std::milli>(1.123456789));
  check(SV("00,011"), SV("{:L%OS}"), std::chrono::duration<double, std::milli>(11.123456789));
  check(SV("01"), SV("{:L%OS}"), std::chrono::duration<long double>(61.123456789));

  check(SV("01:05:06,000000001"), SV("{:L%T}"), 1h + 5min + 6s + 1ns);
  check(SV("01:05:06,000000501"), SV("{:L%T}"), 1h + 5min + 6s + 501ns);
  check(SV("01:05:06,000001000"), SV("{:L%T}"), 1h + 5min + 6s + 1000ns);

  check(SV("01:05:06,001"), SV("{:L%T}"), 1h + 5min + 6s + 1ms);
  check(SV("01:05:06,01"), SV("{:L%T}"), 1h + 5min + 6s + std::chrono::duration<int, std::centi>(1));
  check(SV("01:05:06,1"), SV("{:L%T}"), 1h + 5min + 6s + std::chrono::duration<int, std::deci>(1));
  check(SV("01:05:07,1"), SV("{:L%T}"), 1h + 5min + 6s + std::chrono::duration<int, std::deci>(11));

  check(SV("00:01:02"),
        SV("{:L%T}"),
        std::chrono::duration<float, std::ratio<60>>(1) + std::chrono::duration<float>(2.5));
  check(SV("01:05:11"),
        SV("{:L%T}"),
        std::chrono::duration<double, std::ratio<3600>>(1) + std::chrono::duration<double, std::ratio<60>>(5) +
            std::chrono::duration<double>(11.123456789));
  check(SV("01:06:01"),
        SV("{:L%T}"),
        std::chrono::duration<long double, std::ratio<3600>>(1) +
            std::chrono::duration<long double, std::ratio<60>>(5) + std::chrono::duration<long double>(61.123456789));

  check(SV("0"), SV("{:L%j}"), std::chrono::duration<float, std::milli>(1.));
  check(SV("1"), SV("{:L%j}"), std::chrono::duration<double, std::milli>(86'400'000));
  check(SV("1"), SV("{:L%j}"), std::chrono::duration<long double, std::ratio<7 * 24 * 3600>>(0.14285714286));

  check(SV("1000000"), SV("{:L%Q}"), 1'000'000s); // The Standard mandates not localized.
  check(SV("1"), SV("{:L%Q}"), std::chrono::duration<float, std::milli>(1.));
  check(SV("1.123456789"), SV("{:.6L%Q}"), std::chrono::duration<double, std::milli>(1.123456789));
  check(SV("1.123456789"), SV("{:.9L%Q}"), std::chrono::duration<long double, std::milli>(1.123456789));

  // Use supplied locale (ja_JP). This locale has a different alternate.
  check(loc, SV("00.000000001"), SV("{:L%S}"), 1ns);
  check(loc, SV("00.000000501"), SV("{:L%S}"), 501ns);
  check(loc, SV("00.000001000"), SV("{:L%S}"), 1000ns);
  check(loc, SV("00.000000000001"), SV("{:L%S}"), std::chrono::duration<int, std::pico>(1));
  check(loc, SV("00.000000000000001"), SV("{:L%S}"), std::chrono::duration<int, std::femto>(1));
  check(loc, SV("00.000000000000000001"), SV("{:L%S}"), std::chrono::duration<int, std::atto>(1));

  check(loc, SV("00.001"), SV("{:L%S}"), 1ms);
  check(loc, SV("00.01"), SV("{:L%S}"), std::chrono::duration<int, std::centi>(1));
  check(loc, SV("00.1"), SV("{:L%S}"), std::chrono::duration<int, std::deci>(1));
  check(loc, SV("01.1"), SV("{:L%S}"), std::chrono::duration<int, std::deci>(11));

  check(loc, SV("00.001"), SV("{:L%S}"), std::chrono::duration<float, std::milli>(1.123456789));
  check(loc, SV("00.011"), SV("{:L%S}"), std::chrono::duration<double, std::milli>(11.123456789));
  check(loc, SV("01"), SV("{:L%S}"), std::chrono::duration<long double>(61.123456789));

#if defined(__APPLE__) || defined(_AIX)
  check(SV("00.000000001"), SV("{:%OS}"), 1ns);
  check(SV("00.000000501"), SV("{:%OS}"), 501ns);
  check(SV("00.000001000"), SV("{:%OS}"), 1000ns);
  check(SV("00.000000000001"), SV("{:%OS}"), std::chrono::duration<int, std::pico>(1));
  check(SV("00.000000000000001"), SV("{:%OS}"), std::chrono::duration<int, std::femto>(1));
  check(SV("00.000000000000000001"), SV("{:%OS}"), std::chrono::duration<int, std::atto>(1));

  check(SV("00.001"), SV("{:%OS}"), 1ms);
  check(SV("00.01"), SV("{:%OS}"), std::chrono::duration<int, std::centi>(1));
  check(SV("00.1"), SV("{:%OS}"), std::chrono::duration<int, std::deci>(1));
  check(SV("01.1"), SV("{:%OS}"), std::chrono::duration<int, std::deci>(11));

  check(SV("00.001"), SV("{:%OS}"), std::chrono::duration<float, std::milli>(1.123456789));
  check(SV("00.011"), SV("{:%OS}"), std::chrono::duration<double, std::milli>(11.123456789));
  check(SV("01"), SV("{:%OS}"), std::chrono::duration<long double>(61.123456789));
#else  // defined(__APPLE__) || defined(_AIX)
  check(loc, SV("〇.000000001"), SV("{:L%OS}"), 1ns);
  check(loc, SV("〇.000000501"), SV("{:L%OS}"), 501ns);
  check(loc, SV("〇.000001000"), SV("{:L%OS}"), 1000ns);
  check(loc, SV("〇.000000000001"), SV("{:L%OS}"), std::chrono::duration<int, std::pico>(1));
  check(loc, SV("〇.000000000000001"), SV("{:L%OS}"), std::chrono::duration<int, std::femto>(1));
  check(loc, SV("〇.000000000000000001"), SV("{:L%OS}"), std::chrono::duration<int, std::atto>(1));

  check(loc, SV("〇.001"), SV("{:L%OS}"), 1ms);
  check(loc, SV("〇.01"), SV("{:L%OS}"), std::chrono::duration<int, std::centi>(1));
  check(loc, SV("〇.1"), SV("{:L%OS}"), std::chrono::duration<int, std::deci>(1));
  check(loc, SV("一.1"), SV("{:L%OS}"), std::chrono::duration<int, std::deci>(11));

  check(loc, SV("〇.001"), SV("{:L%OS}"), std::chrono::duration<float, std::milli>(1.123456789));
  check(loc, SV("〇.011"), SV("{:L%OS}"), std::chrono::duration<double, std::milli>(11.123456789));
  check(loc, SV("一"), SV("{:L%OS}"), std::chrono::duration<long double>(61.123456789));
#endif // defined(__APPLE__) || defined(_AIX)

  check(loc, SV("01:05:06.000000001"), SV("{:L%T}"), 1h + 5min + 6s + 1ns);
  check(loc, SV("01:05:06.000000501"), SV("{:L%T}"), 1h + 5min + 6s + 501ns);
  check(loc, SV("01:05:06.000001000"), SV("{:L%T}"), 1h + 5min + 6s + 1000ns);

  check(loc, SV("01:05:06.001"), SV("{:L%T}"), 1h + 5min + 6s + 1ms);
  check(loc, SV("01:05:06.01"), SV("{:L%T}"), 1h + 5min + 6s + std::chrono::duration<int, std::centi>(1));
  check(loc, SV("01:05:06.1"), SV("{:L%T}"), 1h + 5min + 6s + std::chrono::duration<int, std::deci>(1));
  check(loc, SV("01:05:07.1"), SV("{:L%T}"), 1h + 5min + 6s + std::chrono::duration<int, std::deci>(11));

  check(loc,
        SV("00:01:02"),
        SV("{:L%T}"),
        std::chrono::duration<float, std::ratio<60>>(1) + std::chrono::duration<float>(2.5));
  check(loc,
        SV("01:05:11"),
        SV("{:L%T}"),
        std::chrono::duration<double, std::ratio<3600>>(1) + std::chrono::duration<double, std::ratio<60>>(5) +
            std::chrono::duration<double>(11.123456789));
  check(loc,
        SV("01:06:01"),
        SV("{:L%T}"),
        std::chrono::duration<long double, std::ratio<3600>>(1) +
            std::chrono::duration<long double, std::ratio<60>>(5) + std::chrono::duration<long double>(61.123456789));

  check(loc, SV("0"), SV("{:L%j}"), std::chrono::duration<float, std::milli>(1.));
  check(loc, SV("1"), SV("{:L%j}"), std::chrono::duration<double, std::milli>(86'400'000));
  check(loc, SV("1"), SV("{:L%j}"), std::chrono::duration<long double, std::ratio<7 * 24 * 3600>>(0.14285714286));

  check(loc, SV("1000000"), SV("{:L%Q}"), 1'000'000s); // The Standard mandates not localized.
  check(loc, SV("1"), SV("{:L%Q}"), std::chrono::duration<float, std::milli>(1.));
  check(loc, SV("1.123456789"), SV("{:.6L%Q}"), std::chrono::duration<double, std::milli>(1.123456789));
  check(loc, SV("1.123456789"), SV("{:.9L%Q}"), std::chrono::duration<long double, std::milli>(1.123456789));

  std::locale::global(std::locale::classic());
}

template <class CharT>
static void test_valid_values() {
  test_valid_positive_integral_values<CharT>();
  test_valid_negative_integral_values<CharT>();
  test_valid_fractional_values<CharT>();
}

template <class CharT>
static void test() {
  using namespace std::literals::chrono_literals;

  test_no_chrono_specs<CharT>();
  test_valid_values<CharT>();
  check_invalid_types<CharT>(
      {SV("H"), SV("I"), SV("j"), SV("M"), SV("n"), SV("O"),  SV("p"),  SV("q"),  SV("Q"),  SV("r"),
       SV("R"), SV("S"), SV("t"), SV("T"), SV("X"), SV("EX"), SV("OH"), SV("OI"), SV("OM"), SV("OS")},
      0ms);

  check_exception("Expected '%' or '}' in the chrono format-string", SV("{:A"), 0ms);
  check_exception("The chrono-specs contains a '{'", SV("{:%%{"), 0ms);
  check_exception("End of input while parsing the modifier chrono conversion-spec", SV("{:%"), 0ms);
  check_exception("End of input while parsing the modifier E", SV("{:%E"), 0ms);
  check_exception("End of input while parsing the modifier O", SV("{:%O"), 0ms);

  // Precision not allowed
  check_exception("Expected '%' or '}' in the chrono format-string", SV("{:.3}"), 0ms);
}

int main(int, char**) {
  test<char>();

#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  test<wchar_t>();
#endif

  return 0;
}
