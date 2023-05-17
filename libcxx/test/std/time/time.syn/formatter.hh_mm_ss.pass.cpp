//===----------------------------------------------------------------------===//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: no-localization
// UNSUPPORTED: libcpp-has-no-incomplete-format

// TODO FMT Evaluate gcc-12 status
// UNSUPPORTED: gcc-12

// TODO FMT Investigate Windows issues.
// UNSUPPORTED: msvc, target={{.+}}-windows-gnu

// XFAIL: LIBCXX-FREEBSD-FIXME

// XFAIL: availability-fp_to_chars-missing

// REQUIRES: locale.fr_FR.UTF-8
// REQUIRES: locale.ja_JP.UTF-8

// <chrono>

// template<class Rep, class Period, class charT>
//   struct formatter<chrono::hh_mm_ss<duration<Rep, Period>>, charT>;

#include <chrono>
#include <format>

#include <cassert>
#include <concepts>
#include <locale>
#include <iostream>
#include <ratio>
#include <type_traits>

#include "formatter_tests.h"
#include "make_string.h"
#include "platform_support.h" // locale name macros
#include "string_literal.h"
#include "test_macros.h"

template <class CharT>
static void test_no_chrono_specs() {
  using namespace std::literals::chrono_literals;

  std::locale::global(std::locale(LOCALE_fr_FR_UTF_8));

  // Non localized output
  check(SV("00:00:00.000"), SV("{}"), std::chrono::hh_mm_ss{0ms});
  check(SV("*00:00:00.000*"), SV("{:*^14}"), std::chrono::hh_mm_ss{0ms});
  check(SV("*00:00:00.000"), SV("{:*>13}"), std::chrono::hh_mm_ss{0ms});

  std::locale::global(std::locale::classic());
}

template <class CharT>
static void test_valid_values() {
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
        std::chrono::hh_mm_ss(0s));

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
        std::chrono::hh_mm_ss(23h + 31min + 30s + 123ms));

  check(SV("-%H='03'\t"
           "%OH='03'\t"
           "%I='03'\t"
           "%OI='03'\t"
           "%M='02'\t"
           "%OM='02'\t"
           "%S='01.123456789012'\t"
           "%OS='01.123456789012'\t"
           "%p='AM'\t"
           "%R='03:02'\t"
           "%T='03:02:01.123456789012'\t"
           "%r='03:02:01 AM'\t"
           "%X='03:02:01'\t"
           "%EX='03:02:01'\t"
           "\n"),
        fmt,
        std::chrono::hh_mm_ss(-(3h + 2min + 1s + std::chrono::duration<std::int64_t, std::pico>(123456789012))));

  // The number of fractional seconds is 0 according to the Standard
  // TODO FMT Determine what to do.
  check(SV("%H='01'\t"
           "%OH='01'\t"
           "%I='01'\t"
           "%OI='01'\t"
           "%M='01'\t"
           "%OM='01'\t"
           "%S='01'\t"
           "%OS='01'\t"
           "%p='AM'\t"
           "%R='01:01'\t"
           "%T='01:01:01'\t"
           "%r='01:01:01 AM'\t"
           "%X='01:01:01'\t"
           "%EX='01:01:01'\t"
           "\n"),
        fmt,
        std::chrono::hh_mm_ss(std::chrono::duration<double>(3661.123456)));

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
#elif defined(__APPLE__)
           "%r=''\t"
#else
           "%r='12:00:00 '\t"
#endif
           "%X='00:00:00'\t"
           "%EX='00:00:00'\t"
           "\n"),
        lfmt,
        std::chrono::hh_mm_ss(0s));

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
#elif defined(__APPLE__)
           "%r=''\t"
#else
           "%r='11:31:30 '\t"
#endif
           "%X='23:31:30'\t"
           "%EX='23:31:30'\t"
           "\n"),
        lfmt,
        std::chrono::hh_mm_ss(23h + 31min + 30s + 123ms));

  check(SV("-%H='03'\t"
           "%OH='03'\t"
           "%I='03'\t"
           "%OI='03'\t"
           "%M='02'\t"
           "%OM='02'\t"
           "%S='01,123456789012'\t"
           "%OS='01,123456789012'\t"
#if defined(_AIX)
           "%p='AM'\t"
#else
           "%p=''\t"
#endif
           "%R='03:02'\t"
           "%T='03:02:01,123456789012'\t"
#ifdef _WIN32
           "%r='03:02:01'\t"
#elif defined(_AIX)
           "%r='03:02:01 AM'\t"
#elif defined(__APPLE__)
           "%r=''\t"
#else
           "%r='03:02:01 '\t"
#endif
           "%X='03:02:01'\t"
           "%EX='03:02:01'\t"
           "\n"),
        lfmt,
        std::chrono::hh_mm_ss(-(3h + 2min + 1s + std::chrono::duration<std::int64_t, std::pico>(123456789012))));

  check(SV("%H='01'\t"
           "%OH='01'\t"
           "%I='01'\t"
           "%OI='01'\t"
           "%M='01'\t"
           "%OM='01'\t"
           "%S='01'\t"
           "%OS='01'\t"
#if defined(_AIX)
           "%p='AM'\t"
#else
           "%p=''\t"
#endif
           "%R='01:01'\t"
           "%T='01:01:01'\t"
#ifdef _WIN32
           "%r='01:01:01'\t"
#elif defined(_AIX)
           "%r='01:01:01 AM'\t"
#elif defined(__APPLE__)
           "%r=''\t"
#else
           "%r='01:01:01 '\t"
#endif
           "%X='01:01:01'\t"
           "%EX='01:01:01'\t"
           "\n"),
        lfmt,
        std::chrono::hh_mm_ss(std::chrono::duration<double>(3661.123456)));

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
#  if defined(__APPLE__)
           "%r='11:31:30 PM'\t"
           "%X='23時31分30秒'\t"
           "%EX='23時31分30秒'\t"
#  else
           "%r='午後11:31:30'\t"
           "%X='23:31:30'\t"
           "%EX='23:31:30'\t"
#  endif
           "\n"),
        lfmt,
        std::chrono::hh_mm_ss(23h + 31min + 30s + 123ms));

  check(loc,
        SV("-%H='03'\t"
           "%OH='03'\t"
           "%I='03'\t"
           "%OI='03'\t"
           "%M='02'\t"
           "%OM='02'\t"
           "%S='01.123456789012'\t"
           "%OS='01.123456789012'\t"
#  if defined(__APPLE__)
           "%p='AM'\t"
#  else
           "%p='午前'\t"
#  endif
           "%R='03:02'\t"
           "%T='03:02:01.123456789012'\t"
#  if defined(__APPLE__)
           "%r='03:02:01 AM'\t"
           "%X='03時02分01秒'\t"
           "%EX='03時02分01秒'\t"
#  else
           "%r='午前03:02:01'\t"
           "%X='03:02:01'\t"
           "%EX='03:02:01'\t"
#  endif
           "\n"),
        lfmt,
        std::chrono::hh_mm_ss(-(3h + 2min + 1s + std::chrono::duration<std::int64_t, std::pico>(123456789012))));

  check(loc,
        SV("%H='01'\t"
           "%OH='01'\t"
           "%I='01'\t"
           "%OI='01'\t"
           "%M='01'\t"
           "%OM='01'\t"
           "%S='01'\t"
           "%OS='01'\t"
#  if defined(__APPLE__)
           "%p='AM'\t"
#  else
           "%p='午前'\t"
#  endif
           "%R='01:01'\t"
           "%T='01:01:01'\t"
#  if defined(__APPLE__)
           "%r='01:01:01 AM'\t"
           "%X='01時01分01秒'\t"
           "%EX='01時01分01秒'\t"
#  else
           "%r='午前01:01:01'\t"
           "%X='01:01:01'\t"
           "%EX='01:01:01'\t"
#  endif
           "\n"),
        lfmt,
        std::chrono::hh_mm_ss(std::chrono::duration<double>(3661.123456)));
#else  // defined(__APPLE__) || defined(_AIX)
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
        std::chrono::hh_mm_ss(0s));

  // TODO FMT What should fractions be in alternate display mode?
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
        std::chrono::hh_mm_ss(23h + 31min + 30s + 123ms));

  check(loc,
        SV("-%H='03'\t"
           "%OH='三'\t"
           "%I='03'\t"
           "%OI='三'\t"
           "%M='02'\t"
           "%OM='二'\t"
           "%S='01.123456789012'\t"
           "%OS='一.123456789012'\t"
           "%p='午前'\t"
           "%R='03:02'\t"
           "%T='03:02:01.123456789012'\t"
           "%r='午前03時02分01秒'\t"
           "%X='03時02分01秒'\t"
           "%EX='03時02分01秒'\t"
           "\n"),
        lfmt,
        std::chrono::hh_mm_ss(-(3h + 2min + 1s + std::chrono::duration<std::int64_t, std::pico>(123456789012))));

  check(loc,
        SV("%H='01'\t"
           "%OH='一'\t"
           "%I='01'\t"
           "%OI='一'\t"
           "%M='01'\t"
           "%OM='一'\t"
           "%S='01'\t"
           "%OS='一'\t"
           "%p='午前'\t"
           "%R='01:01'\t"
           "%T='01:01:01'\t"
           "%r='午前01時01分01秒'\t"
           "%X='01時01分01秒'\t"
           "%EX='01時01分01秒'\t"
           "\n"),
        lfmt,
        std::chrono::hh_mm_ss(std::chrono::duration<double>(3661.123456)));
#endif // defined(__APPLE__) || defined(_AIX)

  std::locale::global(std::locale::classic());
}

template <class CharT>
static void test_invalid_values() {
  using namespace std::literals::chrono_literals;

  // This looks odd, however the 24 hours is not valid for a 24 hour clock.
  // TODO FMT discuss what the "proper" behaviour is.
  check_exception("formatting a hour needs a valid value", SV("{:%H"), std::chrono::hh_mm_ss{24h});
  check_exception("formatting a hour needs a valid value", SV("{:%OH"), std::chrono::hh_mm_ss{24h});
  check_exception("formatting a hour needs a valid value", SV("{:%I"), std::chrono::hh_mm_ss{24h});
  check_exception("formatting a hour needs a valid value", SV("{:%OI"), std::chrono::hh_mm_ss{24h});
  check(SV("00"), SV("{:%M}"), std::chrono::hh_mm_ss{24h});
  check(SV("00"), SV("{:%OM}"), std::chrono::hh_mm_ss{24h});
  check(SV("00"), SV("{:%S}"), std::chrono::hh_mm_ss{24h});
  check(SV("00"), SV("{:%OS}"), std::chrono::hh_mm_ss{24h});
  check_exception("formatting a hour needs a valid value", SV("{:%p"), std::chrono::hh_mm_ss{24h});
  check_exception("formatting a hour needs a valid value", SV("{:%R"), std::chrono::hh_mm_ss{24h});
  check_exception("formatting a hour needs a valid value", SV("{:%T"), std::chrono::hh_mm_ss{24h});
  check_exception("formatting a hour needs a valid value", SV("{:%r"), std::chrono::hh_mm_ss{24h});
  check_exception("formatting a hour needs a valid value", SV("{:%X"), std::chrono::hh_mm_ss{24h});
  check_exception("formatting a hour needs a valid value", SV("{:%EX"), std::chrono::hh_mm_ss{24h});
}

template <class CharT>
static void test() {
  using namespace std::literals::chrono_literals;

  test_no_chrono_specs<CharT>();
  test_valid_values<CharT>();
  test_invalid_values<CharT>();
  check_invalid_types<CharT>(
      {SV("H"),
       SV("I"),
       SV("M"),
       SV("S"),
       SV("p"),
       SV("r"),
       SV("R"),
       SV("T"),
       SV("X"),
       SV("OH"),
       SV("OI"),
       SV("OM"),
       SV("OS"),
       SV("EX")},
      std::chrono::hh_mm_ss{0ms});

  check_exception("Expected '%' or '}' in the chrono format-string", SV("{:A"), std::chrono::hh_mm_ss{0ms});
  check_exception("The chrono-specs contains a '{'", SV("{:%%{"), std::chrono::hh_mm_ss{0ms});
  check_exception(
      "End of input while parsing the modifier chrono conversion-spec", SV("{:%"), std::chrono::hh_mm_ss{0ms});
  check_exception("End of input while parsing the modifier E", SV("{:%E"), std::chrono::hh_mm_ss{0ms});
  check_exception("End of input while parsing the modifier O", SV("{:%O"), std::chrono::hh_mm_ss{0ms});

  check_exception("Expected '%' or '}' in the chrono format-string", SV("{:.3}"), std::chrono::hh_mm_ss{0ms});
}

int main(int, char**) {
  test<char>();

#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  test<wchar_t>();
#endif

  return 0;
}
