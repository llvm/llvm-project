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

// template<class charT, class traits, class Duration, class TimeZonePtr>
// basic_ostream<charT, traits>&
// operator<<(basic_ostream<charT, traits>& os,
//            const zoned_time<Duration, TimeZonePtr>& t);

#include <chrono>
#include <cassert>
#include <sstream>

#include "assert_macros.h"
#include "concat_macros.h"
#include "make_string.h"
#include "platform_support.h" // locale name macros
#include "test_macros.h"
#include "../test_offset_time_zone.h"

#define SV(S) MAKE_STRING_VIEW(CharT, S)

#define TEST_EQUAL(OUT, EXPECTED)                                                                                      \
  TEST_REQUIRE(OUT == EXPECTED,                                                                                        \
               TEST_WRITE_CONCATENATED(                                                                                \
                   "\nExpression      ", #OUT, "\nExpected output ", EXPECTED, "\nActual output   ", OUT, '\n'));

template <class CharT, class Duration, class TimeZonePtr>
static std::basic_string<CharT> stream_c_locale(std::chrono::zoned_time<Duration, TimeZonePtr> time_point) {
  std::basic_stringstream<CharT> sstr;
  sstr << time_point;
  return sstr.str();
}

template <class CharT, class Duration, class TimeZonePtr>
static std::basic_string<CharT> stream_fr_FR_locale(std::chrono::zoned_time<Duration, TimeZonePtr> time_point) {
  std::basic_stringstream<CharT> sstr;
  const std::locale locale(LOCALE_fr_FR_UTF_8);
  sstr.imbue(locale);
  sstr << time_point;
  return sstr.str();
}

template <class CharT, class Duration, class TimeZonePtr>
static std::basic_string<CharT> stream_ja_JP_locale(std::chrono::zoned_time<Duration, TimeZonePtr> time_point) {
  std::basic_stringstream<CharT> sstr;
  const std::locale locale(LOCALE_ja_JP_UTF_8);
  sstr.imbue(locale);
  sstr << time_point;
  return sstr.str();
}

template <class CharT>
static void test_c() {
  using namespace std::literals::chrono_literals;

  { //  Different durations
    TEST_EQUAL(stream_c_locale<CharT>(
                   std::chrono::zoned_time("Etc/GMT-1", std::chrono::sys_time<std::chrono::nanoseconds>{42ns})),
               SV("1970-01-01 01:00:00.000000042 +01"));

    TEST_EQUAL(stream_c_locale<CharT>(
                   std::chrono::zoned_time("Etc/GMT-1", std::chrono::sys_time<std::chrono::microseconds>{42us})),
               SV("1970-01-01 01:00:00.000042 +01"));

    TEST_EQUAL(stream_c_locale<CharT>(
                   std::chrono::zoned_time("Etc/GMT-1", std::chrono::sys_time<std::chrono::milliseconds>{42ms})),
               SV("1970-01-01 01:00:00.042 +01"));

    TEST_EQUAL(
        stream_c_locale<CharT>(std::chrono::zoned_time("Etc/GMT-1", std::chrono::sys_time<std::chrono::seconds>{42s})),
        SV("1970-01-01 01:00:42 +01"));

    TEST_EQUAL(stream_c_locale<CharT>(std::chrono::zoned_time(
                   "Etc/GMT-1", std::chrono::sys_time<std::chrono::days>{std::chrono::days{42}})),
               SV("1970-02-12 01:00:00 +01"));

    TEST_EQUAL(stream_c_locale<CharT>(std::chrono::zoned_time(
                   "Etc/GMT-1", std::chrono::sys_time<std::chrono::weeks>{std::chrono::weeks{42}})),
               SV("1970-10-22 01:00:00 +01"));
  }

  { // Daylight saving time switches
    // Pick an historic date where it's well known what the time zone rules were.
    // This makes it unlikely updates to the database change these rules.

    // Z Europe/Berlin 0:53:28 - LMT 1893 Ap
    // ...
    // 1 DE CE%sT 1980
    // 1 E CE%sT
    //
    // ...
    // R E 1979 1995 - S lastSu 1u 0 -
    // R E 1981 ma - Mar lastSu 1u 1 S

    // Pick an historic date where it's well known what the time zone rules were.
    // This makes it unlikely updates to the database change these rules.

    // Start of daylight saving time
    TEST_EQUAL(stream_c_locale<CharT>(std::chrono::zoned_time(
                   "Europe/Berlin", std::chrono::sys_days{std::chrono::March / 30 / 1986} + 0h + 59min + 59s)),
               SV("1986-03-30 01:59:59 CET"));

    TEST_EQUAL(stream_c_locale<CharT>(std::chrono::zoned_time(
                   "Europe/Berlin", std::chrono::sys_days{std::chrono::March / 30 / 1986} + 1h)),
               SV("1986-03-30 03:00:00 CEST"));

    // End of daylight saving time
    TEST_EQUAL(stream_c_locale<CharT>(std::chrono::zoned_time(
                   "Europe/Berlin", std::chrono::sys_days{std::chrono::September / 28 / 1986} + 0h + 59min + 59s)),
               SV("1986-09-28 02:59:59 CEST"));

    TEST_EQUAL(stream_c_locale<CharT>(std::chrono::zoned_time(
                   "Europe/Berlin", std::chrono::sys_days{std::chrono::September / 28 / 1986} + 1h)),
               SV("1986-09-28 02:00:00 CET"));

    TEST_EQUAL(stream_c_locale<CharT>(std::chrono::zoned_time(
                   "Europe/Berlin", std::chrono::sys_days{std::chrono::September / 28 / 1986} + 1h + 59min + 59s)),
               SV("1986-09-28 02:59:59 CET"));

    TEST_EQUAL(stream_c_locale<CharT>(std::chrono::zoned_time(
                   "Europe/Berlin", std::chrono::sys_days{std::chrono::September / 28 / 1986} + 2h)),
               SV("1986-09-28 03:00:00 CET"));
  }

  { // offset pointer
    TEST_EQUAL(stream_c_locale<CharT>(std::chrono::zoned_time(
                   offset_time_zone<offset_time_zone_flags::none>{}, std::chrono::sys_seconds{})),
               SV("1970-01-01 00:00:00 +00s"));

    TEST_EQUAL(stream_c_locale<CharT>(std::chrono::zoned_time(
                   offset_time_zone<offset_time_zone_flags::none>{"42"}, std::chrono::sys_seconds{})),
               SV("1969-12-31 23:59:18 +42s"));

    TEST_EQUAL(stream_c_locale<CharT>(std::chrono::zoned_time(
                   offset_time_zone<offset_time_zone_flags::none>{"-42"}, std::chrono::sys_seconds{})),
               SV("1970-01-01 00:00:42 -42s"));
  }
}

template <class CharT>
static void test_fr_FR() {
  using namespace std::literals::chrono_literals;

  { //  Different durations

    TEST_EQUAL(stream_fr_FR_locale<CharT>(
                   std::chrono::zoned_time("Etc/GMT-1", std::chrono::sys_time<std::chrono::nanoseconds>{42ns})),
               SV("1970-01-01 01:00:00,000000042 +01"));

    TEST_EQUAL(stream_fr_FR_locale<CharT>(
                   std::chrono::zoned_time("Etc/GMT-1", std::chrono::sys_time<std::chrono::microseconds>{42us})),
               SV("1970-01-01 01:00:00,000042 +01"));

    TEST_EQUAL(stream_fr_FR_locale<CharT>(
                   std::chrono::zoned_time("Etc/GMT-1", std::chrono::sys_time<std::chrono::milliseconds>{42ms})),
               SV("1970-01-01 01:00:00,042 +01"));

    TEST_EQUAL(stream_fr_FR_locale<CharT>(
                   std::chrono::zoned_time("Etc/GMT-1", std::chrono::sys_time<std::chrono::seconds>{42s})),
               SV("1970-01-01 01:00:42 +01"));

    TEST_EQUAL(stream_fr_FR_locale<CharT>(std::chrono::zoned_time(
                   "Etc/GMT-1", std::chrono::sys_time<std::chrono::days>{std::chrono::days{42}})),
               SV("1970-02-12 01:00:00 +01"));

    TEST_EQUAL(stream_fr_FR_locale<CharT>(std::chrono::zoned_time(
                   "Etc/GMT-1", std::chrono::sys_time<std::chrono::weeks>{std::chrono::weeks{42}})),
               SV("1970-10-22 01:00:00 +01"));
  }

  { // Daylight saving time switches
    // Pick an historic date where it's well known what the time zone rules were.
    // This makes it unlikely updates to the database change these rules.

    // Z Europe/Berlin 0:53:28 - LMT 1893 Ap
    // ...
    // 1 DE CE%sT 1980
    // 1 E CE%sT
    //
    // ...
    // R E 1979 1995 - S lastSu 1u 0 -
    // R E 1981 ma - Mar lastSu 1u 1 S

    // Pick an historic date where it's well known what the time zone rules were.
    // This makes it unlikely updates to the database change these rules.

    // Start of daylight saving time
    TEST_EQUAL(stream_fr_FR_locale<CharT>(std::chrono::zoned_time(
                   "Europe/Berlin", std::chrono::sys_days{std::chrono::March / 30 / 1986} + 0h + 59min + 59s)),
               SV("1986-03-30 01:59:59 CET"));

    TEST_EQUAL(stream_fr_FR_locale<CharT>(std::chrono::zoned_time(
                   "Europe/Berlin", std::chrono::sys_days{std::chrono::March / 30 / 1986} + 1h)),
               SV("1986-03-30 03:00:00 CEST"));

    // End of daylight saving time
    TEST_EQUAL(stream_fr_FR_locale<CharT>(std::chrono::zoned_time(
                   "Europe/Berlin", std::chrono::sys_days{std::chrono::September / 28 / 1986} + 0h + 59min + 59s)),
               SV("1986-09-28 02:59:59 CEST"));

    TEST_EQUAL(stream_fr_FR_locale<CharT>(std::chrono::zoned_time(
                   "Europe/Berlin", std::chrono::sys_days{std::chrono::September / 28 / 1986} + 1h)),
               SV("1986-09-28 02:00:00 CET"));

    TEST_EQUAL(stream_fr_FR_locale<CharT>(std::chrono::zoned_time(
                   "Europe/Berlin", std::chrono::sys_days{std::chrono::September / 28 / 1986} + 1h + 59min + 59s)),
               SV("1986-09-28 02:59:59 CET"));

    TEST_EQUAL(stream_fr_FR_locale<CharT>(std::chrono::zoned_time(
                   "Europe/Berlin", std::chrono::sys_days{std::chrono::September / 28 / 1986} + 2h)),
               SV("1986-09-28 03:00:00 CET"));
  }

  { // offset pointer
    TEST_EQUAL(stream_fr_FR_locale<CharT>(std::chrono::zoned_time(
                   offset_time_zone<offset_time_zone_flags::none>{}, std::chrono::sys_seconds{})),
               SV("1970-01-01 00:00:00 +00s"));

    TEST_EQUAL(stream_fr_FR_locale<CharT>(std::chrono::zoned_time(
                   offset_time_zone<offset_time_zone_flags::none>{"42"}, std::chrono::sys_seconds{})),
               SV("1969-12-31 23:59:18 +42s"));

    TEST_EQUAL(stream_fr_FR_locale<CharT>(std::chrono::zoned_time(
                   offset_time_zone<offset_time_zone_flags::none>{"-42"}, std::chrono::sys_seconds{})),
               SV("1970-01-01 00:00:42 -42s"));
  }
}

template <class CharT>
static void test_ja_JP() {
  using namespace std::literals::chrono_literals;

  { //  Different durations

    TEST_EQUAL(stream_ja_JP_locale<CharT>(
                   std::chrono::zoned_time("Etc/GMT-1", std::chrono::sys_time<std::chrono::nanoseconds>{42ns})),
               SV("1970-01-01 01:00:00.000000042 +01"));

    TEST_EQUAL(stream_ja_JP_locale<CharT>(
                   std::chrono::zoned_time("Etc/GMT-1", std::chrono::sys_time<std::chrono::microseconds>{42us})),
               SV("1970-01-01 01:00:00.000042 +01"));

    TEST_EQUAL(stream_ja_JP_locale<CharT>(
                   std::chrono::zoned_time("Etc/GMT-1", std::chrono::sys_time<std::chrono::milliseconds>{42ms})),
               SV("1970-01-01 01:00:00.042 +01"));

    TEST_EQUAL(stream_ja_JP_locale<CharT>(
                   std::chrono::zoned_time("Etc/GMT-1", std::chrono::sys_time<std::chrono::seconds>{42s})),
               SV("1970-01-01 01:00:42 +01"));

    TEST_EQUAL(stream_ja_JP_locale<CharT>(std::chrono::zoned_time(
                   "Etc/GMT-1", std::chrono::sys_time<std::chrono::days>{std::chrono::days{42}})),
               SV("1970-02-12 01:00:00 +01"));

    TEST_EQUAL(stream_ja_JP_locale<CharT>(std::chrono::zoned_time(
                   "Etc/GMT-1", std::chrono::sys_time<std::chrono::weeks>{std::chrono::weeks{42}})),
               SV("1970-10-22 01:00:00 +01"));
  }

  { // Daylight saving time switches
    // Pick an historic date where it's well known what the time zone rules were.
    // This makes it unlikely updates to the database change these rules.

    // Z Europe/Berlin 0:53:28 - LMT 1893 Ap
    // ...
    // 1 DE CE%sT 1980
    // 1 E CE%sT
    //
    // ...
    // R E 1979 1995 - S lastSu 1u 0 -
    // R E 1981 ma - Mar lastSu 1u 1 S

    // Pick an historic date where it's well known what the time zone rules were.
    // This makes it unlikely updates to the database change these rules.

    // Start of daylight saving time
    TEST_EQUAL(stream_ja_JP_locale<CharT>(std::chrono::zoned_time(
                   "Europe/Berlin", std::chrono::sys_days{std::chrono::March / 30 / 1986} + 0h + 59min + 59s)),
               SV("1986-03-30 01:59:59 CET"));

    TEST_EQUAL(stream_ja_JP_locale<CharT>(std::chrono::zoned_time(
                   "Europe/Berlin", std::chrono::sys_days{std::chrono::March / 30 / 1986} + 1h)),
               SV("1986-03-30 03:00:00 CEST"));

    // End of daylight saving time
    TEST_EQUAL(stream_ja_JP_locale<CharT>(std::chrono::zoned_time(
                   "Europe/Berlin", std::chrono::sys_days{std::chrono::September / 28 / 1986} + 0h + 59min + 59s)),
               SV("1986-09-28 02:59:59 CEST"));

    TEST_EQUAL(stream_ja_JP_locale<CharT>(std::chrono::zoned_time(
                   "Europe/Berlin", std::chrono::sys_days{std::chrono::September / 28 / 1986} + 1h)),
               SV("1986-09-28 02:00:00 CET"));

    TEST_EQUAL(stream_ja_JP_locale<CharT>(std::chrono::zoned_time(
                   "Europe/Berlin", std::chrono::sys_days{std::chrono::September / 28 / 1986} + 1h + 59min + 59s)),
               SV("1986-09-28 02:59:59 CET"));

    TEST_EQUAL(stream_ja_JP_locale<CharT>(std::chrono::zoned_time(
                   "Europe/Berlin", std::chrono::sys_days{std::chrono::September / 28 / 1986} + 2h)),
               SV("1986-09-28 03:00:00 CET"));
  }

  { // offset pointer
    TEST_EQUAL(stream_ja_JP_locale<CharT>(std::chrono::zoned_time(
                   offset_time_zone<offset_time_zone_flags::none>{}, std::chrono::sys_seconds{})),
               SV("1970-01-01 00:00:00 +00s"));

    TEST_EQUAL(stream_ja_JP_locale<CharT>(std::chrono::zoned_time(
                   offset_time_zone<offset_time_zone_flags::none>{"42"}, std::chrono::sys_seconds{})),
               SV("1969-12-31 23:59:18 +42s"));

    TEST_EQUAL(stream_ja_JP_locale<CharT>(std::chrono::zoned_time(
                   offset_time_zone<offset_time_zone_flags::none>{"-42"}, std::chrono::sys_seconds{})),
               SV("1970-01-01 00:00:42 -42s"));
  }
}

template <class CharT>
static void test() {
  test_c<CharT>();
  test_fr_FR<CharT>();
  test_ja_JP<CharT>();
}

int main(int, char**) {
  test<char>();

#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  test<wchar_t>();
#endif

  return 0;
}
