//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// TODO TZDB review the test based on review comments in
// https://github.com/llvm/llvm-project/pull/85619

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20, c++23, c++26
// UNSUPPORTED: no-filesystem, no-localization, no-tzdb

// XFAIL: libcpp-has-no-experimental-tzdb
// XFAIL: availability-tzdb-missing

// <chrono>

// class time_zone;

// template <class _Duration>
//   sys_info get_info(const sys_time<_Duration>& time) const;

// This test uses the system provided database. This makes the test portable,
// but may cause failures when the database information changes. Historic data
// may change if new facts are uncovered, future data may change when regions
// change their time zone or daylight saving time. Most tests will not look in
// the future to attempt to avoid issues. All tests list the data on which they
// are based, this makes debugging easier upon failure; including to see whether
// the provided data has not been changed
//
//
// The data in the tests can be validated by using the zdump tool. For
// example
//   zdump -v Asia/Hong_Kong
// show all transistions in the Hong Kong time zone. Or
//   zdump -c1970,1980 -v Asia/Hong_Kong
// shows all transitions in Hong Kong between 1970 and 1980.

#include <algorithm>
#include <cassert>
#include <chrono>
#include <format>

#include "test_macros.h"
#include "assert_macros.h"
#include "concat_macros.h"

/***** ***** HELPERS ***** *****/

[[nodiscard]] static std::chrono::sys_seconds to_sys_seconds(
    std::chrono::year year,
    std::chrono::month month,
    std::chrono::day day,
    std::chrono::hours h   = std::chrono::hours(0),
    std::chrono::minutes m = std::chrono::minutes{0},
    std::chrono::seconds s = std::chrono::seconds{0}) {
  std::chrono::year_month_day result{year, month, day};

  return std::chrono::time_point_cast<std::chrono::seconds>(static_cast<std::chrono::sys_days>(result)) + h + m + s;
}

static void assert_equal(const std::chrono::sys_info& lhs, const std::chrono::sys_info& rhs) {
  TEST_REQUIRE(lhs.begin == rhs.begin,
               TEST_WRITE_CONCATENATED("\nBegin:\nExpected output ", lhs.begin, "\nActual output   ", rhs.begin, '\n'));
  TEST_REQUIRE(lhs.end == rhs.end,
               TEST_WRITE_CONCATENATED("\nEnd:\nExpected output ", lhs.end, "\nActual output   ", rhs.end, '\n'));
  TEST_REQUIRE(
      lhs.offset == rhs.offset,
      TEST_WRITE_CONCATENATED("\nOffset:\nExpected output ", lhs.offset, "\nActual output   ", rhs.offset, '\n'));
  TEST_REQUIRE(lhs.save == rhs.save,
               TEST_WRITE_CONCATENATED("\nSave:\nExpected output ", lhs.save, "\nActual output   ", rhs.save, '\n'));
  TEST_REQUIRE(
      lhs.abbrev == rhs.abbrev,
      TEST_WRITE_CONCATENATED("\nAbbrev:\nExpected output ", lhs.abbrev, "\nActual output   ", rhs.abbrev, '\n'));
}

static void assert_equal(std::string_view expected, const std::chrono::sys_info& value) {
  // Note the output of operator<< is implementation defined, use this
  // format to keep the test portable.
  std::string result = std::format(
      "[{}, {}) {:%T} {:%Q%q} {}",
      value.begin,
      value.end,
      std::chrono::hh_mm_ss{value.offset},
      value.save,
      value.abbrev);

  TEST_REQUIRE(expected == result,
               TEST_WRITE_CONCATENATED("\nExpected output ", expected, "\nActual output   ", result, '\n'));
}

static void
assert_range(std::string_view expected, const std::chrono::sys_info& begin, const std::chrono::sys_info& end) {
  assert_equal(expected, begin);
  assert_equal(expected, end);
}

static void assert_cycle(
    std::string_view expected_1,
    const std::chrono::sys_info& begin_1,
    const std::chrono::sys_info& end_1,
    std::string_view expected_2,
    const std::chrono::sys_info& begin_2,
    const std::chrono::sys_info& end_2

) {
  assert_range(expected_1, begin_1, end_1);
  assert_range(expected_2, begin_2, end_2);
}

/***** ***** TESTS ***** *****/

static void test_gmt() {
  // Simple zone always valid, no rule entries, lookup using a link.
  // L Etc/GMT GMT
  // Z Etc/GMT 0 - GMT

  const std::chrono::time_zone* tz = std::chrono::locate_zone("GMT");

  assert_equal(
      std::chrono::sys_info(
          std::chrono::sys_seconds::min(),
          std::chrono::sys_seconds::max(),
          std::chrono::seconds(0),
          std::chrono::minutes(0),
          "GMT"),
      tz->get_info(std::chrono::sys_seconds::min()));
  assert_equal(
      std::chrono::sys_info(
          std::chrono::sys_seconds::min(),
          std::chrono::sys_seconds::max(),
          std::chrono::seconds(0),
          std::chrono::minutes(0),
          "GMT"),
      tz->get_info(std::chrono::sys_seconds(std::chrono::seconds{0})));

  assert_equal(
      std::chrono::sys_info(
          std::chrono::sys_seconds::min(),
          std::chrono::sys_seconds::max(),
          std::chrono::seconds(0),
          std::chrono::minutes(0),
          "GMT"),
      tz->get_info(std::chrono::sys_seconds::max() - std::chrono::seconds{1})); // max is not valid
}

static void test_durations() {
  // Doesn't test a location, instead tests whether different duration
  // specializations work.
  const std::chrono::time_zone* tz = std::chrono::locate_zone("GMT");

  // Using the GMT zone means every call gives the same result.
  std::chrono::sys_info expected(
      std::chrono::sys_seconds::min(),
      std::chrono::sys_seconds::max(),
      std::chrono::seconds(0),
      std::chrono::minutes(0),
      "GMT");

  assert_equal(expected, tz->get_info(std::chrono::sys_time<std::chrono::nanoseconds>{}));
  assert_equal(expected, tz->get_info(std::chrono::sys_time<std::chrono::microseconds>{}));
  assert_equal(expected, tz->get_info(std::chrono::sys_time<std::chrono::milliseconds>{}));
  assert_equal(expected, tz->get_info(std::chrono::sys_time<std::chrono::seconds>{}));
  assert_equal(expected, tz->get_info(std::chrono::sys_time<std::chrono::minutes>{}));
  assert_equal(expected, tz->get_info(std::chrono::sys_time<std::chrono::minutes>{}));
  assert_equal(expected, tz->get_info(std::chrono::sys_time<std::chrono::hours>{}));
  assert_equal(expected, tz->get_info(std::chrono::sys_time<std::chrono::days>{}));
  assert_equal(expected, tz->get_info(std::chrono::sys_time<std::chrono::weeks>{}));
  assert_equal(expected, tz->get_info(std::chrono::sys_time<std::chrono::months>{}));
  assert_equal(expected, tz->get_info(std::chrono::sys_time<std::chrono::years>{}));
}

static void test_antarctica_syowa() {
  // One change, no rules, no dst changes
  // This change uses an ON field with a day number
  //
  // There don't seem to be rule-less zones that use last day or a
  // contrained day

  // Z Antarctica/Syowa 0 - -00 1957 Ja 29
  // 3 - +03

  const std::chrono::time_zone* tz = std::chrono::locate_zone("Antarctica/Syowa");

  std::chrono::sys_seconds transition =
      to_sys_seconds(std::chrono::year(1957), std::chrono::January, std::chrono::day(29));

  assert_equal(
      std::chrono::sys_info(
          std::chrono::sys_seconds::min(), //
          transition,                      //
          std::chrono::seconds(0),         //
          std::chrono::minutes(0),         //
          "-00"),                          //
      tz->get_info(std::chrono::sys_seconds::min()));

  assert_equal(
      std::chrono::sys_info(
          std::chrono::sys_seconds::min(), //
          transition,                      //
          std::chrono::seconds(0),         //
          std::chrono::minutes(0),         //
          "-00"),                          //
      tz->get_info(transition - std::chrono::seconds(1)));

  assert_equal(
      std::chrono::sys_info(
          transition,                      //
          std::chrono::sys_seconds::max(), //
          std::chrono::hours(3),           //
          std::chrono::minutes(0),         //
          "+03"),                          //
      tz->get_info(transition));
}

static void test_asia_hong_kong() {
  // A more typical entry, first some hard-coded entires and then at the
  // end a rules based entry. This rule is valid for its entire period
  //
  // Z Asia/Hong_Kong 7:36:42 - LMT 1904 O 30 0:36:42
  // 8 - HKT 1941 Jun 15 3
  // 8 1 HKST 1941 O 1 4
  // 8 0:30 HKWT 1941 D 25
  // 9 - JST 1945 N 18 2
  // 8 HK HK%sT
  //
  // R HK 1946 o - Ap 21 0 1 S
  // R HK 1946 o - D 1 3:30s 0 -
  // R HK 1947 o - Ap 13 3:30s 1 S
  // R HK 1947 o - N 30 3:30s 0 -
  // R HK 1948 o - May 2 3:30s 1 S
  // R HK 1948 1952 - O Su>=28 3:30s 0 -
  // R HK 1949 1953 - Ap Su>=1 3:30 1 S
  // R HK 1953 1964 - O Su>=31 3:30 0 -
  // R HK 1954 1964 - Mar Su>=18 3:30 1 S
  // R HK 1965 1976 - Ap Su>=16 3:30 1 S
  // R HK 1965 1976 - O Su>=16 3:30 0 -
  // R HK 1973 o - D 30 3:30 1 S
  // R HK 1979 o - May 13 3:30 1 S
  // R HK 1979 o - O 21 3:30 0 -

  using namespace std::literals::chrono_literals;
  const std::chrono::time_zone* tz = std::chrono::locate_zone("Asia/Hong_Kong");

  assert_equal(
      std::chrono::sys_info(
          std::chrono::sys_seconds::min(),
          to_sys_seconds(1904y, std::chrono::October, 29d, 17h), // 7:36:42 - LMT 1904 O 30 0:36:42
          7h + 36min + 42s,
          0min,
          "LMT"),
      tz->get_info(std::chrono::sys_seconds::min()));

  assert_equal(
      std::chrono::sys_info(
          std::chrono::sys_seconds::min(),
          to_sys_seconds(1904y, std::chrono::October, 29d, 17h), // 7:36:42 - LMT 1904 O 30 0:36:42
          7h + 36min + 42s,
          0min,
          "LMT"),
      tz->get_info(to_sys_seconds(1904y, std::chrono::October, 29d, 16h, 59min, 59s)));

  assert_range("[1904-10-29 17:00:00, 1941-06-14 19:00:00) 08:00:00 0min HKT", // 8 - HKT 1941 Jun 15 3
               tz->get_info(to_sys_seconds(1904y, std::chrono::October, 29d, 17h)),
               tz->get_info(to_sys_seconds(1941y, std::chrono::June, 14d, 18h, 59min, 59s)));

  assert_range("[1941-06-14 19:00:00, 1941-09-30 19:00:00) 09:00:00 60min HKST", // 8 1 HKST 1941 O 1 4
               tz->get_info(to_sys_seconds(1941y, std::chrono::June, 14d, 19h)),
               tz->get_info(to_sys_seconds(1941y, std::chrono::September, 30d, 18h, 59min, 59s)));

  assert_range("[1941-09-30 19:00:00, 1941-12-24 15:30:00) 08:30:00 30min HKWT", // 8 0:30 HKWT 1941 D 25
               tz->get_info(to_sys_seconds(1941y, std::chrono::September, 30d, 19h)),
               tz->get_info(to_sys_seconds(1941y, std::chrono::December, 24d, 15h, 29min, 59s)));

  assert_range("[1941-12-24 15:30:00, 1945-11-17 17:00:00) 09:00:00 0min JST", // 9 - JST 1945 N 18 2
               tz->get_info(to_sys_seconds(1941y, std::chrono::December, 24d, 15h, 30min)),
               tz->get_info(to_sys_seconds(1945y, std::chrono::November, 17d, 16h, 59min, 59s)));

  assert_range("[1945-11-17 17:00:00, 1946-04-20 16:00:00) 08:00:00 0min HKT", // 8 HK%sT
               tz->get_info(to_sys_seconds(1945y, std::chrono::November, 17d, 17h)),
               tz->get_info(to_sys_seconds(1946y, std::chrono::April, 20d, 15h, 59min, 59s)));

  assert_cycle( // 8 HK%sT
      "[1946-04-20 16:00:00, 1946-11-30 19:30:00) 09:00:00 60min HKST",
      tz->get_info(to_sys_seconds(1946y, std::chrono::April, 20d, 16h)),                // 1946 o Ap 21 0 1 S
      tz->get_info(to_sys_seconds(1946y, std::chrono::November, 30d, 19h, 29min, 59s)), // 1946 o D 1 3:30s 0 -
      "[1946-11-30 19:30:00, 1947-04-12 19:30:00) 08:00:00 0min HKT",
      tz->get_info(to_sys_seconds(1946y, std::chrono::November, 30d, 19h, 30min)),    // 1946 o D 1 3:30s 0 -
      tz->get_info(to_sys_seconds(1947y, std::chrono::April, 12d, 19h, 29min, 59s))); // 1947 o Ap 13 3:30s 1 S

  assert_cycle( // 8 HK%sT
      "[1947-04-12 19:30:00, 1947-11-29 19:30:00) 09:00:00 60min HKST",
      tz->get_info(to_sys_seconds(1947y, std::chrono::April, 12d, 19h, 30min)),         // 1947 o Ap 13 3:30s 1 S
      tz->get_info(to_sys_seconds(1947y, std::chrono::November, 29d, 19h, 29min, 59s)), // 1947 o N 30 3:30s 0 -
      "[1947-11-29 19:30:00, 1948-05-01 19:30:00) 08:00:00 0min HKT",
      tz->get_info(to_sys_seconds(1947y, std::chrono::November, 29d, 19h, 30min)), // 1947 o N 30 3:30s 0 -
      tz->get_info(to_sys_seconds(1948y, std::chrono::May, 1d, 19h, 29min, 59s))); // 1948 o May 2 3:30s 1 S

  assert_cycle( // 8 HK%sT
      "[1948-05-01 19:30:00, 1948-10-30 19:30:00) 09:00:00 60min HKST",
      tz->get_info(to_sys_seconds(1948y, std::chrono::May, 1d, 19h, 30min)),           // 1948 o May 2 3:30s 1 S
      tz->get_info(to_sys_seconds(1948y, std::chrono::October, 30d, 19h, 29min, 59s)), // 1948 1952 O Su>=28 3:30s 0 -
      "[1948-10-30 19:30:00, 1949-04-02 19:30:00) 08:00:00 0min HKT",
      tz->get_info(to_sys_seconds(1948y, std::chrono::October, 30d, 19h, 30min)),    // 1948 1952 O Su>=28 3:30s 0 -
      tz->get_info(to_sys_seconds(1949y, std::chrono::April, 2d, 19h, 29min, 59s))); // 1949 1953 Ap Su>=1 3:30 1 S

  assert_cycle( // 8 HK%sT
      "[1949-04-02 19:30:00, 1949-10-29 19:30:00) 09:00:00 60min HKST",
      tz->get_info(to_sys_seconds(1949y, std::chrono::April, 2d, 19h, 30min)),         // 1949 1953 Ap Su>=1 3:30 1 S
      tz->get_info(to_sys_seconds(1949y, std::chrono::October, 29d, 19h, 29min, 59s)), // 1948 1952 O Su>=28 3:30s 0
      "[1949-10-29 19:30:00, 1950-04-01 19:30:00) 08:00:00 0min HKT",
      tz->get_info(to_sys_seconds(1949y, std::chrono::October, 29d, 19h, 30min)),    // 1948 1952 O Su>=28 3:30s 0
      tz->get_info(to_sys_seconds(1950y, std::chrono::April, 1d, 19h, 29min, 59s))); // 1949 1953 Ap Su>=1 3:30 1 S

  assert_range(
      "[1953-10-31 18:30:00, 1954-03-20 19:30:00) 08:00:00 0min HKT",
      tz->get_info(to_sys_seconds(1953y, std::chrono::October, 31d, 18h, 30min)),     // 1953 1964 - O Su>=31 3:30 0 -
      tz->get_info(to_sys_seconds(1954y, std::chrono::March, 20d, 19h, 29min, 59s))); // 1954 1964 - Mar Su>=18 3:30 1 S

  assert_cycle( // 8 HK%sT
      "[1953-04-04 19:30:00, 1953-10-31 18:30:00) 09:00:00 60min HKST",
      tz->get_info(to_sys_seconds(1953y, std::chrono::April, 4d, 19h, 30min)),         // 1949 1953 Ap Su>=1 3:30 1 S
      tz->get_info(to_sys_seconds(1953y, std::chrono::October, 31d, 18h, 29min, 59s)), // 1953 1964 - O Su>=31 3:30 0 -
      "[1953-10-31 18:30:00, 1954-03-20 19:30:00) 08:00:00 0min HKT",
      tz->get_info(to_sys_seconds(1953y, std::chrono::October, 31d, 18h, 30min)),     // 1953 1964 - O Su>=31 3:30 0 -
      tz->get_info(to_sys_seconds(1954y, std::chrono::March, 20d, 19h, 29min, 59s))); // 1954 1964 - Mar Su>=18 3:30 1 S

  assert_cycle( // 8 HK%sT
      "[1972-04-15 19:30:00, 1972-10-21 18:30:00) 09:00:00 60min HKST",
      tz->get_info(to_sys_seconds(1972y, std::chrono::April, 19d, 19h, 30min)),        // 1965 1976 - Ap Su>=16 3:30 1 S
      tz->get_info(to_sys_seconds(1972y, std::chrono::October, 21d, 18h, 29min, 59s)), // 1965 1976 - O Su>=16 3:30 0 -
      "[1972-10-21 18:30:00, 1973-04-21 19:30:00) 08:00:00 0min HKT",
      tz->get_info(to_sys_seconds(1972y, std::chrono::October, 21d, 18h, 30min)),     // 1965 1976 - O Su>=16 3:30 0 -
      tz->get_info(to_sys_seconds(1973y, std::chrono::April, 21d, 19h, 29min, 59s))); // 1965 1976 - Ap Su>=16 3:30 1 S

  assert_range( // 8 HK%sT
      "[1973-04-21 19:30:00, 1973-10-20 18:30:00) 09:00:00 60min HKST",
      tz->get_info(to_sys_seconds(1973y, std::chrono::April, 21d, 19h, 30min)), // 1965 1976 - Ap Su>=16 3:30 1 S
      tz->get_info(to_sys_seconds(1973y, std::chrono::October, 20d, 18h, 29min, 59s))); // 1965 1976 - O Su>=16 3:30 0 -

  assert_range( // 8 HK%sT, test "1973 o - D 30 3:30 1 S"
      "[1973-10-20 18:30:00, 1973-12-29 19:30:00) 08:00:00 0min HKT",
      tz->get_info(to_sys_seconds(1973y, std::chrono::October, 20d, 18h, 30min)),        // 1965 1976 - O Su>=16 3:30
      tz->get_info(to_sys_seconds(1973y, std::chrono::December, 29d, 19h, 29min, 59s))); // 1973 o - D 30 3:30 1 S

  assert_range( // 8 HK%sT
      "[1973-12-29 19:30:00, 1974-10-19 18:30:00) 09:00:00 60min HKST",
      tz->get_info(to_sys_seconds(1973y, std::chrono::December, 29d, 19h, 30min)),      // 1973 o - D 30 3:30 1 S
      tz->get_info(to_sys_seconds(1974y, std::chrono::October, 19d, 18h, 29min, 59s))); // 1965 1976 - O Su>=16 3:30

  assert_range( // 8 HK%sT, between 1973 and 1979 no rule is active so falls back to default
      "[1976-04-17 19:30:00, 1976-10-16 18:30:00) 09:00:00 60min HKST",
      tz->get_info(to_sys_seconds(1976y, std::chrono::April, 17d, 19h, 30min)), // 1965 1976 - Ap Su>=16 3:30 1 S
      tz->get_info(to_sys_seconds(1976y, std::chrono::October, 16d, 18h, 29min, 59s))); // 1965 1976 - O Su>=16 3:30 0 -

  assert_range( // 8 HK%sT, between 1973 and 1979 no rule is active so falls back to default
      "[1976-10-16 18:30:00, 1979-05-12 19:30:00) 08:00:00 0min HKT",
      tz->get_info(to_sys_seconds(1976y, std::chrono::October, 16d, 18h, 30min)),   // 1965 1976 - O Su>=16 3:30 0 -
      tz->get_info(to_sys_seconds(1979y, std::chrono::May, 12d, 19h, 29min, 59s))); // 1979 o - May 13 3:30 1 S

  assert_range( // 8 HK%sT
      "[1979-05-12 19:30:00, 1979-10-20 18:30:00) 09:00:00 60min HKST",
      tz->get_info(to_sys_seconds(1979y, std::chrono::May, 12d, 19h, 30min)),           // 1979 o - May 13 3:30 1 S
      tz->get_info(to_sys_seconds(1979y, std::chrono::October, 20d, 18h, 29min, 59s))); // 1979 o - O 21 3:30 0 -

  assert_equal(
      std::chrono::sys_info(
          to_sys_seconds(1979y, std::chrono::October, 20d, 18h, 30min),
          std::chrono::sys_seconds::max(),
          8h,
          std::chrono::minutes(0),
          "HKT"),
      tz->get_info(to_sys_seconds(1979y, std::chrono::October, 20d, 18h, 30min)));

  assert_equal(
      std::chrono::sys_info(
          to_sys_seconds(1979y, std::chrono::October, 20d, 18h, 30min),
          std::chrono::sys_seconds::max(),
          8h,
          std::chrono::minutes(0),
          "HKT"),
      tz->get_info(std::chrono::sys_seconds::max() - std::chrono::seconds{1})); // max is not valid
}

static void test_europe_berlin() {
  // A more typical entry, first some hard-coded entires and then at the
  // end a rules based entry. This rule is valid for its entire period
  //

  // Z Europe/Berlin 0:53:28 - LMT 1893 Ap
  // 1 c CE%sT 1945 May 24 2
  // 1 So CE%sT 1946
  // 1 DE CE%sT 1980
  // 1 E CE%sT
  //
  // R c 1916 o - Ap 30 23 1 S
  // R c 1916 o - O 1 1 0 -
  // R c 1917 1918 - Ap M>=15 2s 1 S
  // R c 1917 1918 - S M>=15 2s 0 -
  // R c 1940 o - Ap 1 2s 1 S
  // R c 1942 o - N 2 2s 0 -
  // R c 1943 o - Mar 29 2s 1 S
  // R c 1943 o - O 4 2s 0 -
  // R c 1944 1945 - Ap M>=1 2s 1 S
  // R c 1944 o - O 2 2s 0 -
  // R c 1945 o - S 16 2s 0 -
  // R c 1977 1980 - Ap Su>=1 2s 1 S
  // R c 1977 o - S lastSu 2s 0 -
  // R c 1978 o - O 1 2s 0 -
  // R c 1979 1995 - S lastSu 2s 0 -
  // R c 1981 ma - Mar lastSu 2s 1 S
  // R c 1996 ma - O lastSu 2s 0 -
  //
  // R So 1945 o - May 24 2 2 M
  // R So 1945 o - S 24 3 1 S
  // R So 1945 o - N 18 2s 0 -
  //
  // R DE 1946 o - Ap 14 2s 1 S
  // R DE 1946 o - O 7 2s 0 -
  // R DE 1947 1949 - O Su>=1 2s 0 -
  // R DE 1947 o - Ap 6 3s 1 S
  // R DE 1947 o - May 11 2s 2 M
  // R DE 1947 o - Jun 29 3 1 S
  // R DE 1948 o - Ap 18 2s 1 S
  // R DE 1949 o - Ap 10 2s 1 S
  //
  // R E 1977 1980 - Ap Su>=1 1u 1 S
  // R E 1977 o - S lastSu 1u 0 -
  // R E 1978 o - O 1 1u 0 -
  // R E 1979 1995 - S lastSu 1u 0 -
  // R E 1981 ma - Mar lastSu 1u 1 S
  // R E 1996 ma - O lastSu 1u 0 -
  //
  // Note the European Union decided to stop the seasonal change in
  // 2021. In 2023 seasonal changes are still in effect.

  using namespace std::literals::chrono_literals;
  const std::chrono::time_zone* tz = std::chrono::locate_zone("Europe/Berlin");

  assert_equal(
      std::chrono::sys_info(
          std::chrono::sys_seconds::min(),
          to_sys_seconds(1893y, std::chrono::March, 31d, 23h, 6min, 32s), // 0:53:28 - LMT 1893 Ap
          53min + 28s,
          0min,
          "LMT"),
      tz->get_info(std::chrono::sys_seconds::min()));

  assert_equal(
      std::chrono::sys_info(
          std::chrono::sys_seconds::min(),
          to_sys_seconds(1893y, std::chrono::March, 31d, 23h, 6min, 32s), // 0:53:28 - LMT 1893 Ap
          53min + 28s,
          0min,
          "LMT"),
      tz->get_info(to_sys_seconds(1893y, std::chrono::March, 31d, 23h, 6min, 31s)));

  assert_range(
      // 1 CE%sT before 1916 o - Ap 30 23 1 S
      "[1893-03-31 23:06:32, 1916-04-30 22:00:00) 01:00:00 0min CET",
      tz->get_info(to_sys_seconds(1893y, std::chrono::March, 31d, 23h, 6min, 32s)),
      tz->get_info(to_sys_seconds(1916y, std::chrono::April, 30d, 21h, 59min, 59s)));

  assert_cycle(
      // 1 CE%sT
      "[1916-04-30 22:00:00, 1916-09-30 23:00:00) 02:00:00 60min CEST",
      tz->get_info(to_sys_seconds(1916y, std::chrono::April, 30d, 22h)),                 // 1916 o - Ap 30 23 1 S
      tz->get_info(to_sys_seconds(1916y, std::chrono::September, 30d, 22h, 59min, 59s)), // o - O 1 1 0 -
      "[1916-09-30 23:00:00, 1917-04-16 01:00:00) 01:00:00 0min CET",
      tz->get_info(to_sys_seconds(1916y, std::chrono::September, 30d, 23h)),         // o - O 1 1 0 -
      tz->get_info(to_sys_seconds(1917y, std::chrono::April, 16d, 0h, 59min, 59s))); // 1917 1918 - Ap M>=15 2s 1 S

  assert_cycle(
      // 1 CE%sT
      "[1917-04-16 01:00:00, 1917-09-17 01:00:00) 02:00:00 60min CEST",
      tz->get_info(to_sys_seconds(1917y, std::chrono::April, 16d, 1h)),                 // 1917 1918 Ap M>=15 2s 1 S
      tz->get_info(to_sys_seconds(1917y, std::chrono::September, 17d, 0h, 59min, 59s)), // 1917 1918 S M>=15 2s 0 -
      "[1917-09-17 01:00:00, 1918-04-15 01:00:00) 01:00:00 0min CET",
      tz->get_info(to_sys_seconds(1917y, std::chrono::September, 17d, 1h)),          // 1917 1918 S M>=15 2s 0 -
      tz->get_info(to_sys_seconds(1918y, std::chrono::April, 15d, 0h, 59min, 59s))); // 1917 1918 Ap M>=15 2s 1 S

  assert_cycle(
      // 1 CE%sT (The cycle is more than 1 year)
      "[1918-04-15 01:00:00, 1918-09-16 01:00:00) 02:00:00 60min CEST",
      tz->get_info(to_sys_seconds(1918y, std::chrono::April, 15d, 1h)),                 // 1917 1918 Ap M>=15 2s 1 S
      tz->get_info(to_sys_seconds(1918y, std::chrono::September, 16d, 0h, 59min, 59s)), // 1917 1918 S M>=15 2s 0 -
      "[1918-09-16 01:00:00, 1940-04-01 01:00:00) 01:00:00 0min CET",
      tz->get_info(to_sys_seconds(1918y, std::chrono::September, 16d, 1h)),         // 1917 1918 S M>=15 2s 0 -
      tz->get_info(to_sys_seconds(1940y, std::chrono::April, 1d, 0h, 59min, 59s))); // 1940 o Ap 1 2s 1 S

  assert_cycle(
      // 1 CE%sT (The cycle is more than 1 year)
      "[1940-04-01 01:00:00, 1942-11-02 01:00:00) 02:00:00 60min CEST",
      tz->get_info(to_sys_seconds(1940y, std::chrono::April, 1d, 1h)),                // 1940 o Ap 1 2s 1 S
      tz->get_info(to_sys_seconds(1942y, std::chrono::November, 2d, 0h, 59min, 59s)), // 1942 o N 2 2s 0 -
      "[1942-11-02 01:00:00, 1943-03-29 01:00:00) 01:00:00 0min CET",
      tz->get_info(to_sys_seconds(1942y, std::chrono::November, 2d, 1h)),            // 1942 o N 2 2s 0 -
      tz->get_info(to_sys_seconds(1943y, std::chrono::March, 29d, 0h, 59min, 59s))); // 1943 o Mar 29 2s 1 S

  assert_range(
      // Here the zone changes from c (C-Eur) to So (SovietZone).
      // The rule c ends on 1945-09-16, instead it ends at the zone change date/time
      // There is a tricky part in the time
      // "1 c CE%sT" has an offset of 1 at the moment the rule
      // ends there is a save of 60 minutes. This means the
      // local offset to UTC is 2 hours. The rule ends at
      // 1945-05-24 02:00:00 local time, which is
      // 1945-05-24 00:00:00 UTC.
      "[1945-04-02 01:00:00, 1945-05-24 00:00:00) 02:00:00 60min CEST",
      tz->get_info(to_sys_seconds(1945y, std::chrono::April, 2d, 1h)),              // 1 CE%sT & 1945 Ap M>=1 2s 1 S
      tz->get_info(to_sys_seconds(1945y, std::chrono::May, 23d, 23h, 59min, 59s))); // 1 c CE%sT & 1945 May 24 2

  assert_range( // --
      "[1945-05-24 00:00:00, 1945-09-24 00:00:00) 03:00:00 120min CEMT",
      tz->get_info(to_sys_seconds(1945y, std::chrono::May, 24d)),                         // 1 c CE%sT & 1945 May 24 2
      tz->get_info(to_sys_seconds(1945y, std::chrono::September, 23d, 23h, 59min, 59s))); // 1945 o S 24 3 1 S

  assert_range(
      // 1 c CE%sT 1945 May 24 2
      "[1945-09-24 00:00:00, 1945-11-18 01:00:00) 02:00:00 60min CEST",
      tz->get_info(to_sys_seconds(1945y, std::chrono::September, 24d)),                 // 1945 o S 24 3 1 S
      tz->get_info(to_sys_seconds(1945y, std::chrono::November, 18d, 0h, 59min, 59s))); // 1945 o N 18 2s 0 -
  assert_range(                                                                         // --
                                                                                        // Merges 2 continuations
      "[1945-11-18 01:00:00, 1946-04-14 01:00:00) 01:00:00 0min CET",
      tz->get_info(to_sys_seconds(1945y, std::chrono::November, 18d, 1h)),           // 1 c CE%sT & 1945 o N 18 2s 0 -
      tz->get_info(to_sys_seconds(1946y, std::chrono::April, 14d, 0h, 59min, 59s))); // 1 So CE%sT & 1946 o Ap 14 2s 1 S

  assert_range(
      // 1 DE CE%sT 1980
      "[1946-04-14 01:00:00, 1946-10-07 01:00:00) 02:00:00 60min CEST",
      tz->get_info(to_sys_seconds(1946y, std::chrono::April, 14d, 1h)),               // 1946 o Ap 14 2s 1 S
      tz->get_info(to_sys_seconds(1946y, std::chrono::October, 7d, 0h, 59min, 59s))); // 1946 o O 7 2s 0 -

  // Note 1947 is an interesting year with 4 rules
  // R DE 1947 1949 - O Su>=1 2s 0 -
  // R DE 1947 o - Ap 6 3s 1 S
  // R DE 1947 o - May 11 2s 2 M
  // R DE 1947 o - Jun 29 3 1 S
  assert_range(
      // 1 DE CE%sT 1980
      "[1946-10-07 01:00:00, 1947-04-06 02:00:00) 01:00:00 0min CET",
      tz->get_info(to_sys_seconds(1946y, std::chrono::October, 7d, 1h)),            // 1946 o O 7 2s 0 -
      tz->get_info(to_sys_seconds(1947y, std::chrono::April, 6d, 1h, 59min, 59s))); // 1947 o Ap 6 3s 1 S

  assert_range(
      // 1 DE CE%sT 1980
      "[1947-04-06 02:00:00, 1947-05-11 01:00:00) 02:00:00 60min CEST",
      tz->get_info(to_sys_seconds(1947y, std::chrono::April, 6d, 2h)),             // 1947 o Ap 6 3s 1 S
      tz->get_info(to_sys_seconds(1947y, std::chrono::May, 11d, 0h, 59min, 59s))); // 1947 o May 11 2s 2 M

  assert_range(
      // 1 DE CE%sT 1980
      "[1947-05-11 01:00:00, 1947-06-29 00:00:00) 03:00:00 120min CEMT",
      tz->get_info(to_sys_seconds(1947y, std::chrono::May, 11d, 1h)),                // 1947 o May 11 2s 2 M
      tz->get_info(to_sys_seconds(1947y, std::chrono::June, 28d, 23h, 59min, 59s))); // 1947 o Jun 29 3 1 S

  assert_cycle(
      // 1 DE CE%sT 1980
      "[1947-06-29 00:00:00, 1947-10-05 01:00:00) 02:00:00 60min CEST",
      tz->get_info(to_sys_seconds(1947y, std::chrono::June, 29d)),                   // 1947 o Jun 29 3 1 S
      tz->get_info(to_sys_seconds(1947y, std::chrono::October, 5d, 0h, 59min, 59s)), // 1947 1949 O Su>=1 2s 0 -
      "[1947-10-05 01:00:00, 1948-04-18 01:00:00) 01:00:00 0min CET",
      tz->get_info(to_sys_seconds(1947y, std::chrono::October, 5d, 1h)),             // 1947 1949 O Su>=1 2s 0 -
      tz->get_info(to_sys_seconds(1948y, std::chrono::April, 18d, 0h, 59min, 59s))); // 1948 o Ap 18 2s 1 S

  assert_cycle(
      // 1 DE CE%sT 1980
      "[1948-04-18 01:00:00, 1948-10-03 01:00:00) 02:00:00 60min CEST",
      tz->get_info(to_sys_seconds(1948y, std::chrono::April, 18d, 1h)),              // 1948 o Ap 18 2s 1 S
      tz->get_info(to_sys_seconds(1948y, std::chrono::October, 3d, 0h, 59min, 59s)), // 1947 1949 O Su>=1 2s 0 -
      "[1948-10-03 01:00:00, 1949-04-10 01:00:00) 01:00:00 0min CET",
      tz->get_info(to_sys_seconds(1948y, std::chrono::October, 3d, 1h)),             // 1947 1949 O Su>=1 2s 0 -
      tz->get_info(to_sys_seconds(1949y, std::chrono::April, 10d, 0h, 59min, 59s))); // 1949 o Ap 10 2s 1 S

  assert_cycle( // Note the end time is  in a different continuation.
      "[1949-04-10 01:00:00, 1949-10-02 01:00:00) 02:00:00 60min CEST",              // 1 DE CE%sT 1980
      tz->get_info(to_sys_seconds(1949y, std::chrono::April, 10d, 1h)),              //  1949 o Ap 10 2s 1 S
      tz->get_info(to_sys_seconds(1949y, std::chrono::October, 2d, 0h, 59min, 59s)), //  1947 1949 O Su>=1 2s 0 -
      "[1949-10-02 01:00:00, 1980-04-06 01:00:00) 01:00:00 0min CET",
      tz->get_info(to_sys_seconds(1949y, std::chrono::October, 2d, 1h)),   //  1947 1949 O Su>=1 2s 0 -
      tz->get_info(                                                        // 1 E CE%sT
          to_sys_seconds(1980y, std::chrono::April, 6d, 0h, 59min, 59s))); //  1977 1980 Ap Su>=1 1u 1 S

  assert_cycle(
      // 1 E CE%sT
      "[2020-03-29 01:00:00, 2020-10-25 01:00:00) 02:00:00 60min CEST",
      tz->get_info(to_sys_seconds(2020y, std::chrono::March, 29d, 1h)),               // 1981 ma Mar lastSu 1u 1 S
      tz->get_info(to_sys_seconds(2020y, std::chrono::October, 25d, 0h, 59min, 59s)), // 1996 ma O lastSu 1u 0 -
      "[2020-10-25 01:00:00, 2021-03-28 01:00:00) 01:00:00 0min CET",
      tz->get_info(to_sys_seconds(2020y, std::chrono::October, 25d, 1h)),            // 1996 ma O lastSu 1u 0 -
      tz->get_info(to_sys_seconds(2021y, std::chrono::March, 28d, 0h, 59min, 59s))); // 1981 ma Mar lastSu 1u 1 S

  assert_cycle(
      // 1 E CE%sT
      "[2021-03-28 01:00:00, 2021-10-31 01:00:00) 02:00:00 60min CEST",
      tz->get_info(to_sys_seconds(2021y, std::chrono::March, 28d, 1h)),               // 1981 ma Mar lastSu 1u 1 S
      tz->get_info(to_sys_seconds(2021y, std::chrono::October, 31d, 0h, 59min, 59s)), // 1996 ma O lastSu 1u 0 -
      "[2021-10-31 01:00:00, 2022-03-27 01:00:00) 01:00:00 0min CET",
      tz->get_info(to_sys_seconds(2021y, std::chrono::October, 31d, 1h)),            // 1996 ma O lastSu 1u 0 -
      tz->get_info(to_sys_seconds(2022y, std::chrono::March, 27d, 0h, 59min, 59s))); // 1981 ma Mar lastSu 1u 1 S
}

static void test_america_st_johns() {
  // A more typical entry,
  // Uses letters both when DST is ative and not and has multiple
  // letters. Uses negetive offsets.
  // Switches several times between their own and Canadian rules
  // Switches the stdoff from -3:30:52 to -3:30 while observing the same rule

  // Z America/St_Johns -3:30:52 - LMT 1884
  // -3:30:52 j N%sT 1918
  // -3:30:52 C N%sT 1919
  // -3:30:52 j N%sT 1935 Mar 30
  // -3:30 j N%sT 1942 May 11
  // -3:30 C N%sT 1946
  // -3:30 j N%sT 2011 N
  // -3:30 C N%sT
  //
  // R j 1917 o - Ap 8 2 1 D
  // R j 1917 o - S 17 2 0 S
  // R j 1919 o - May 5 23 1 D
  // R j 1919 o - Au 12 23 0 S
  // R j 1920 1935 - May Su>=1 23 1 D
  // R j 1920 1935 - O lastSu 23 0 S
  // R j 1936 1941 - May M>=9 0 1 D
  // R j 1936 1941 - O M>=2 0 0 S
  // R j 1946 1950 - May Su>=8 2 1 D
  // R j 1946 1950 - O Su>=2 2 0 S
  // R j 1951 1986 - Ap lastSu 2 1 D
  // R j 1951 1959 - S lastSu 2 0 S
  // R j 1960 1986 - O lastSu 2 0 S
  // R j 1987 o - Ap Su>=1 0:1 1 D
  // R j 1987 2006 - O lastSu 0:1 0 S
  // R j 1988 o - Ap Su>=1 0:1 2 DD
  // R j 1989 2006 - Ap Su>=1 0:1 1 D
  // R j 2007 2011 - Mar Su>=8 0:1 1 D
  // R j 2007 2010 - N Su>=1 0:1 0 S
  //
  // R C 1918 o - Ap 14 2 1 D
  // R C 1918 o - O 27 2 0 S
  // R C 1942 o - F 9 2 1 W
  // R C 1945 o - Au 14 23u 1 P
  // R C 1945 o - S 30 2 0 S
  // R C 1974 1986 - Ap lastSu 2 1 D
  // R C 1974 2006 - O lastSu 2 0 S
  // R C 1987 2006 - Ap Su>=1 2 1 D
  // R C 2007 ma - Mar Su>=8 2 1 D
  // R C 2007 ma - N Su>=1 2 0 S

  using namespace std::literals::chrono_literals;
  const std::chrono::time_zone* tz = std::chrono::locate_zone("America/St_Johns");

  assert_equal( // --
      std::chrono::sys_info(
          std::chrono::sys_seconds::min(),
          to_sys_seconds(1884y, std::chrono::January, 1d, 3h, 30min, 52s), // -3:30:52 - LMT 1884
          -(3h + 30min + 52s),
          0min,
          "LMT"),
      tz->get_info(std::chrono::sys_seconds::min()));

  assert_equal( // --
      std::chrono::sys_info(
          std::chrono::sys_seconds::min(),
          to_sys_seconds(1884y, std::chrono::January, 1d, 3h, 30min, 52s), // -3:30:52 - LMT 1884
          -(3h + 30min + 52s),
          0min,
          "LMT"),
      tz->get_info(to_sys_seconds(1884y, std::chrono::January, 1d, 3h, 30min, 51s)));

  assert_range( // -3:30:52 j N%sT 1918
      "[1884-01-01 03:30:52, 1917-04-08 05:30:52) -03:30:52 0min NST",
      tz->get_info(to_sys_seconds(1884y, std::chrono::January, 1d, 3h, 30min, 52s)), // no rule active
      tz->get_info(to_sys_seconds(1917y, std::chrono::April, 8d, 5h, 30min, 51s)));  // 1917 o Ap 8 2 1 D

  assert_range( // -3:30:52 j N%sT 1918
      "[1917-04-08 05:30:52, 1917-09-17 04:30:52) -02:30:52 60min NDT",
      tz->get_info(to_sys_seconds(1917y, std::chrono::April, 8d, 5h, 30min, 52s)),       // 1917 o Ap 8 2 1 D
      tz->get_info(to_sys_seconds(1917y, std::chrono::September, 17d, 4h, 30min, 51s))); // 1917 o S 17 2 0 S

  assert_range("[1917-09-17 04:30:52, 1918-04-14 05:30:52) -03:30:52 0min NST",
               tz->get_info(                                                            // -3:30:52 j N%sT 1918
                   to_sys_seconds(1917y, std::chrono::September, 17d, 4h, 30min, 52s)), //   1917 o S 17 2 0 S
               tz->get_info(                                                            // -3:30:52 C N%sT 1919
                   to_sys_seconds(1918y, std::chrono::April, 14d, 5h, 30min, 51s)));    //   1918 o Ap 14 2 1 D

  assert_range( // -3:30:52 C N%sT 1919
      "[1918-04-14 05:30:52, 1918-10-27 04:30:52) -02:30:52 60min NDT",
      tz->get_info(to_sys_seconds(1918y, std::chrono::April, 14d, 5h, 30min, 52s)),    // 1918 o Ap 14 2 1 D
      tz->get_info(to_sys_seconds(1918y, std::chrono::October, 27d, 4h, 30min, 51s))); // 1918 o O 27 2 0 S

  assert_range("[1918-10-27 04:30:52, 1919-05-06 02:30:52) -03:30:52 0min NST",
               tz->get_info(                                                          // -3:30:52 C N%sT 1919
                   to_sys_seconds(1918y, std::chrono::October, 27d, 4h, 30min, 52s)), //   1918 o O 27 2 0 S
               tz->get_info(                                                          // -3:30:52 j N%sT 1935 Mar 30
                   to_sys_seconds(1919y, std::chrono::May, 6d, 2h, 30min, 51s)));     //   1919 o May 5 23 1 D

  assert_range( // -3:30:52 j N%sT 1935 Mar 30
      "[1934-10-29 01:30:52, 1935-03-30 03:30:52) -03:30:52 0min NST",
      tz->get_info(to_sys_seconds(1934y, std::chrono::October, 29d, 1h, 30min, 52s)), // 1920 1935 O lastSu 23 0 S
      tz->get_info(to_sys_seconds(1935y, std::chrono::March, 30d, 3h, 30min, 51s)));  // 1920 1935 May Su>=1 23 1 D

  assert_range( // -3:30 j N%sT 1942 May 11
                // Changed the stdoff while the same rule remains active.
      "[1935-03-30 03:30:52, 1935-05-06 02:30:00) -03:30:00 0min NST",
      tz->get_info(to_sys_seconds(1935y, std::chrono::March, 30d, 3h, 30min, 52s)), // 1920 1935 O lastSu 23 0 S
      tz->get_info(to_sys_seconds(1935y, std::chrono::May, 6d, 2h, 29min, 59s)));   // 1920 1935 May Su>=1 23 1 D

  assert_range( // -3:30 j N%sT 1942 May 11
      "[1935-05-06 02:30:00, 1935-10-28 01:30:00) -02:30:00 60min NDT",
      tz->get_info(to_sys_seconds(1935y, std::chrono::May, 6d, 2h, 30min, 0s)),        // 1920 1935 May Su>=1 23 1 D
      tz->get_info(to_sys_seconds(1935y, std::chrono::October, 28d, 1h, 29min, 59s))); // 1920 1935 O lastSu 23 0 S

  assert_range( // -3:30 j N%sT 1942 May 11
      "[1941-10-06 02:30:00, 1942-05-11 03:30:00) -03:30:00 0min NST",
      tz->get_info(to_sys_seconds(1941y, std::chrono::October, 6d, 2h, 30min, 0s)), // 1936 1941 O M>=2 0 0 S
      tz->get_info(to_sys_seconds(1942y, std::chrono::May, 11d, 3h, 29min, 59s)));  // 1946 1950 May Su>=8 2 1 D

  assert_range( // -3:30 C N%sT 1946
      "[1942-05-11 03:30:00, 1945-08-14 23:00:00) -02:30:00 60min NWT",
      tz->get_info(to_sys_seconds(1942y, std::chrono::May, 11d, 3h, 30min, 0s)),       // 1942 o F 9 2 1 W
      tz->get_info(to_sys_seconds(1945y, std::chrono::August, 14d, 22h, 59min, 59s))); // 1945 o Au 14 23u 1 P

  assert_range( // -3:30 C N%sT 1946
      "[1945-08-14 23:00:00, 1945-09-30 04:30:00) -02:30:00 60min NPT",
      tz->get_info(to_sys_seconds(1945y, std::chrono::August, 14d, 23h, 0min, 0s)),      // 1945 o Au 14 23u 1 P
      tz->get_info(to_sys_seconds(1945y, std::chrono::September, 30d, 4h, 29min, 59s))); // 1945 o S 30 2 0 S

  assert_range(
      "[1945-09-30 04:30:00, 1946-05-12 05:30:00) -03:30:00 0min NST",
      tz->get_info(
          to_sys_seconds(1945y, std::chrono::September, 30d, 4h, 30min, 0s)), // -3:30 C N%sT 1946 & 945 o S 30 2 0 S
      tz->get_info(to_sys_seconds(
          1946y, std::chrono::May, 12d, 5h, 29min, 59s))); // -3:30 j N%sT 2011 N & 1946 1950 May Su>=8 2 1 D

  assert_range( // -3:30 j N%sT 2011 N
      "[1988-04-03 03:31:00, 1988-10-30 01:31:00) -01:30:00 120min NDDT",
      tz->get_info(to_sys_seconds(1988y, std::chrono::April, 3d, 3h, 31min, 0s)),      // 1988 o Ap Su>=1 0:1 2 DD
      tz->get_info(to_sys_seconds(1988y, std::chrono::October, 30d, 1h, 30min, 59s))); // 1987 2006 O lastSu 0:1 0 S

  assert_range("[2011-03-13 03:31:00, 2011-11-06 04:30:00) -02:30:00 60min NDT",
               tz->get_info(                                                            // -3:30 j N%sT 2011 N
                   to_sys_seconds(2011y, std::chrono::March, 13d, 3h, 31min, 0s)),      //   2007 2011 Mar Su>=8 0:1 1 D
               tz->get_info(                                                            // -3:30 C N%sT
                   to_sys_seconds(2011y, std::chrono::November, 6d, 04h, 29min, 59s))); //   2007 ma N Su>=1 2 0 S
}

static void test_get_at_standard_time_universal() {
  // Z Asia/Barnaul 5:35 - LMT 1919 D 10
  // ...
  // 7 R +07/+08 1995 May 28
  // 6 R +06/+07 2011 Mar 27 2s
  // ...
  //
  // ...
  // R R 1985 2010 - Mar lastSu 2s 1 S
  // R R 1996 2010 - O lastSu 2s 0 -

  using namespace std::literals::chrono_literals;
  const std::chrono::time_zone* tz = std::chrono::locate_zone("Asia/Barnaul");

  assert_equal(
      std::chrono::sys_info(
          to_sys_seconds(2010y, std::chrono::October, 30d, 20h),
          to_sys_seconds(2011y, std::chrono::March, 26d, 20h),
          6h,
          0min,
          "+06"),
      tz->get_info(to_sys_seconds(2010y, std::chrono::October, 31d, 10h)));
}

static void test_get_at_standard_time_standard() {
  // Z Africa/Bissau -1:2:20 - LMT 1912 Ja 1 1u
  using namespace std::literals::chrono_literals;
  const std::chrono::time_zone* tz = std::chrono::locate_zone("Africa/Bissau");

  assert_equal(
      std::chrono::sys_info(
          std::chrono::sys_seconds::min(),
          to_sys_seconds(1912y, std::chrono::January, 1d, 1h),
          -(1h + 2min + 20s),
          0min,
          "LMT"),
      tz->get_info(std::chrono::sys_seconds::min()));
}

static void test_get_at_save_universal() {
  // Z America/Tijuana -7:48:4 - LMT 1922 Ja 1 0:11:56
  // -7 - MST 1924
  // -8 - PST 1927 Jun 10 23
  // -7 - MST 1930 N 15
  // -8 - PST 1931 Ap
  // -8 1 PDT 1931 S 30
  // -8 - PST 1942 Ap 24
  // -8 1 PWT 1945 Au 14 23u
  // ...

  using namespace std::literals::chrono_literals;
  const std::chrono::time_zone* tz = std::chrono::locate_zone("America/Tijuana");

  assert_equal(
      std::chrono::sys_info(
          to_sys_seconds(1942y, std::chrono::April, 24d, 8h),
          to_sys_seconds(1945y, std::chrono::August, 14d, 23h),
          -7h,
          60min,
          "PWT"),
      tz->get_info(to_sys_seconds(1942y, std::chrono::April, 24d, 8h)));
}

static void test_get_at_rule_standard() {
  // Z Antarctica/Macquarie 0 - -00 1899 N
  // 10 - AEST 1916 O 1 2
  // 10 1 AEDT 1917 F
  // 10 AU AE%sT 1919 Ap 1 0s
  // ...
  //
  // R AU 1917 o - Ja 1 2s 1 D
  // R AU 1917 o - Mar lastSu 2s 0 S
  // R AU 1942 o - Ja 1 2s 1 D
  // ...

  using namespace std::literals::chrono_literals;
  const std::chrono::time_zone* tz = std::chrono::locate_zone("Antarctica/Macquarie");

  // Another rule where the S propagates?
  assert_equal(
      std::chrono::sys_info(
          to_sys_seconds(1916y, std::chrono::September, 30d, 16h),
          to_sys_seconds(1917y, std::chrono::March, 24d, 16h),
          11h,
          60min,
          "AEDT"),
      tz->get_info(to_sys_seconds(1916y, std::chrono::September, 30d, 16h)));
}

static void test_get_at_rule_universal() {
  // Z America/Nuuk -3:26:56 - LMT 1916 Jul 28
  // -3 - -03 1980 Ap 6 2
  // -3 E -03/-02 2023 O 29 1u
  // -2 E -02/-01
  //
  // R E 1977 1980 - Ap Su>=1 1u 1 S
  // R E 1977 o - S lastSu 1u 0 -
  // R E 1978 o - O 1 1u 0 -
  // R E 1979 1995 - S lastSu 1u 0 -
  // R E 1981 ma - Mar lastSu 1u 1 S
  // R E 1996 ma - O lastSu 1u 0 -

  using namespace std::literals::chrono_literals;
  const std::chrono::time_zone* tz = std::chrono::locate_zone("America/Nuuk");

  assert_equal(
      std::chrono::sys_info(
          to_sys_seconds(1980y, std::chrono::April, 6d, 5h),
          to_sys_seconds(1980y, std::chrono::September, 28d, 1h),
          -2h,
          60min,
          "-02"),
      tz->get_info(to_sys_seconds(1980y, std::chrono::April, 6d, 5h)));
}

static void test_format_with_alternatives_west() {
  // Z America/Nuuk -3:26:56 - LMT 1916 Jul 28
  // -3 - -03 1980 Ap 6 2
  // -3 E -03/-02 2023 O 29 1u
  // -2 E -02/-01
  //
  // ...
  // R E 1981 ma - Mar lastSu 1u 1 S
  // R E 1996 ma - O lastSu 1u 0 -

  using namespace std::literals::chrono_literals;
  const std::chrono::time_zone* tz = std::chrono::locate_zone("America/Nuuk");

  assert_cycle( // -3 E -03/-02
      "[2019-10-27 01:00:00, 2020-03-29 01:00:00) -03:00:00 0min -03",
      tz->get_info(to_sys_seconds(2019y, std::chrono::October, 27d, 1h)),           // 1981 ma Mar lastSu 1u 1 S
      tz->get_info(to_sys_seconds(2020y, std::chrono::March, 29d, 0h, 59min, 59s)), // 1996 ma O lastSu 1u 0 -
      "[2020-03-29 01:00:00, 2020-10-25 01:00:00) -02:00:00 60min -02",
      tz->get_info(to_sys_seconds(2020y, std::chrono::March, 29d, 1h)),                // 1996 ma O lastSu 1u 0 -
      tz->get_info(to_sys_seconds(2020y, std::chrono::October, 25d, 0h, 59min, 59s))); // 1981 ma Mar lastSu 1u 1 S
}

static void test_format_with_alternatives_east() {
  // Z Asia/Barnaul 5:35 - LMT 1919 D 10
  // ...
  // 6 R +06/+07 2011 Mar 27 2s
  // ...
  //
  // ...
  // R R 1985 2010 - Mar lastSu 2s 1 S
  // R R 1996 2010 - O lastSu 2s 0 -

  using namespace std::literals::chrono_literals;
  const std::chrono::time_zone* tz = std::chrono::locate_zone("Asia/Barnaul");

  assert_cycle( // 6 R +06/+07 2011 Mar 27 2s
      "[2000-03-25 20:00:00, 2000-10-28 20:00:00) 07:00:00 60min +07",
      tz->get_info(to_sys_seconds(2000y, std::chrono::March, 25d, 20h)),               // 1985 2010 Mar lastSu 2s 1 S
      tz->get_info(to_sys_seconds(2000y, std::chrono::October, 28d, 19h, 59min, 59s)), // 1996 2010 O lastSu 2s 0 -
      "[2000-10-28 20:00:00, 2001-03-24 20:00:00) 06:00:00 0min +06",
      tz->get_info(to_sys_seconds(2000y, std::chrono::October, 28d, 20h)),            // 1996 2010 O lastSu 2s 0 -
      tz->get_info(to_sys_seconds(2001y, std::chrono::March, 24d, 19h, 59min, 59s))); // 1985 2010 Mar lastSu 2s 1 S
}

static void test_africa_algiers() {
  using namespace std::literals::chrono_literals;
  const std::chrono::time_zone* tz = std::chrono::locate_zone("Africa/Algiers");

  assert_equal(
      std::chrono::sys_info(
          to_sys_seconds(1977y, std::chrono::October, 20d, 23h),
          to_sys_seconds(1978y, std::chrono::March, 24d),
          1h,
          std::chrono::minutes(0),
          "CET"),
      tz->get_info(to_sys_seconds(1977y, std::chrono::October, 20d, 23h)));

  assert_range("[1977-05-06 00:00:00, 1977-10-20 23:00:00) 01:00:00 60min WEST", // 0 d WE%sT 1977 O 21
               tz->get_info(to_sys_seconds(1977y, std::chrono::May, 6d)),
               tz->get_info(to_sys_seconds(1977y, std::chrono::October, 20d, 22h, 59min, 59s)));

  assert_range("[1977-10-20 23:00:00, 1978-03-24 00:00:00) 01:00:00 0min CET", // 1 d CE%sT 1979 O 26
               tz->get_info(to_sys_seconds(1977y, std::chrono::October, 20d, 23h)),
               tz->get_info(to_sys_seconds(1978y, std::chrono::March, 23d, 23h, 59min, 59s)));
}

static void test_africa_casablanca() {
  // Z Africa/Casablanca -0:30:20 - LMT 1913 O 26
  // 0 M +00/+01 1984 Mar 16
  // 1 - +01 1986
  // 0 M +00/+01 2018 O 28 3
  // 1 M +01/+00
  //
  // ...
  // R M 2013 2018 - O lastSu 3 0 -
  // R M 2014 2018 - Mar lastSu 2 1 -
  // R M 2014 o - Jun 28 3 0 -
  // R M 2014 o - Au 2 2 1 -
  // R M 2015 o - Jun 14 3 0 -
  // R M 2015 o - Jul 19 2 1 -
  // R M 2016 o - Jun 5 3 0 -
  // R M 2016 o - Jul 10 2 1 -
  // R M 2017 o - May 21 3 0 -
  // R M 2017 o - Jul 2 2 1 -
  // R M 2018 o - May 13 3 0 -
  // R M 2018 o - Jun 17 2 1 -
  // R M 2019 o - May 5 3 -1 -
  // R M 2019 o - Jun 9 2 0 -
  // R M 2020 o - Ap 19 3 -1 -
  // ...

  using namespace std::literals::chrono_literals;
  const std::chrono::time_zone* tz = std::chrono::locate_zone("Africa/Casablanca");

  assert_range("[2018-06-17 02:00:00, 2018-10-28 02:00:00) 01:00:00 60min +01",
               tz->get_info(to_sys_seconds(2018y, std::chrono::June, 17d, 2h)),
               tz->get_info(to_sys_seconds(2018y, std::chrono::October, 28d, 1h, 59min, 59s)));

  assert_range("[2018-10-28 02:00:00, 2019-05-05 02:00:00) 01:00:00 0min +01",
               tz->get_info( // 1 M +01/+00 & R M 2018 o - Jun 17 2 1 -
                   to_sys_seconds(2018y, std::chrono::October, 28d, 2h)),
               tz->get_info( // 1 M +01/+00 & R M 2019 o - May 5 3 -1 -
                   to_sys_seconds(2019y, std::chrono::May, 5d, 1h, 59min, 59s)));

  // 1 M +01/+00
  // Note the SAVE contains a negative value
  assert_range("[2019-05-05 02:00:00, 2019-06-09 02:00:00) 00:00:00 -60min +00",
               tz->get_info(to_sys_seconds(2019y, std::chrono::May, 5d, 2h)),               // R M 2019 o - May 5 3 -1 -
               tz->get_info(to_sys_seconds(2019y, std::chrono::June, 9d, 1h, 59min, 59s))); // R M 2019 o - Jun 9 2 0 -

  assert_range("[2019-06-09 02:00:00, 2020-04-19 02:00:00) 01:00:00 0min +01",
               tz->get_info( // 1 M +01/+00 & R M 2019 o - Jun 9 2 0 -
                   to_sys_seconds(2019y, std::chrono::June, 9d, 2h)),
               tz->get_info( // 1 M +01/+00 & R M 2020 o - Ap 19 3 -1 -
                   to_sys_seconds(2020y, std::chrono::April, 19d, 1h, 59min, 59s))); //
}

static void test_africa_ceuta() {
  // Z Africa/Ceuta -0:21:16 - LMT 1900 D 31 23:38:44
  // 0 - WET 1918 May 6 23
  // 0 1 WEST 1918 O 7 23
  // 0 - WET 1924
  // 0 s WE%sT 1929
  // 0 - WET 1967
  // 0 Sp WE%sT 1984 Mar 16
  // 1 - CET 1986
  // 1 E CE%sT
  //
  // ...
  // R s 1926 o - Ap 17 23 1 S
  // R s 1926 1929 - O Sa>=1 24s 0 -
  // R s 1927 o - Ap 9 23 1 S
  // R s 1928 o - Ap 15 0 1 S
  // R s 1929 o - Ap 20 23 1 S
  // R s 1937 o - Jun 16 23 1 S
  // ...
  //
  // R Sp 1967 o - Jun 3 12 1 S
  // R Sp 1967 o - O 1 0 0 -
  // R Sp 1974 o - Jun 24 0 1 S
  // R Sp 1974 o - S 1 0 0 -
  // R Sp 1976 1977 - May 1 0 1 S
  // R Sp 1976 o - Au 1 0 0 -
  // R Sp 1977 o - S 28 0 0 -
  // R Sp 1978 o - Jun 1 0 1 S
  // R Sp 1978 o - Au 4 0 0 -

  using namespace std::literals::chrono_literals;
  const std::chrono::time_zone* tz = std::chrono::locate_zone("Africa/Ceuta");

  assert_range(

      "[1928-10-07 00:00:00, 1967-06-03 12:00:00) 00:00:00 0min WET",
      tz->get_info(to_sys_seconds(1928y, std::chrono::October, 7d)),       // 0 s WE%sT 1929 & 1926 1929 O Sa>=1 24s 0 -
      tz->get_info(                                                        // No transitions in "0 - WET 1967"
          to_sys_seconds(1967y, std::chrono::June, 3d, 11h, 59min, 59s))); // 0 - WET 1967 & 1967 o Jun 3 12 1 S
}

static void test_africa_freetown() {
  // Z Africa/Freetown -0:53 - LMT 1882
  // -0:53 - FMT 1913 Jul
  // -1 SL %s 1939 S 5
  // -1 - -01 1941 D 6 24
  // 0 - GMT
  //
  // R SL 1932 o - D 1 0 0:20 -0040
  // R SL 1933 1938 - Mar 31 24 0 -01
  // R SL 1933 1939 - Au 31 24 0:20 -0040
  // R SL 1939 o - May 31 24 0 -01

  using namespace std::literals::chrono_literals;
  const std::chrono::time_zone* tz = std::chrono::locate_zone("Africa/Freetown");

  // When a continuation has a named rule, the tranisition time determined by
  // the active rule can be wrong. The next continuation may set the clock to an
  // earlier time. This is tested for San Luis. This tests the rule is not used
  // when the rule is not a named rule.
  //
  // Fixes:
  //   Expected output [1882-01-01 00:53:00, 1913-07-01 00:53:00) -00:53:00 0min FMT
  //   Actual output   [1882-01-01 00:53:00, 1913-07-01 00:46:00) -00:53:00 0min FMT

  assert_range("[1882-01-01 00:53:00, 1913-07-01 00:53:00) -00:53:00 0min FMT",
               tz->get_info(to_sys_seconds(1882y, std::chrono::January, 1d, 0h, 53min)), // -0:53 - FMT 1913 Jul
               tz->get_info( // -1 SL %s 1939 S 5 & before first rule
                   to_sys_seconds(1913y, std::chrono::July, 1d, 0h, 52min, 59s)));

  // Tests whether the "-1 SL %s 1939 S 5" until gets the proper local time
  // adjustment.
  assert_range("[1939-09-01 01:00:00, 1939-09-05 00:40:00) -00:40:00 20min -0040",
               tz->get_info( // -1 SL %s 1939 S 5 & R SL 1933 1939 - Au 31 24 0:20 -0040
                   to_sys_seconds(1939y, std::chrono::September, 1d, 1h)),
               tz->get_info( // -1 - -01 1941 D 6 24
                   to_sys_seconds(1939y, std::chrono::September, 5d, 0h, 39min, 59s)));
}

static void test_africa_windhoek() {
  // Tests the LETTER/S used before the first rule per
  // https://data.iana.org/time-zones/tz-how-to.html
  //   If switching to a named rule before any transition has happened,
  //   assume standard time (SAVE zero), and use the LETTER data from
  //   the earliest transition with a SAVE of zero.

  // Z Africa/Windhoek 1:8:24 - LMT 1892 F 8
  // 1:30 - +0130 1903 Mar
  // 2 - SAST 1942 S 20 2
  // 2 1 SAST 1943 Mar 21 2
  // 2 - SAST 1990 Mar 21
  // 2 NA %s
  //
  // R NA 1994 o - Mar 21 0 -1 WAT
  // R NA 1994 2017 - S Su>=1 2 0 CAT
  // R NA 1995 2017 - Ap Su>=1 2 -1 WAT

  using namespace std::literals::chrono_literals;
  const std::chrono::time_zone* tz = std::chrono::locate_zone("Africa/Windhoek");

  assert_range( // 2 - EET 2012 N 10 2
      "[1990-03-20 22:00:00, 1994-03-20 22:00:00) 02:00:00 0min CAT",
      tz->get_info(to_sys_seconds(1990y, std::chrono::March, 20d, 22h)),
      tz->get_info(to_sys_seconds(1994y, std::chrono::March, 20d, 21h, 59min, 59s)));
}

static void test_america_adak() {
  // Z America/Adak 12:13:22 - LMT 1867 O 19 12:44:35
  // ...
  // -11 u B%sT 1983 O 30 2
  // -10 u AH%sT 1983 N 30
  // -10 u H%sT
  //
  // ...
  // R u 1945 o - S 30 2 0 S
  // R u 1967 2006 - O lastSu 2 0 S
  // R u 1967 1973 - Ap lastSu 2 1 D
  // R u 1974 o - Ja 6 2 1 D
  // R u 1975 o - F lastSu 2 1 D
  // R u 1976 1986 - Ap lastSu 2 1 D
  // R u 1987 2006 - Ap Su>=1 2 1 D
  // ...

  using namespace std::literals::chrono_literals;
  const std::chrono::time_zone* tz = std::chrono::locate_zone("America/Adak");

  assert_range( // 2 - EET 2012 N 10 2
      "[1983-10-30 12:00:00, 1983-11-30 10:00:00) -10:00:00 0min AHST",
      tz->get_info(to_sys_seconds(1983y, std::chrono::October, 30d, 12h)),              // -11 u B%sT 1983 O 30 2
      tz->get_info(to_sys_seconds(1983y, std::chrono::November, 30d, 9h, 59min, 59s))); // -10 u AH%sT 1983 N 30
}

static void test_america_auncion() {
  // R y 2013 ma - Mar Su>=22 0 0 -
  // Z America/Asuncion -3:50:40 - LMT 1890
  // -3:50:40 - AMT 1931 O 10
  // -4 - -04 1972 O
  // -3 - -03 1974 Ap
  // -4 y -04/-03
  //
  // R y 1975 1988 - O 1 0 1 -
  // R y 1975 1978 - Mar 1 0 0 -
  // R y 1979 1991 - Ap 1 0 0 -
  // ...

  using namespace std::literals::chrono_literals;
  const std::chrono::time_zone* tz = std::chrono::locate_zone("America/Asuncion");

  assert_range("[1974-04-01 03:00:00, 1975-10-01 04:00:00) -04:00:00 0min -04",
               tz->get_info(to_sys_seconds(1974y, std::chrono::April, 1d, 3h)),
               tz->get_info(to_sys_seconds(1975y, std::chrono::October, 1d, 3h, 59min, 59s)));

  assert_range("[1975-10-01 04:00:00, 1976-03-01 03:00:00) -03:00:00 60min -03",
               tz->get_info(to_sys_seconds(1975y, std::chrono::October, 1d, 4h)),
               tz->get_info(to_sys_seconds(1976y, std::chrono::March, 1d, 2h, 59min, 59s)));
}

static void test_america_ciudad_juarez() {
  // Z America/Ciudad_Juarez -7:5:56 - LMT 1922 Ja 1 7u
  // -7 - MST 1927 Jun 10 23
  // -6 - CST 1930 N 15
  // -7 m MST 1932 Ap
  // -6 - CST 1996
  // -6 m C%sT 1998
  // ...
  //
  // R m 1939 o - F 5 0 1 D
  // R m 1939 o - Jun 25 0 0 S
  // R m 1940 o - D 9 0 1 D
  // R m 1941 o - Ap 1 0 0 S
  // R m 1943 o - D 16 0 1 W
  // R m 1944 o - May 1 0 0 S
  // R m 1950 o - F 12 0 1 D
  // R m 1950 o - Jul 30 0 0 S
  // R m 1996 2000 - Ap Su>=1 2 1 D
  // R m 1996 2000 - O lastSu 2 0 S
  // ...

  using namespace std::literals::chrono_literals;
  const std::chrono::time_zone* tz = std::chrono::locate_zone("America/Ciudad_Juarez");

  // 1996 has a similar issue, instead of __time the __until end before
  // the first rule in 1939. Between the two usages of RULE Mexico
  // a different continuation RULE is active
  assert_range("[1996-04-07 08:00:00, 1996-10-27 07:00:00) -05:00:00 60min CDT",
               tz->get_info(to_sys_seconds(1996y, std::chrono::April, 7d, 8h)),
               tz->get_info(to_sys_seconds(1996y, std::chrono::October, 27d, 6h, 59min, 59s)));
}

static void test_america_argentina_buenos_aires() {
  // Z America/Argentina/Buenos_Aires -3:53:48 - LMT 1894 O 31
  // -4:16:48 - CMT 1920 May
  // -4 - -04 1930 D
  // -4 A -04/-03 1969 O 5
  // -3 A -03/-02 1999 O 3
  // -4 A -04/-03 2000 Mar 3
  // -3 A -03/-02
  //
  // ...
  // R A 1989 1992 - O Su>=15 0 1 -
  // R A 1999 o - O Su>=1 0 1 -
  // R A 2000 o - Mar 3 0 0 -
  // R A 2007 o - D 30 0 1 -
  // ...

  // The 1999 switch uses the same rule, but with a different stdoff.
  //   R A 1999 o - O Su>=1 0 1 -
  //     stdoff -3 -> 1999-10-03 03:00:00
  //     stdoff -4 -> 1999-10-03 04:00:00
  // This generates an invalid entry and this is evaluated as a transition.
  // Looking at the zdump like output in libc++ this generates jumps in
  // the UTC time

  using namespace std::literals::chrono_literals;
  const std::chrono::time_zone* tz = std::chrono::locate_zone("America/Argentina/Buenos_Aires");

  assert_range("[1999-10-03 03:00:00, 2000-03-03 03:00:00) -03:00:00 60min -03",
               tz->get_info(to_sys_seconds(1999y, std::chrono::October, 3d, 3h)),
               tz->get_info(to_sys_seconds(2000y, std::chrono::March, 3d, 2h, 59min, 59s)));
  assert_range("[2000-03-03 03:00:00, 2007-12-30 03:00:00) -03:00:00 0min -03",
               tz->get_info(to_sys_seconds(2000y, std::chrono::March, 3d, 3h)),
               tz->get_info(to_sys_seconds(2007y, std::chrono::December, 30d, 2h, 59min, 59s)));
}

static void test_america_argentina_la_rioja() {
  // Z America/Argentina/La_Rioja -4:27:24 - LMT 1894 O 31
  // ...
  // -4 A -04/-03 1969 O 5
  // -3 A -03/-02 1991 Mar
  // -4 - -04 1991 May 7
  // -3 A -03/-02 1999 O 3
  // ...
  //
  // ...
  // R A 1988 o - D 1 0 1 -
  // R A 1989 1993 - Mar Su>=1 0 0 -
  // R A 1989 1992 - O Su>=15 0 1 -
  // R A 1999 o - O Su>=1 0 1 -
  // ...

  using namespace std::literals::chrono_literals;
  const std::chrono::time_zone* tz = std::chrono::locate_zone("America/Argentina/La_Rioja");

  assert_range("[1990-10-21 03:00:00, 1991-03-01 02:00:00) -02:00:00 60min -02",
               tz->get_info(to_sys_seconds(1990y, std::chrono::October, 21d, 3h)),
               tz->get_info(to_sys_seconds(1991y, std::chrono::March, 1d, 1h, 59min, 59s)));
}

static void test_america_argentina_san_luis() {
  // Z America/Argentina/San_Luis -4:25:24 - LMT 1894 O 31
  // ...
  // -4 A -04/-03 1969 O 5
  // -3 A -03/-02 1990
  // -3 1 -02 1990 Mar 14
  // -4 - -04 1990 O 15
  // -4 1 -03 1991 Mar
  // -4 - -04 1991 Jun
  // -3 - -03 1999 O 3
  // -4 1 -03 2000 Mar 3
  // -4 - -04 2004 Jul 25
  // -3 A -03/-02 2008 Ja 21
  // -4 Sa -04/-03 2009 O 11
  // -3 - -03
  //
  // ...
  // R A 1988 o - D 1 0 1 -
  // R A 1989 1993 - Mar Su>=1 0 0 -
  // R A 1989 1992 - O Su>=15 0 1 -
  // R A 1999 o - O Su>=1 0 1 -
  // R A 2000 o - Mar 3 0 0 -
  // R A 2007 o - D 30 0 1 -
  // R A 2008 2009 - Mar Su>=15 0 0 -
  // R A 2008 o - O Su>=15 0 1 -
  //
  // R Sa 2008 2009 - Mar Su>=8 0 0 -
  // R Sa 2007 2008 - O Su>=8 0 1 -

  using namespace std::literals::chrono_literals;
  const std::chrono::time_zone* tz = std::chrono::locate_zone("America/Argentina/San_Luis");

  assert_range("[1989-10-15 03:00:00, 1990-03-14 02:00:00) -02:00:00 60min -02",
               tz->get_info( // -3 A -03/-02 1990 & R A 1989 1992 - O Su>=15 0 1 -
                   to_sys_seconds(1989y, std::chrono::October, 15d, 3h)),
               tz->get_info( // UNTIL -3 1 -02 1990 Mar 14
                   to_sys_seconds(1990y, std::chrono::March, 14d, 1h, 59min, 59s)));

  assert_range("[2008-01-21 02:00:00, 2008-03-09 03:00:00) -03:00:00 60min -03",
               tz->get_info(to_sys_seconds(2008y, std::chrono::January, 21d, 2h)),
               tz->get_info(to_sys_seconds(2008y, std::chrono::March, 9d, 2h, 59min, 59s)));
}

static void test_america_indiana_knox() {
  // Z America/Indiana/Knox -5:46:30 - LMT 1883 N 18 12:13:30
  // -6 u C%sT 1947
  // -6 St C%sT 1962 Ap 29 2
  // -5 - EST 1963 O 27 2
  // -6 u C%sT 1991 O 27 2
  // -5 - EST 2006 Ap 2 2
  // -6 u C%sT
  //
  // ...
  // R u 1976 1986 - Ap lastSu 2 1 D
  // R u 1987 2006 - Ap Su>=1 2 1 D
  // R u 2007 ma - Mar Su>=8 2 1 D
  // R u 2007 ma - N Su>=1 2 0 S

  using namespace std::literals::chrono_literals;
  const std::chrono::time_zone* tz = std::chrono::locate_zone("America/Indiana/Knox");

  // The continuations
  // -5 - EST
  // -6 u C%sT
  // have different offsets. The start time of the first active rule in
  // RULE u should use the offset at the end of -5 - EST.
  assert_range("[2006-04-02 07:00:00, 2006-10-29 07:00:00) -05:00:00 60min CDT",
               tz->get_info(to_sys_seconds(2006y, std::chrono::April, 2d, 7h)),
               tz->get_info(to_sys_seconds(2006y, std::chrono::October, 29d, 6h, 59min, 59s)));
}

static void test_america_punta_arenas() {
  // Z America/Punta_Arenas -4:43:40 - LMT 1890
  // ...
  // -4 - -04 1919 Jul
  // -4:42:45 - SMT 1927 S
  // -5 x -05/-04 1932 S
  // ...
  //
  // R x 1927 1931 - S 1 0 1 -
  // R x 1928 1932 - Ap 1 0 0 -
  // ...

  using namespace std::literals::chrono_literals;
  const std::chrono::time_zone* tz = std::chrono::locate_zone("America/Punta_Arenas");

  assert_equal(
      std::chrono::sys_info(
          to_sys_seconds(1927y, std::chrono::September, 1d, 4h, 42min, 45s),
          to_sys_seconds(1928y, std::chrono::April, 1d, 4h),
          -4h,
          60min,
          "-04"),
      tz->get_info(to_sys_seconds(1927y, std::chrono::September, 1d, 4h, 42min, 45s)));

  assert_equal(
      std::chrono::sys_info(
          to_sys_seconds(1927y, std::chrono::September, 1d, 4h, 42min, 45s),
          to_sys_seconds(1928y, std::chrono::April, 1d, 4h),
          -4h,
          60min,
          "-04"),
      tz->get_info(to_sys_seconds(1928y, std::chrono::April, 1d, 3h, 59min, 59s)));
}

static void test_europ_ljubljana() {
  // Z Europe/Ljubljana 0:58:4 - LMT 1884
  // 1 - CET 1941 Ap 18 23
  // 1 c CE%sT 1945 May 8 2s
  // 1 1 CEST 1945 S 16 2s
  // 1 - CET 1982 N 27
  // 1 E CE%sT
  //
  // ...
  // R c 1943 o - O 4 2s 0 -
  // R c 1944 1945 - Ap M>=1 2s 1 S
  // R c 1944 o - O 2 2s 0 -
  // R c 1945 o - S 16 2s 0 -
  // R c 1977 1980 - Ap Su>=1 2s 1 S
  // ...

  using namespace std::literals::chrono_literals;
  const std::chrono::time_zone* tz = std::chrono::locate_zone("Europe/Ljubljana");

  assert_equal(
      std::chrono::sys_info(
          to_sys_seconds(1945y, std::chrono::April, 2d, 1h),
          to_sys_seconds(1945y, std::chrono::September, 16d, 1h),
          2h,
          60min,
          "CEST"),
      tz->get_info(to_sys_seconds(1945y, std::chrono::April, 2d, 1h)));

  assert_equal(
      std::chrono::sys_info(
          to_sys_seconds(1945y, std::chrono::April, 2d, 1h),
          to_sys_seconds(1945y, std::chrono::September, 16d, 1h),
          2h,
          60min,
          "CEST"),
      tz->get_info(to_sys_seconds(1945y, std::chrono::September, 16d, 0h, 59min, 59s)));
}

int main(int, const char**) {
  // Basic tests
  test_gmt();
  test_durations();
  test_antarctica_syowa();
  test_asia_hong_kong();
  test_europe_berlin();

  test_america_st_johns();

  // Small tests for not-yet tested conditions
  test_get_at_standard_time_universal();
  test_get_at_standard_time_standard();
  test_get_at_save_universal();
  test_get_at_rule_standard();
  test_get_at_rule_universal();

  test_format_with_alternatives_west();
  test_format_with_alternatives_east();

  // Tests based on bugs found
  test_africa_algiers();
  test_africa_casablanca();
  test_africa_ceuta();
  test_africa_freetown();
  test_africa_windhoek();
  test_america_adak();
  test_america_argentina_buenos_aires();
  test_america_argentina_la_rioja();
  test_america_argentina_san_luis();
  test_america_auncion();
  test_america_ciudad_juarez();
  test_america_indiana_knox();

  // Reverse search bugs
  test_america_punta_arenas();
  test_europ_ljubljana();

  return 0;
}
