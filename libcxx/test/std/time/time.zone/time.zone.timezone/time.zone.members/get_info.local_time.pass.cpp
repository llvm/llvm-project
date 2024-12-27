//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: no-filesystem, no-localization, no-tzdb
// REQUIRES: long_tests

// XFAIL: libcpp-has-no-experimental-tzdb
// XFAIL: availability-tzdb-missing

// TODO TZDB Investigate why this fails.
// UNSUPPORTED: target={{.*}}

// <chrono>

// class time_zone;

// template <class _Duration>
//   local_info get_info(const local_time<_Duration>& time) const;

// This test uses the system provided database. This makes the test portable,
// but may cause failures when the database information changes. Historic data
// may change if new facts are uncovered, future data may change when regions
// change their time zone or daylight saving time. Most tests will not look in
// the future to attempt to avoid issues. All tests list the data on which they
// are based, this makes debugging easier upon failure; including to see whether
// the provided data has not been changed.
//
// The first part of the test is manually crafted, the second part compares the
// transitions for all time zones in the database.

#include <algorithm>
#include <cassert>
#include <chrono>
#include <format>

#include "test_macros.h"
#include "assert_macros.h"
#include "concat_macros.h"

// The year range to validate. The dates used in practice are expected to be
// inside the tested range.
constexpr std::chrono::year first{1800};
constexpr std::chrono::year last{2100};

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

[[nodiscard]] static std::chrono::local_seconds to_local_seconds(
    std::chrono::year year,
    std::chrono::month month,
    std::chrono::day day,
    std::chrono::hours h   = std::chrono::hours(0),
    std::chrono::minutes m = std::chrono::minutes{0},
    std::chrono::seconds s = std::chrono::seconds{0}) {
  std::chrono::year_month_day result{year, month, day};

  return std::chrono::time_point_cast<std::chrono::seconds>(static_cast<std::chrono::local_days>(result)) + h + m + s;
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

static void assert_equal(const std::chrono::local_info& lhs, const std::chrono::local_info& rhs) {
  TEST_REQUIRE(
      lhs.result == rhs.result,
      TEST_WRITE_CONCATENATED("\nResult:\nExpected output ", lhs.result, "\nActual output   ", rhs.result, '\n'));

  assert_equal(lhs.first, rhs.first);
  assert_equal(lhs.second, rhs.second);
}

/***** ***** TESTS ***** *****/

static void test_gmt() {
  // Simple zone always valid, no rule entries, lookup using a link.
  // L Etc/GMT GMT
  // Z Etc/GMT 0 - GMT

  using namespace std::literals::chrono_literals;
  const std::chrono::time_zone* tz = std::chrono::locate_zone("GMT");

  assert_equal(
      std::chrono::local_info(
          std::chrono::local_info::unique,
          std::chrono::sys_info(std::chrono::sys_seconds::min(), std::chrono::sys_seconds::max(), 0s, 0min, "GMT"),
          std::chrono::sys_info(std::chrono::sys_seconds(0s), std::chrono::sys_seconds(0s), 0s, 0min, "")),
      tz->get_info(std::chrono::local_seconds::min()));
}

static void test_local_time_out_of_range() {
  // Fixed positive offset
  // Etc/GMT-1 1 - +01

  using namespace std::literals::chrono_literals;
  { // lower bound
    const std::chrono::time_zone* tz = std::chrono::locate_zone("Etc/GMT-1");

    assert_equal(
        std::chrono::local_info(
            -1,
            std::chrono::sys_info(std::chrono::sys_seconds::min(), std::chrono::sys_seconds::max(), 1h, 0min, "+01"),
            std::chrono::sys_info(std::chrono::sys_seconds(0s), std::chrono::sys_seconds(0s), 0s, 0min, "")),
        tz->get_info(std::chrono::local_seconds::min()));

    assert_equal(
        std::chrono::local_info(
            -1,
            std::chrono::sys_info(std::chrono::sys_seconds::min(), std::chrono::sys_seconds::max(), 1h, 0min, "+01"),
            std::chrono::sys_info(std::chrono::sys_seconds(0s), std::chrono::sys_seconds(0s), 0s, 0min, "")),
        tz->get_info(std::chrono::local_seconds::min() + 59min + 59s));

    assert_equal(
        std::chrono::local_info(
            std::chrono::local_info::unique,
            std::chrono::sys_info(std::chrono::sys_seconds::min(), std::chrono::sys_seconds::max(), 1h, 0min, "+01"),
            std::chrono::sys_info(std::chrono::sys_seconds(0s), std::chrono::sys_seconds(0s), 0s, 0min, "")),
        tz->get_info(std::chrono::local_seconds::min() + 1h));
  }

  { // upper bound
    const std::chrono::time_zone* tz = std::chrono::locate_zone("Etc/GMT+1");

    assert_equal(
        std::chrono::local_info(
            -2,
            std::chrono::sys_info(std::chrono::sys_seconds::min(), std::chrono::sys_seconds::max(), -1h, 0min, "-01"),
            std::chrono::sys_info(std::chrono::sys_seconds(0s), std::chrono::sys_seconds(0s), 0s, 0min, "")),
        tz->get_info(std::chrono::local_seconds::max() - 1s));

    assert_equal(
        std::chrono::local_info(
            std::chrono::local_info::unique,
            std::chrono::sys_info(std::chrono::sys_seconds::min(), std::chrono::sys_seconds::max(), -1h, 0min, "-01"),
            std::chrono::sys_info(std::chrono::sys_seconds(0s), std::chrono::sys_seconds(0s), 0s, 0min, "")),
        tz->get_info(std::chrono::local_seconds::max() - 1h - 1s));
  }
}

static void test_indian_kerguelen() {
  // One change, no rules, no dst changes.

  // Z Indian/Kerguelen 0 - -00 1950
  // 5 - +05

  using namespace std::literals::chrono_literals;
  const std::chrono::time_zone* tz = std::chrono::locate_zone("Indian/Kerguelen");

  assert_equal(
      std::chrono::local_info(
          std::chrono::local_info::unique,
          std::chrono::sys_info(
              std::chrono::sys_seconds::min(), to_sys_seconds(1950y, std::chrono::January, 1d), 0s, 0min, "-00"),
          std::chrono::sys_info(std::chrono::sys_seconds(0s), std::chrono::sys_seconds(0s), 0s, 0min, "")),
      tz->get_info(std::chrono::local_seconds::min()));

  assert_equal(
      std::chrono::local_info(
          std::chrono::local_info::nonexistent,
          std::chrono::sys_info(
              std::chrono::sys_seconds::min(), to_sys_seconds(1950y, std::chrono::January, 1d), 0s, 0min, "-00"),
          std::chrono::sys_info(
              to_sys_seconds(1950y, std::chrono::January, 1d), std::chrono::sys_seconds::max(), 5h, 0min, "+05")),
      tz->get_info(to_local_seconds(1950y, std::chrono::January, 1d)));

  assert_equal(
      std::chrono::local_info(
          std::chrono::local_info::unique,
          std::chrono::sys_info(
              to_sys_seconds(1950y, std::chrono::January, 1d), std::chrono::sys_seconds::max(), 5h, 0min, "+05"),
          std::chrono::sys_info(std::chrono::sys_seconds(0s), std::chrono::sys_seconds(0s), 0s, 0min, "")),
      tz->get_info(to_local_seconds(1950y, std::chrono::January, 1d, 5h)));

  assert_equal(
      std::chrono::local_info(
          std::chrono::local_info::unique,
          std::chrono::sys_info(
              to_sys_seconds(1950y, std::chrono::January, 1d), std::chrono::sys_seconds::max(), 5h, 0min, "+05"),
          std::chrono::sys_info(std::chrono::sys_seconds(0s), std::chrono::sys_seconds(0s), 0s, 0min, "")),
      tz->get_info(std::chrono::local_seconds::max() - 1s));
}

static void test_antarctica_rothera() {
  // One change, no rules, no dst changes

  // Z Antarctica/Rothera 0 - -00 1976 D
  // -3 - -03

  using namespace std::literals::chrono_literals;
  const std::chrono::time_zone* tz = std::chrono::locate_zone("Antarctica/Rothera");

  assert_equal(
      std::chrono::local_info(
          std::chrono::local_info::unique,
          std::chrono::sys_info(
              std::chrono::sys_seconds::min(), to_sys_seconds(1976y, std::chrono::December, 1d), 0s, 0min, "-00"),
          std::chrono::sys_info(std::chrono::sys_seconds(0s), std::chrono::sys_seconds(0s), 0s, 0min, "")),
      tz->get_info(std::chrono::local_seconds::min()));

  assert_equal(
      std::chrono::local_info(
          std::chrono::local_info::unique,
          std::chrono::sys_info(
              std::chrono::sys_seconds::min(), to_sys_seconds(1976y, std::chrono::December, 1d), 0s, 0min, "-00"),
          std::chrono::sys_info(std::chrono::sys_seconds(0s), std::chrono::sys_seconds(0s), 0s, 0min, "")),
      tz->get_info(to_local_seconds(1976y, std::chrono::November, 30d, 20h, 59min, 59s)));

  assert_equal(
      std::chrono::local_info(
          std::chrono::local_info::ambiguous,
          std::chrono::sys_info(
              std::chrono::sys_seconds::min(), to_sys_seconds(1976y, std::chrono::December, 1d), 0s, 0min, "-00"),
          std::chrono::sys_info(
              to_sys_seconds(1976y, std::chrono::December, 1d), std::chrono::sys_seconds::max(), -3h, 0min, "-03")),
      tz->get_info(to_local_seconds(1976y, std::chrono::November, 30d, 21h)));

  assert_equal(
      std::chrono::local_info(
          std::chrono::local_info::ambiguous,
          std::chrono::sys_info(
              std::chrono::sys_seconds::min(), to_sys_seconds(1976y, std::chrono::December, 1d), 0s, 0min, "-00"),
          std::chrono::sys_info(
              to_sys_seconds(1976y, std::chrono::December, 1d), std::chrono::sys_seconds::max(), -3h, 0min, "-03")),
      tz->get_info(to_local_seconds(1976y, std::chrono::November, 30d, 23h, 59min, 59s)));

  assert_equal(
      std::chrono::local_info(
          std::chrono::local_info::unique,
          std::chrono::sys_info(
              to_sys_seconds(1976y, std::chrono::December, 1d), std::chrono::sys_seconds::max(), -3h, 0min, "-03"),
          std::chrono::sys_info(std::chrono::sys_seconds(0s), std::chrono::sys_seconds(0s), 0s, 0min, "")),
      tz->get_info(to_local_seconds(1976y, std::chrono::December, 1d)));

  assert_equal(
      std::chrono::local_info(
          std::chrono::local_info::unique,
          std::chrono::sys_info(
              to_sys_seconds(1976y, std::chrono::December, 1d), std::chrono::sys_seconds::max(), -3h, 0min, "-03"),
          std::chrono::sys_info(std::chrono::sys_seconds(0s), std::chrono::sys_seconds(0s), 0s, 0min, "")),
      tz->get_info(std::chrono::local_seconds::max() - 3h - 1s));

  assert_equal(
      std::chrono::local_info(
          -2,
          std::chrono::sys_info(
              to_sys_seconds(1976y, std::chrono::December, 1d), std::chrono::sys_seconds::max(), -3h, 0min, "-03"),
          std::chrono::sys_info(std::chrono::sys_seconds(0s), std::chrono::sys_seconds(0s), 0s, 0min, "")),
      tz->get_info(std::chrono::local_seconds::max() - 1s));
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
      std::chrono::local_info(
          -1,
          std::chrono::sys_info(
              std::chrono::sys_seconds::min(),
              to_sys_seconds(1904y, std::chrono::October, 29d, 17h),
              7h + 36min + 42s,
              0min,
              "LMT"),
          std::chrono::sys_info(std::chrono::sys_seconds(0s), std::chrono::sys_seconds(0s), 0s, 0min, "")),
      tz->get_info(std::chrono::local_seconds::min()));

  assert_equal(
      std::chrono::local_info(
          -1,
          std::chrono::sys_info(
              std::chrono::sys_seconds::min(),
              to_sys_seconds(1904y, std::chrono::October, 29d, 17h),
              7h + 36min + 42s,
              0min,
              "LMT"),
          std::chrono::sys_info(std::chrono::sys_seconds(0s), std::chrono::sys_seconds(0s), 0s, 0min, "")),
      tz->get_info(std::chrono::local_seconds::min() + 7h + 36min + 41s));

  assert_equal(
      std::chrono::local_info(
          std::chrono::local_info::unique,
          std::chrono::sys_info(
              std::chrono::sys_seconds::min(),
              to_sys_seconds(1904y, std::chrono::October, 29d, 17h),
              7h + 36min + 42s,
              0min,
              "LMT"),
          std::chrono::sys_info(std::chrono::sys_seconds(0s), std::chrono::sys_seconds(0s), 0s, 0min, "")),
      tz->get_info(std::chrono::local_seconds::min() + 7h + 36min + 42s));

  assert_equal(
      std::chrono::local_info(
          std::chrono::local_info::unique,
          std::chrono::sys_info(
              std::chrono::sys_seconds::min(),
              to_sys_seconds(1904y, std::chrono::October, 29d, 17h),
              7h + 36min + 42s,
              0min,
              "LMT"),
          std::chrono::sys_info(std::chrono::sys_seconds(0s), std::chrono::sys_seconds(0s), 0s, 0min, "")),
      tz->get_info(to_local_seconds(1904y, std::chrono::October, 30d, 0h, 36min, 41s)));

  assert_equal(
      std::chrono::local_info(
          std::chrono::local_info::nonexistent,
          std::chrono::sys_info(
              std::chrono::sys_seconds::min(),
              to_sys_seconds(1904y, std::chrono::October, 29d, 17h),
              7h + 36min + 42s,
              0min,
              "LMT"),
          std::chrono::sys_info(
              to_sys_seconds(1904y, std::chrono::October, 29d, 17h),
              to_sys_seconds(1941y, std::chrono::June, 14d, 19h),
              8h,
              0min,
              "HKT")),
      tz->get_info(to_local_seconds(1904y, std::chrono::October, 30d, 0h, 36min, 42s)));

  assert_equal(
      std::chrono::local_info(
          std::chrono::local_info::nonexistent,
          std::chrono::sys_info(
              std::chrono::sys_seconds::min(),
              to_sys_seconds(1904y, std::chrono::October, 29d, 17h),
              7h + 36min + 42s,
              0min,
              "LMT"),
          std::chrono::sys_info(
              to_sys_seconds(1904y, std::chrono::October, 29d, 17h),
              to_sys_seconds(1941y, std::chrono::June, 14d, 19h),
              8h,
              0min,
              "HKT")),
      tz->get_info(to_local_seconds(1904y, std::chrono::October, 30d, 0h, 59min, 59s)));

  assert_equal(
      std::chrono::local_info(
          std::chrono::local_info::unique,
          std::chrono::sys_info(
              to_sys_seconds(1904y, std::chrono::October, 29d, 17h),
              to_sys_seconds(1941y, std::chrono::June, 14d, 19h),
              8h,
              0min,
              "HKT"),
          std::chrono::sys_info(std::chrono::sys_seconds(0s), std::chrono::sys_seconds(0s), 0s, 0min, "")),
      tz->get_info(to_local_seconds(1904y, std::chrono::October, 30d, 1h)));

  assert_equal(
      std::chrono::local_info(
          std::chrono::local_info::unique,
          std::chrono::sys_info(
              to_sys_seconds(1904y, std::chrono::October, 29d, 17h),
              to_sys_seconds(1941y, std::chrono::June, 14d, 19h),
              8h,
              0min,
              "HKT"),
          std::chrono::sys_info(std::chrono::sys_seconds(0s), std::chrono::sys_seconds(0s), 0s, 0min, "")),
      tz->get_info(to_local_seconds(1941y, std::chrono::June, 15d, 2h, 59min, 59s)));

  assert_equal(
      std::chrono::local_info(
          std::chrono::local_info::nonexistent,
          std::chrono::sys_info(
              to_sys_seconds(1904y, std::chrono::October, 29d, 17h),
              to_sys_seconds(1941y, std::chrono::June, 14d, 19h),
              8h,
              0min,
              "HKT"),
          std::chrono::sys_info(
              to_sys_seconds(1941y, std::chrono::June, 14d, 19h),
              to_sys_seconds(1941y, std::chrono::September, 30d, 19h),
              9h,
              60min,
              "HKST")),
      tz->get_info(to_local_seconds(1941y, std::chrono::June, 15d, 3h)));

  assert_equal(
      std::chrono::local_info(
          std::chrono::local_info::nonexistent,
          std::chrono::sys_info(
              to_sys_seconds(1904y, std::chrono::October, 29d, 17h),
              to_sys_seconds(1941y, std::chrono::June, 14d, 19h),
              8h,
              0min,
              "HKT"),
          std::chrono::sys_info(
              to_sys_seconds(1941y, std::chrono::June, 14d, 19h),
              to_sys_seconds(1941y, std::chrono::September, 30d, 19h),
              9h,
              60min,
              "HKST")),
      tz->get_info(to_local_seconds(1941y, std::chrono::June, 15d, 3h, 59min, 59s)));

  assert_equal(
      std::chrono::local_info(
          std::chrono::local_info::unique,
          std::chrono::sys_info(
              to_sys_seconds(1941y, std::chrono::June, 14d, 19h),
              to_sys_seconds(1941y, std::chrono::September, 30d, 19h),
              9h,
              60min,
              "HKST"),
          std::chrono::sys_info(std::chrono::sys_seconds(0s), std::chrono::sys_seconds(0s), 0s, 0min, "")),
      tz->get_info(to_local_seconds(1941y, std::chrono::June, 15d, 4h)));

  assert_equal(
      std::chrono::local_info(
          std::chrono::local_info::unique,
          std::chrono::sys_info(
              to_sys_seconds(1941y, std::chrono::June, 14d, 19h),
              to_sys_seconds(1941y, std::chrono::September, 30d, 19h),
              9h,
              60min,
              "HKST"),
          std::chrono::sys_info(std::chrono::sys_seconds(0s), std::chrono::sys_seconds(0s), 0s, 0min, "")),
      tz->get_info(to_local_seconds(1941y, std::chrono::October, 1d, 3h, 29min, 29s)));

  assert_equal(
      std::chrono::local_info(
          std::chrono::local_info::ambiguous,
          std::chrono::sys_info(
              to_sys_seconds(1941y, std::chrono::June, 14d, 19h),
              to_sys_seconds(1941y, std::chrono::September, 30d, 19h),
              9h,
              60min,
              "HKST"),
          std::chrono::sys_info(
              to_sys_seconds(1941y, std::chrono::September, 30d, 19h),
              to_sys_seconds(1941y, std::chrono::December, 24d, 15h, 30min),
              8h + 30min,
              30min,
              "HKWT")),
      tz->get_info(to_local_seconds(1941y, std::chrono::October, 1d, 3h, 30min)));

  assert_equal(
      std::chrono::local_info(
          std::chrono::local_info::ambiguous,
          std::chrono::sys_info(
              to_sys_seconds(1941y, std::chrono::June, 14d, 19h),
              to_sys_seconds(1941y, std::chrono::September, 30d, 19h),
              9h,
              60min,
              "HKST"),
          std::chrono::sys_info(
              to_sys_seconds(1941y, std::chrono::September, 30d, 19h),
              to_sys_seconds(1941y, std::chrono::December, 24d, 15h, 30min),
              8h + 30min,
              30min,
              "HKWT")),
      tz->get_info(to_local_seconds(1941y, std::chrono::October, 1d, 3h, 59min, 59s)));

  assert_equal(
      std::chrono::local_info(
          std::chrono::local_info::unique,
          std::chrono::sys_info(
              to_sys_seconds(1941y, std::chrono::September, 30d, 19h),
              to_sys_seconds(1941y, std::chrono::December, 24d, 15h, 30min),
              8h + 30min,
              30min,
              "HKWT"),
          std::chrono::sys_info(std::chrono::sys_seconds(0s), std::chrono::sys_seconds(0s), 0s, 0min, "")),
      tz->get_info(to_local_seconds(1941y, std::chrono::October, 1d, 4h)));
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
      std::chrono::local_info(
          -1,
          std::chrono::sys_info(
              std::chrono::sys_seconds::min(),
              to_sys_seconds(1893y, std::chrono::March, 31d, 23h, 6min, 32s),
              53min + 28s,
              0min,
              "LMT"),
          std::chrono::sys_info(std::chrono::sys_seconds(0s), std::chrono::sys_seconds(0s), 0s, 0min, "")),
      tz->get_info(std::chrono::local_seconds::min()));

  assert_equal(
      std::chrono::local_info(
          -1,
          std::chrono::sys_info(
              std::chrono::sys_seconds::min(),
              to_sys_seconds(1893y, std::chrono::March, 31d, 23h, 6min, 32s),
              53min + 28s,
              0min,
              "LMT"),
          std::chrono::sys_info(std::chrono::sys_seconds(0s), std::chrono::sys_seconds(0s), 0s, 0min, "")),
      tz->get_info(std::chrono::local_seconds::min() + 53min + 27s));

  assert_equal(
      std::chrono::local_info(
          std::chrono::local_info::unique,
          std::chrono::sys_info(
              std::chrono::sys_seconds::min(),
              to_sys_seconds(1893y, std::chrono::March, 31d, 23h, 6min, 32s),
              53min + 28s,
              0min,
              "LMT"),
          std::chrono::sys_info(std::chrono::sys_seconds(0s), std::chrono::sys_seconds(0s), 0s, 0min, "")),
      tz->get_info(std::chrono::local_seconds::min() + 53min + 28s));

  assert_equal(
      std::chrono::local_info(
          std::chrono::local_info::unique,
          std::chrono::sys_info(
              std::chrono::sys_seconds::min(),
              to_sys_seconds(1893y, std::chrono::March, 31d, 23h, 6min, 32s),
              53min + 28s,
              0min,
              "LMT"),
          std::chrono::sys_info(std::chrono::sys_seconds(0s), std::chrono::sys_seconds(0s), 0s, 0min, "")),
      tz->get_info(to_local_seconds(1893y, std::chrono::March, 31d, 23h, 59min, 59s)));

  assert_equal(
      std::chrono::local_info(
          std::chrono::local_info::unique,
          std::chrono::sys_info(
              to_sys_seconds(1946y, std::chrono::October, 7d, 1h),
              to_sys_seconds(1947y, std::chrono::April, 6d, 2h),
              1h,
              0min,
              "CET"),
          std::chrono::sys_info(std::chrono::sys_seconds(0s), std::chrono::sys_seconds(0s), 0s, 0min, "")),
      tz->get_info(to_local_seconds(1947y, std::chrono::April, 6d, 2h, 59min, 59s)));

  assert_equal(
      std::chrono::local_info(
          std::chrono::local_info::nonexistent,
          std::chrono::sys_info(
              to_sys_seconds(1946y, std::chrono::October, 7d, 1h),
              to_sys_seconds(1947y, std::chrono::April, 6d, 2h),
              1h,
              0min,
              "CET"),
          std::chrono::sys_info(
              to_sys_seconds(1947y, std::chrono::April, 6d, 2h),
              to_sys_seconds(1947y, std::chrono::May, 11d, 1h),
              2h,
              60min,
              "CEST")),
      tz->get_info(to_local_seconds(1947y, std::chrono::April, 6d, 3h)));

  assert_equal(
      std::chrono::local_info(
          std::chrono::local_info::nonexistent,
          std::chrono::sys_info(
              to_sys_seconds(1946y, std::chrono::October, 7d, 1h),
              to_sys_seconds(1947y, std::chrono::April, 6d, 2h),
              1h,
              0min,
              "CET"),
          std::chrono::sys_info(
              to_sys_seconds(1947y, std::chrono::April, 6d, 2h),
              to_sys_seconds(1947y, std::chrono::May, 11d, 1h),
              2h,
              60min,
              "CEST")),
      tz->get_info(to_local_seconds(1947y, std::chrono::April, 6d, 3h, 59min, 59s)));

  assert_equal(
      std::chrono::local_info(
          std::chrono::local_info::unique,
          std::chrono::sys_info(
              to_sys_seconds(1947y, std::chrono::April, 6d, 2h),
              to_sys_seconds(1947y, std::chrono::May, 11d, 1h),
              2h,
              60min,
              "CEST"),
          std::chrono::sys_info(std::chrono::sys_seconds(0s), std::chrono::sys_seconds(0s), 0s, 0min, "")),
      tz->get_info(to_local_seconds(1947y, std::chrono::April, 6d, 4h)));

  assert_equal(
      std::chrono::local_info(
          std::chrono::local_info::unique,
          std::chrono::sys_info(
              to_sys_seconds(1947y, std::chrono::April, 6d, 2h),
              to_sys_seconds(1947y, std::chrono::May, 11d, 1h),
              2h,
              60min,
              "CEST"),
          std::chrono::sys_info(std::chrono::sys_seconds(0s), std::chrono::sys_seconds(0s), 0s, 0min, "")),
      tz->get_info(to_local_seconds(1947y, std::chrono::May, 11d, 2h, 59min, 59s)));

  assert_equal(
      std::chrono::local_info(
          std::chrono::local_info::nonexistent,
          std::chrono::sys_info(
              to_sys_seconds(1947y, std::chrono::April, 6d, 2h),
              to_sys_seconds(1947y, std::chrono::May, 11d, 1h),
              2h,
              60min,
              "CEST"),
          std::chrono::sys_info(
              to_sys_seconds(1947y, std::chrono::May, 11d, 1h),
              to_sys_seconds(1947y, std::chrono::June, 29d),
              3h,
              120min,
              "CEMT")),
      tz->get_info(to_local_seconds(1947y, std::chrono::May, 11d, 3h)));

  assert_equal(
      std::chrono::local_info(
          std::chrono::local_info::nonexistent,
          std::chrono::sys_info(
              to_sys_seconds(1947y, std::chrono::April, 6d, 2h),
              to_sys_seconds(1947y, std::chrono::May, 11d, 1h),
              2h,
              60min,
              "CEST"),
          std::chrono::sys_info(
              to_sys_seconds(1947y, std::chrono::May, 11d, 1h),
              to_sys_seconds(1947y, std::chrono::June, 29d),
              3h,
              120min,
              "CEMT")),
      tz->get_info(to_local_seconds(1947y, std::chrono::May, 11d, 3h, 59min, 59s)));

  assert_equal(
      std::chrono::local_info(
          std::chrono::local_info::unique,
          std::chrono::sys_info(
              to_sys_seconds(1947y, std::chrono::May, 11d, 1h),
              to_sys_seconds(1947y, std::chrono::June, 29d),
              3h,
              120min,
              "CEMT"),
          std::chrono::sys_info(std::chrono::sys_seconds(0s), std::chrono::sys_seconds(0s), 0s, 0min, "")),
      tz->get_info(to_local_seconds(1947y, std::chrono::May, 11d, 4h)));

  assert_equal(
      std::chrono::local_info(
          std::chrono::local_info::unique,
          std::chrono::sys_info(
              to_sys_seconds(1947y, std::chrono::May, 11d, 1h),
              to_sys_seconds(1947y, std::chrono::June, 29d),
              3h,
              120min,
              "CEMT"),
          std::chrono::sys_info(std::chrono::sys_seconds(0s), std::chrono::sys_seconds(0s), 0s, 0min, "")),
      tz->get_info(to_local_seconds(1947y, std::chrono::June, 29d, 1h, 59min, 59s)));

  assert_equal(
      std::chrono::local_info(
          std::chrono::local_info::ambiguous,
          std::chrono::sys_info(
              to_sys_seconds(1947y, std::chrono::May, 11d, 1h),
              to_sys_seconds(1947y, std::chrono::June, 29d),
              3h,
              120min,
              "CEMT"),
          std::chrono::sys_info(
              to_sys_seconds(1947y, std::chrono::June, 29d),
              to_sys_seconds(1947y, std::chrono::October, 5d, 1h),
              2h,
              60min,
              "CEST")),
      tz->get_info(to_local_seconds(1947y, std::chrono::June, 29d, 2h)));

  assert_equal(
      std::chrono::local_info(
          std::chrono::local_info::ambiguous,
          std::chrono::sys_info(
              to_sys_seconds(1947y, std::chrono::May, 11d, 1h),
              to_sys_seconds(1947y, std::chrono::June, 29d),
              3h,
              120min,
              "CEMT"),
          std::chrono::sys_info(
              to_sys_seconds(1947y, std::chrono::June, 29d),
              to_sys_seconds(1947y, std::chrono::October, 5d, 1h),
              2h,
              60min,
              "CEST")),
      tz->get_info(to_local_seconds(1947y, std::chrono::June, 29d, 2h, 59min, 59s)));

  assert_equal(
      std::chrono::local_info(
          std::chrono::local_info::unique,
          std::chrono::sys_info(
              to_sys_seconds(1947y, std::chrono::June, 29d),
              to_sys_seconds(1947y, std::chrono::October, 5d, 1h),
              2h,
              60min,
              "CEST"),
          std::chrono::sys_info(std::chrono::sys_seconds(0s), std::chrono::sys_seconds(0s), 0s, 0min, "")),
      tz->get_info(to_local_seconds(1947y, std::chrono::June, 29d, 3h)));

  assert_equal(
      std::chrono::local_info(
          std::chrono::local_info::unique,
          std::chrono::sys_info(
              to_sys_seconds(1947y, std::chrono::June, 29d),
              to_sys_seconds(1947y, std::chrono::October, 5d, 1h),
              2h,
              60min,
              "CEST"),
          std::chrono::sys_info(std::chrono::sys_seconds(0s), std::chrono::sys_seconds(0s), 0s, 0min, "")),
      tz->get_info(to_local_seconds(1947y, std::chrono::October, 5d, 1h, 59min, 59s)));

  assert_equal(
      std::chrono::local_info(
          std::chrono::local_info::ambiguous,
          std::chrono::sys_info(
              to_sys_seconds(1947y, std::chrono::June, 29d),
              to_sys_seconds(1947y, std::chrono::October, 5d, 1h),
              2h,
              60min,
              "CEST"),
          std::chrono::sys_info(
              to_sys_seconds(1947y, std::chrono::October, 5d, 1h),
              to_sys_seconds(1948y, std::chrono::April, 18d, 1h),
              1h,
              0min,
              "CET")),
      tz->get_info(to_local_seconds(1947y, std::chrono::October, 5d, 2h)));

  assert_equal(
      std::chrono::local_info(
          std::chrono::local_info::ambiguous,
          std::chrono::sys_info(
              to_sys_seconds(1947y, std::chrono::June, 29d),
              to_sys_seconds(1947y, std::chrono::October, 5d, 1h),
              2h,
              60min,
              "CEST"),
          std::chrono::sys_info(
              to_sys_seconds(1947y, std::chrono::October, 5d, 1h),
              to_sys_seconds(1948y, std::chrono::April, 18d, 1h),
              1h,
              0min,
              "CET")),
      tz->get_info(to_local_seconds(1947y, std::chrono::October, 5d, 2h, 59min, 59s)));

  assert_equal(
      std::chrono::local_info(
          std::chrono::local_info::unique,
          std::chrono::sys_info(
              to_sys_seconds(1947y, std::chrono::October, 5d, 1h),
              to_sys_seconds(1948y, std::chrono::April, 18d, 1h),
              1h,
              0min,
              "CET"),
          std::chrono::sys_info(std::chrono::sys_seconds(0s), std::chrono::sys_seconds(0s), 0s, 0min, "")),
      tz->get_info(to_local_seconds(1947y, std::chrono::October, 5d, 3h)));
}

static void test_europe_dublin() {
  // Z Europe/Dublin -0:25:21 - LMT 1880 Au 2
  // -0:25:21 - DMT 1916 May 21 2s
  // -0:25:21 1 IST 1916 O 1 2s
  // 0 G %s 1921 D 6
  // ...
  //
  // R G 1916 o - May 21 2s 1 BST
  // R G 1916 o - O 1 2s 0 GMT
  // R G 1917 o - Ap 8 2s 1 BST
  // ...

  using namespace std::literals::chrono_literals;
  const std::chrono::time_zone* tz = std::chrono::locate_zone("Europe/Dublin");

  assert_equal(
      std::chrono::local_info(
          std::chrono::local_info::unique,
          std::chrono::sys_info(
              std::chrono::sys_seconds::min(),
              to_sys_seconds(1880y, std::chrono::August, 2d, 0h, 25min, 21s),
              -(25min + 21s),
              0min,
              "LMT"),
          std::chrono::sys_info(std::chrono::sys_seconds(0s), std::chrono::sys_seconds(0s), 0s, 0min, "")),
      tz->get_info(std::chrono::local_seconds::min()));

  assert_equal(
      std::chrono::local_info(
          std::chrono::local_info::unique,
          std::chrono::sys_info(
              std::chrono::sys_seconds::min(),
              to_sys_seconds(1880y, std::chrono::August, 2d, 0h, 25min, 21s),
              -(25min + 21s),
              0min,
              "LMT"),
          std::chrono::sys_info(std::chrono::sys_seconds(0s), std::chrono::sys_seconds(0s), 0s, 0min, "")),
      tz->get_info(to_local_seconds(1880y, std::chrono::August, 1d, 23h, 59min, 59s)));

  assert_equal(
      std::chrono::local_info(
          std::chrono::local_info::unique,
          std::chrono::sys_info(
              to_sys_seconds(1880y, std::chrono::August, 2d, 0h, 25min, 21s),
              to_sys_seconds(1916y, std::chrono::May, 21d, 2h, 25min, 21s),
              -(25min + 21s),
              0min,
              "DMT"),
          std::chrono::sys_info(std::chrono::sys_seconds(0s), std::chrono::sys_seconds(0s), 0s, 0min, "")),
      tz->get_info(to_local_seconds(1880y, std::chrono::August, 2d)));

  assert_equal(
      std::chrono::local_info(
          std::chrono::local_info::unique,
          std::chrono::sys_info(
              to_sys_seconds(1880y, std::chrono::August, 2d, 0h, 25min, 21s),
              to_sys_seconds(1916y, std::chrono::May, 21d, 2h, 25min, 21s),
              -(25min + 21s),
              0min,
              "DMT"),
          std::chrono::sys_info(std::chrono::sys_seconds(0s), std::chrono::sys_seconds(0s), 0s, 0min, "")),
      tz->get_info(to_local_seconds(1916y, std::chrono::May, 21d, 1h, 59min, 59s)));

  assert_equal(
      std::chrono::local_info(
          std::chrono::local_info::nonexistent,
          std::chrono::sys_info(
              to_sys_seconds(1880y, std::chrono::August, 2d, 0h, 25min, 21s),
              to_sys_seconds(1916y, std::chrono::May, 21d, 2h, 25min, 21s),
              -(25min + 21s),
              0min,
              "DMT"),
          std::chrono::sys_info(
              to_sys_seconds(1916y, std::chrono::May, 21d, 2h, 25min, 21s),
              to_sys_seconds(1916y, std::chrono::October, 1d, 02h, 25min, 21s),
              34min + 39s,
              60min,
              "IST")),
      tz->get_info(to_local_seconds(1916y, std::chrono::May, 21d, 2h)));

  assert_equal(
      std::chrono::local_info(
          std::chrono::local_info::nonexistent,
          std::chrono::sys_info(
              to_sys_seconds(1880y, std::chrono::August, 2d, 0h, 25min, 21s),
              to_sys_seconds(1916y, std::chrono::May, 21d, 2h, 25min, 21s),
              -(25min + 21s),
              0min,
              "DMT"),
          std::chrono::sys_info(
              to_sys_seconds(1916y, std::chrono::May, 21d, 2h, 25min, 21s),
              to_sys_seconds(1916y, std::chrono::October, 1d, 02h, 25min, 21s),
              34min + 39s,
              60min,
              "IST")),
      tz->get_info(to_local_seconds(1916y, std::chrono::May, 21d, 2h, 59min, 59s)));

  assert_equal(
      std::chrono::local_info(
          std::chrono::local_info::unique,
          std::chrono::sys_info(
              to_sys_seconds(1916y, std::chrono::May, 21d, 2h, 25min, 21s),
              to_sys_seconds(1916y, std::chrono::October, 1d, 02h, 25min, 21s),
              34min + 39s,
              60min,
              "IST"),
          std::chrono::sys_info(std::chrono::sys_seconds(0s), std::chrono::sys_seconds(0s), 0s, 0min, "")),
      tz->get_info(to_local_seconds(1916y, std::chrono::May, 21d, 6h)));

  assert_equal(
      std::chrono::local_info(
          std::chrono::local_info::unique,
          std::chrono::sys_info(
              to_sys_seconds(1916y, std::chrono::May, 21d, 2h, 25min, 21s),
              to_sys_seconds(1916y, std::chrono::October, 1d, 02h, 25min, 21s),
              34min + 39s,
              60min,
              "IST"),
          std::chrono::sys_info(std::chrono::sys_seconds(0s), std::chrono::sys_seconds(0s), 0s, 0min, "")),
      tz->get_info(to_local_seconds(1916y, std::chrono::October, 1d, 2h, 25min, 20s)));

  assert_equal(
      std::chrono::local_info(
          std::chrono::local_info::ambiguous,
          std::chrono::sys_info(
              to_sys_seconds(1916y, std::chrono::May, 21d, 2h, 25min, 21s),
              to_sys_seconds(1916y, std::chrono::October, 1d, 02h, 25min, 21s),
              34min + 39s,
              60min,
              "IST"),
          std::chrono::sys_info(
              to_sys_seconds(1916y, std::chrono::October, 1d, 02h, 25min, 21s),
              to_sys_seconds(1917y, std::chrono::April, 8d, 2h),
              0s,
              0min,
              "GMT")),
      tz->get_info(to_local_seconds(1916y, std::chrono::October, 1d, 2h, 59min, 59s)));

  assert_equal(
      std::chrono::local_info(
          std::chrono::local_info::unique,
          std::chrono::sys_info(
              to_sys_seconds(1916y, std::chrono::October, 1d, 02h, 25min, 21s),
              to_sys_seconds(1917y, std::chrono::April, 8d, 2h),
              0s,
              0min,
              "GMT"),
          std::chrono::sys_info(std::chrono::sys_seconds(0s), std::chrono::sys_seconds(0s), 0s, 0min, "")),
      tz->get_info(to_local_seconds(1916y, std::chrono::October, 1d, 3h)));

  assert_equal(
      std::chrono::local_info(
          std::chrono::local_info::unique,
          std::chrono::sys_info(
              to_sys_seconds(1916y, std::chrono::October, 1d, 02h, 25min, 21s),
              to_sys_seconds(1917y, std::chrono::April, 8d, 2h),
              0s,
              0min,
              "GMT"),
          std::chrono::sys_info(std::chrono::sys_seconds(0s), std::chrono::sys_seconds(0s), 0s, 0min, "")),
      tz->get_info(to_local_seconds(1917y, std::chrono::April, 8d, 1h, 59min, 59s)));
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
  // ...
  //
  // R j 1917 o - Ap 8 2 1 D
  // R j 1917 o - S 17 2 0 S
  // R j 1919 o - May 5 23 1 D
  // R j 1919 o - Au 12 23 0 S
  // R j 1920 1935 - May Su>=1 23 1 D
  // ...
  //
  // R C 1918 o - Ap 14 2 1 D
  // R C 1918 o - O 27 2 0 S
  // R C 1942 o - F 9 2 1 W
  // ...

  using namespace std::literals::chrono_literals;
  const std::chrono::time_zone* tz = std::chrono::locate_zone("America/St_Johns");

  assert_equal(
      std::chrono::local_info(
          std::chrono::local_info::unique,
          std::chrono::sys_info(
              std::chrono::sys_seconds::min(),
              to_sys_seconds(1884y, std::chrono::January, 1d, 3h, 30min, 52s),
              -(3h + 30min + 52s),
              0min,
              "LMT"),
          std::chrono::sys_info(std::chrono::sys_seconds(0s), std::chrono::sys_seconds(0s), 0s, 0min, "")),
      tz->get_info(std::chrono::local_seconds::min()));

  assert_equal(
      std::chrono::local_info(
          std::chrono::local_info::unique,
          std::chrono::sys_info(
              std::chrono::sys_seconds::min(),
              to_sys_seconds(1884y, std::chrono::January, 1d, 3h, 30min, 52s),
              -(3h + 30min + 52s),
              0min,
              "LMT"),
          std::chrono::sys_info(std::chrono::sys_seconds(0s), std::chrono::sys_seconds(0s), 0s, 0min, "")),
      tz->get_info(to_local_seconds(1883y, std::chrono::December, 31d, 23h, 59min, 59s)));

  assert_equal(
      std::chrono::local_info(
          std::chrono::local_info::unique,
          std::chrono::sys_info(
              to_sys_seconds(1884y, std::chrono::January, 1d, 3h, 30min, 52s),
              to_sys_seconds(1917y, std::chrono::April, 8d, 5h, 30min, 52s),
              -(3h + 30min + 52s),
              0min,
              "NST"),
          std::chrono::sys_info(std::chrono::sys_seconds(0s), std::chrono::sys_seconds(0s), 0s, 0min, "")),
      tz->get_info(to_local_seconds(1884y, std::chrono::January, 1d)));

  assert_equal(
      std::chrono::local_info(
          std::chrono::local_info::unique,
          std::chrono::sys_info(
              to_sys_seconds(1884y, std::chrono::January, 1d, 3h, 30min, 52s),
              to_sys_seconds(1917y, std::chrono::April, 8d, 5h, 30min, 52s),
              -(3h + 30min + 52s),
              0min,
              "NST"),
          std::chrono::sys_info(std::chrono::sys_seconds(0s), std::chrono::sys_seconds(0s), 0s, 0min, "")),
      tz->get_info(to_local_seconds(1917y, std::chrono::April, 8d, 1h, 59min, 59s)));

  assert_equal(
      std::chrono::local_info(
          std::chrono::local_info::nonexistent,
          std::chrono::sys_info(
              to_sys_seconds(1884y, std::chrono::January, 1d, 3h, 30min, 52s),
              to_sys_seconds(1917y, std::chrono::April, 8d, 5h, 30min, 52s),
              -(3h + 30min + 52s),
              0min,
              "NST"),
          std::chrono::sys_info(
              to_sys_seconds(1917y, std::chrono::April, 8d, 5h, 30min, 52s),
              to_sys_seconds(1917y, std::chrono::September, 17d, 4h, 30min, 52s),
              -(2h + 30min + 52s),
              60min,
              "NDT")),
      tz->get_info(to_local_seconds(1917y, std::chrono::April, 8d, 2h)));

  assert_equal(
      std::chrono::local_info(
          std::chrono::local_info::nonexistent,
          std::chrono::sys_info(
              to_sys_seconds(1884y, std::chrono::January, 1d, 3h, 30min, 52s),
              to_sys_seconds(1917y, std::chrono::April, 8d, 5h, 30min, 52s),
              -(3h + 30min + 52s),
              0min,
              "NST"),
          std::chrono::sys_info(
              to_sys_seconds(1917y, std::chrono::April, 8d, 5h, 30min, 52s),
              to_sys_seconds(1917y, std::chrono::September, 17d, 4h, 30min, 52s),
              -(2h + 30min + 52s),
              60min,
              "NDT")),
      tz->get_info(to_local_seconds(1917y, std::chrono::April, 8d, 2h, 59min, 59s)));

  assert_equal(
      std::chrono::local_info(
          std::chrono::local_info::unique,
          std::chrono::sys_info(
              to_sys_seconds(1917y, std::chrono::April, 8d, 5h, 30min, 52s),
              to_sys_seconds(1917y, std::chrono::September, 17d, 4h, 30min, 52s),
              -(2h + 30min + 52s),
              60min,
              "NDT"),
          std::chrono::sys_info(std::chrono::sys_seconds(0s), std::chrono::sys_seconds(0s), 0s, 0min, "")),
      tz->get_info(to_local_seconds(1917y, std::chrono::April, 8d, 3h)));

  assert_equal(
      std::chrono::local_info(
          std::chrono::local_info::unique,
          std::chrono::sys_info(
              to_sys_seconds(1917y, std::chrono::April, 8d, 5h, 30min, 52s),
              to_sys_seconds(1917y, std::chrono::September, 17d, 4h, 30min, 52s),
              -(2h + 30min + 52s),
              60min,
              "NDT"),
          std::chrono::sys_info(std::chrono::sys_seconds(0s), std::chrono::sys_seconds(0s), 0s, 0min, "")),
      tz->get_info(to_local_seconds(1917y, std::chrono::September, 17d, 0h, 59min, 59s)));

  assert_equal(
      std::chrono::local_info(
          std::chrono::local_info::ambiguous,
          std::chrono::sys_info(
              to_sys_seconds(1917y, std::chrono::April, 8d, 5h, 30min, 52s),
              to_sys_seconds(1917y, std::chrono::September, 17d, 4h, 30min, 52s),
              -(2h + 30min + 52s),
              60min,
              "NDT"),
          std::chrono::sys_info(
              to_sys_seconds(1917y, std::chrono::September, 17d, 4h, 30min, 52s),
              to_sys_seconds(1918y, std::chrono::April, 14d, 5h, 30min, 52s),
              -(3h + 30min + 52s),
              0min,
              "NST")),
      tz->get_info(to_local_seconds(1917y, std::chrono::September, 17d, 1h)));

  assert_equal(
      std::chrono::local_info(
          std::chrono::local_info::ambiguous,
          std::chrono::sys_info(
              to_sys_seconds(1917y, std::chrono::April, 8d, 5h, 30min, 52s),
              to_sys_seconds(1917y, std::chrono::September, 17d, 4h, 30min, 52s),
              -(2h + 30min + 52s),
              60min,
              "NDT"),
          std::chrono::sys_info(
              to_sys_seconds(1917y, std::chrono::September, 17d, 4h, 30min, 52s),
              to_sys_seconds(1918y, std::chrono::April, 14d, 5h, 30min, 52s),
              -(3h + 30min + 52s),
              0min,
              "NST")),
      tz->get_info(to_local_seconds(1917y, std::chrono::September, 17d, 1h, 59min, 59s)));

  assert_equal(
      std::chrono::local_info(
          std::chrono::local_info::unique,
          std::chrono::sys_info(
              to_sys_seconds(1917y, std::chrono::September, 17d, 4h, 30min, 52s),
              to_sys_seconds(1918y, std::chrono::April, 14d, 5h, 30min, 52s),
              -(3h + 30min + 52s),
              0min,
              "NST"),
          std::chrono::sys_info(std::chrono::sys_seconds(0s), std::chrono::sys_seconds(0s), 0s, 0min, "")),
      tz->get_info(to_local_seconds(1917y, std::chrono::September, 17d, 2h)));
}

static void validate_transitions(const std::chrono::time_zone& zone) {
  using namespace std::literals::chrono_literals;

  constexpr auto begin = std::chrono::time_point_cast<std::chrono::seconds>(
      static_cast<std::chrono::sys_days>(std::chrono::year_month_day{first, std::chrono::January, 1d}));
  constexpr auto end = std::chrono::time_point_cast<std::chrono::seconds>(
      static_cast<std::chrono::sys_days>(std::chrono::year_month_day{last, std::chrono::January, 1d}));

  // Builds the set of sys_info objects for the selected time range.
  std::vector<std::chrono::sys_info> input;
  std::chrono::sys_seconds s = begin;
  do {
    input.emplace_back(zone.get_info(s));
    s = input.back().end;
  } while (s < end);

  for (auto previous = input.begin(), next = previous + 1; next != input.end(); ++previous, ++next) {
    // Now iterates to all adjacent objects.
    // For every transition gets the locate time of the
    // - end of the first          (a)
    // - the start if the second   (b)
    // Depending on the difference between 'a' and 'b' different tests are done.
    std::chrono::local_seconds end_previous{previous->end.time_since_epoch() + previous->offset};
    std::chrono::local_seconds begin_next{next->begin.time_since_epoch() + next->offset};

    if (end_previous == begin_next) {
      // unique transition
      // a |------------|
      // b              |----------|
      //                T
      assert_equal(std::chrono::local_info(
                       std::chrono::local_info::unique,
                       *previous,
                       std::chrono::sys_info(std::chrono::sys_seconds(0s), std::chrono::sys_seconds(0s), 0s, 0min, "")),
                   zone.get_info(end_previous - 1s));

      assert_equal(std::chrono::local_info(
                       std::chrono::local_info::unique,
                       *next,
                       std::chrono::sys_info(std::chrono::sys_seconds(0s), std::chrono::sys_seconds(0s), 0s, 0min, "")),
                   zone.get_info(begin_next));

    } else if (end_previous < begin_next) {
      // non-existent transition
      // a |------------|
      // b                 |----------|
      //                T  T
      assert_equal(std::chrono::local_info(
                       std::chrono::local_info::unique,
                       *previous,
                       std::chrono::sys_info(std::chrono::sys_seconds(0s), std::chrono::sys_seconds(0s), 0s, 0min, "")),
                   zone.get_info(end_previous - 1s));

      assert_equal(std::chrono::local_info(std::chrono::local_info::nonexistent, *previous, *next),
                   zone.get_info(end_previous));

      assert_equal(std::chrono::local_info(std::chrono::local_info::nonexistent, *previous, *next),
                   zone.get_info(begin_next - 1s));

      assert_equal(std::chrono::local_info(
                       std::chrono::local_info::unique,
                       *next,
                       std::chrono::sys_info(std::chrono::sys_seconds(0s), std::chrono::sys_seconds(0s), 0s, 0min, "")),
                   zone.get_info(begin_next));

    } else {
      // ambiguous transition
      // a |------------|
      // b           |----------|
      //             T  T
      assert_equal(std::chrono::local_info(
                       std::chrono::local_info::unique,
                       *previous,
                       std::chrono::sys_info(std::chrono::sys_seconds(0s), std::chrono::sys_seconds(0s), 0s, 0min, "")),
                   zone.get_info(begin_next - 1s));

      assert_equal(std::chrono::local_info(std::chrono::local_info::ambiguous, *previous, *next),
                   zone.get_info(begin_next));

      assert_equal(std::chrono::local_info(std::chrono::local_info::ambiguous, *previous, *next),
                   zone.get_info(end_previous - 1s));

      assert_equal(std::chrono::local_info(
                       std::chrono::local_info::unique,
                       *next,
                       std::chrono::sys_info(std::chrono::sys_seconds(0s), std::chrono::sys_seconds(0s), 0s, 0min, "")),
                   zone.get_info(end_previous));
    }
  }
}

int main(int, const char**) {
  test_gmt();
  test_local_time_out_of_range();
  test_indian_kerguelen();
  test_antarctica_rothera();

  test_asia_hong_kong();
  test_europe_berlin();
  test_europe_dublin();
  test_america_st_johns();

  const std::chrono::tzdb& tzdb = std::chrono::get_tzdb();
  for (const auto& zone : tzdb.zones) {
    validate_transitions(zone);
  }

  return 0;
}
