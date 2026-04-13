//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: no-filesystem, no-localization, no-tzdb

// XFAIL: libcpp-has-no-experimental-tzdb
// XFAIL: availability-tzdb-missing

// <chrono>
//
// class utc_clock;

// template<class Duration>
// leap_second_info get_leap_second_info(const utc_time<Duration>& ut);

#include <chrono>
#include <cassert>

#include "test_macros.h"
#include "assert_macros.h"
#include "concat_macros.h"

template <class Duration>
static void test_leap_second_info(
    std::chrono::time_point<std::chrono::utc_clock, Duration> time, bool is_leap_second, std::chrono::seconds elapsed) {
  std::chrono::leap_second_info result = std::chrono::get_leap_second_info(time);
  TEST_REQUIRE(
      result.is_leap_second == is_leap_second && result.elapsed == elapsed,
      TEST_WRITE_CONCATENATED(
          "\nExpected output [",
          is_leap_second,
          ", ",
          elapsed,
          "]\nActual output   [",
          result.is_leap_second,
          ", ",
          result.elapsed,
          "]\n"));
}

static std::chrono::utc_seconds get_utc_time(long long seconds_since_1900) {
  // The file leap-seconds.list stores dates since 1 January 1900, 00:00:00, we want
  // seconds since 1 January 1970.
  constexpr auto offset =
      std::chrono::sys_days{std::chrono::January / 1 / 1970} - std::chrono::sys_days{std::chrono::January / 1 / 1900};
  return std::chrono::utc_seconds{std::chrono::seconds{seconds_since_1900} - offset};
}

// Tests set of existing database entries at the time of writing.
int main(int, const char**) {
  using namespace std::literals::chrono_literals;

  test_leap_second_info(std::chrono::utc_seconds::min(), false, 0s);

  // Epoch transition no transitions.
  test_leap_second_info(std::chrono::utc_seconds{-1s}, false, 0s);
  test_leap_second_info(std::chrono::utc_seconds{0s}, false, 0s);
  test_leap_second_info(std::chrono::utc_seconds{1s}, false, 0s);

  // Transitions from the start of UTC.
  auto test_transition = [](std::chrono::utc_seconds time, std::chrono::seconds elapsed, bool positive) {
    // Note at the time of writing all leap seconds are positive so the else
    // branch is never executed. The private test for this function tests
    // negative leap seconds and uses the else branch.

    if (positive) {
      // Every transition has the following tests
      // - 1ns before the start of the transition is_leap_second -> false, elapsed -> elapsed
      // -         at the start of the transition is_leap_second -> true,  elapsed -> elapsed + 1
      // - 1ns after  the start of the transition is_leap_second -> true,  elapsed -> elapsed + 1
      // - 1ns before the end   of the transition is_leap_second -> true,  elapsed -> elapsed + 1
      // -         at the end   of the transition is_leap_second -> false, elapsed -> elapsed + 1

      test_leap_second_info(time - 1ns, false, elapsed);
      test_leap_second_info(time, true, elapsed + 1s);
      test_leap_second_info(time + 1ns, true, elapsed + 1s);
      test_leap_second_info(time + 1s - 1ns, true, elapsed + 1s);
      test_leap_second_info(time + 1s, false, elapsed + 1s);
    } else {
      // Every transition has the following tests
      // - 1ns before the transition is_leap_second -> false, elapsed -> elapsed
      // -         at the transition is_leap_second -> false  elapsed -> elapsed - 1
      // - 1ns after  the transition is_leap_second -> false, elapsed -> elapsed - 1
      test_leap_second_info(time - 1ns, false, elapsed);
      test_leap_second_info(time, false, elapsed - 1s);
      test_leap_second_info(time + 1ns, false, elapsed - 1s);
    }
  };

  // The timestamps are from leap-seconds.list in the IANA database.
  // Note the times stamps are timestamps without leap seconds so the number
  // here are incremented by x "leap seconds".
  test_transition(get_utc_time(2287785600 + 0), 0s, true);   // 1 Jul 1972
  test_transition(get_utc_time(2303683200 + 1), 1s, true);   // 1 Jan 1973
  test_transition(get_utc_time(2335219200 + 2), 2s, true);   // 1 Jan 1974
  test_transition(get_utc_time(2366755200 + 3), 3s, true);   // 1 Jan 1975
  test_transition(get_utc_time(2398291200 + 4), 4s, true);   // 1 Jan 1976
  test_transition(get_utc_time(2429913600 + 5), 5s, true);   // 1 Jan 1977
  test_transition(get_utc_time(2461449600 + 6), 6s, true);   // 1 Jan 1978
  test_transition(get_utc_time(2492985600 + 7), 7s, true);   // 1 Jan 1979
  test_transition(get_utc_time(2524521600 + 8), 8s, true);   // 1 Jan 1980
  test_transition(get_utc_time(2571782400 + 9), 9s, true);   // 1 Jul 1981
  test_transition(get_utc_time(2603318400 + 10), 10s, true); // 1 Jul 1982
  test_transition(get_utc_time(2634854400 + 11), 11s, true); // 1 Jul 1983
  test_transition(get_utc_time(2698012800 + 12), 12s, true); // 1 Jul 1985
  test_transition(get_utc_time(2776982400 + 13), 13s, true); // 1 Jan 1988
  test_transition(get_utc_time(2840140800 + 14), 14s, true); // 1 Jan 1990
  test_transition(get_utc_time(2871676800 + 15), 15s, true); // 1 Jan 1991
  test_transition(get_utc_time(2918937600 + 16), 16s, true); // 1 Jul 1992
  test_transition(get_utc_time(2950473600 + 17), 17s, true); // 1 Jul 1993
  test_transition(get_utc_time(2982009600 + 18), 18s, true); // 1 Jul 1994
  test_transition(get_utc_time(3029443200 + 19), 19s, true); // 1 Jan 1996
  test_transition(get_utc_time(3076704000 + 20), 20s, true); // 1 Jul 1997
  test_transition(get_utc_time(3124137600 + 21), 21s, true); // 1 Jan 1999
  test_transition(get_utc_time(3345062400 + 22), 22s, true); // 1 Jan 2006
  test_transition(get_utc_time(3439756800 + 23), 23s, true); // 1 Jan 2009
  test_transition(get_utc_time(3550089600 + 24), 24s, true); // 1 Jul 2012
  test_transition(get_utc_time(3644697600 + 25), 25s, true); // 1 Jul 2015
  test_transition(get_utc_time(3692217600 + 26), 26s, true); // 1 Jan 2017

  return 0;
}
