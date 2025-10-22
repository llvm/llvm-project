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

// class utc_clock;

// static sys_time<common_type_t<_Duration, seconds>>
// to_sys(const utc_time<_Duration>& __time);

#include <chrono>
#include <cmath>
#include <cassert>

#include "test_macros.h"
#include "assert_macros.h"
#include "concat_macros.h"

template <class Duration>
static void test_leap_seconds(std::chrono::utc_time<Duration> time, std::chrono::sys_time<Duration> expected) {
  auto result = std::chrono::utc_clock::to_sys(time);
  TEST_REQUIRE(
      result == expected,
      TEST_WRITE_CONCATENATED("\tTime: ", time, "\nExpected output ", expected, "\nActual output   ", result, '\n'));
}

static std::chrono::sys_seconds get_sys_time(long long seconds_since_1900) {
  // The file leap-seconds.list stores dates since 1 January 1900, 00:00:00, we want
  // seconds since 1 January 1970.
  constexpr auto offset =
      std::chrono::sys_days{std::chrono::January / 1 / 1970} - std::chrono::sys_days{std::chrono::January / 1 / 1900};
  return std::chrono::sys_seconds{std::chrono::seconds{seconds_since_1900} - offset};
}

// Tests the set of existing database entries at the time of writing. Since
// the last leap second insertion is several years ago, it's expected all
// systems have the same information. (Adding new entries in the future does
// not affect this test.)
static void test_transitions() {
  using namespace std::literals::chrono_literals;

  test_leap_seconds(std::chrono::utc_seconds::min(), std::chrono::sys_seconds::min());

  // Epoch transition no transitions.
  test_leap_seconds(std::chrono::utc_seconds{-1s}, std::chrono::sys_seconds{-1s});
  test_leap_seconds(std::chrono::utc_seconds{0s}, std::chrono::sys_seconds{0s});
  test_leap_seconds(std::chrono::utc_seconds{1s}, std::chrono::sys_seconds{1s});

  // "sys" is the time of the transition to the next leap second.
  // "elapsed" is the number of leap seconds before the transition.
  // "positive" is the leap second added +1s? If not it's -1s.
  auto test_transition = [](std::chrono::sys_seconds sys, std::chrono::seconds elapsed, bool positive) {
    // Note at the time of writing all leap seconds are positive so the else
    // branch is never executed. The private test for this function tests
    // negative leap seconds and uses the else branch.

    std::chrono::utc_seconds utc = std::chrono::utc_seconds{sys.time_since_epoch()} + elapsed;
    if (positive) {
      // Every transition has the following tests
      // - 1ns before the start of the transition no adjustment needed
      // -         at the start of the transition sys is clamped at the time just prior to the moment
      //                                          of the leap second insertion. The exact value depends
      //                                          on the resolution of the result type.
      // - 1ns before the end   of the transition sys is still clamped like before
      // -         at the end   of the transition sys is 1s behind the utc time
      // - 1ns after  the end   of the transition sys is still 1s behind the utc time
      test_leap_seconds(utc - 1ns, sys - 1ns);
      test_leap_seconds(utc, sys - 1s);
      test_leap_seconds(utc + 0ns, sys - 1ns);
      test_leap_seconds(utc + 1s - 1ns, sys - 1ns);
      test_leap_seconds(utc + 1s, sys);
      test_leap_seconds(utc + 1s + 0ns, sys + 0ns);
      test_leap_seconds(utc + 1s + 1ns, sys + 1ns);
    } else {
      // Every transition has the following tests
      // - 1ns before the transition no adjustment needed
      // -         at the transition sys is 1s ahead of the utc time
      // - 1ns after  the transition sys is still 1s ahead of the utc time
      test_leap_seconds(utc - 1ns, sys - 1ns);
      test_leap_seconds(utc, sys + 1s);
      test_leap_seconds(utc + 1ns, sys + 1s + 1ns);
    }
  };

  // Transitions from the start of UTC.
  test_transition(get_sys_time(2287785600), 0s, true);  // 1 Jul 1972
  test_transition(get_sys_time(2303683200), 1s, true);  // 1 Jan 1973
  test_transition(get_sys_time(2335219200), 2s, true);  // 1 Jan 1974
  test_transition(get_sys_time(2366755200), 3s, true);  // 1 Jan 1975
  test_transition(get_sys_time(2398291200), 4s, true);  // 1 Jan 1976
  test_transition(get_sys_time(2429913600), 5s, true);  // 1 Jan 1977
  test_transition(get_sys_time(2461449600), 6s, true);  // 1 Jan 1978
  test_transition(get_sys_time(2492985600), 7s, true);  // 1 Jan 1979
  test_transition(get_sys_time(2524521600), 8s, true);  // 1 Jan 1980
  test_transition(get_sys_time(2571782400), 9s, true);  // 1 Jul 1981
  test_transition(get_sys_time(2603318400), 10s, true); // 1 Jul 1982
  test_transition(get_sys_time(2634854400), 11s, true); // 1 Jul 1983
  test_transition(get_sys_time(2698012800), 12s, true); // 1 Jul 1985
  test_transition(get_sys_time(2776982400), 13s, true); // 1 Jan 1988
  test_transition(get_sys_time(2840140800), 14s, true); // 1 Jan 1990
  test_transition(get_sys_time(2871676800), 15s, true); // 1 Jan 1991
  test_transition(get_sys_time(2918937600), 16s, true); // 1 Jul 1992
  test_transition(get_sys_time(2950473600), 17s, true); // 1 Jul 1993
  test_transition(get_sys_time(2982009600), 18s, true); // 1 Jul 1994
  test_transition(get_sys_time(3029443200), 19s, true); // 1 Jan 1996
  test_transition(get_sys_time(3076704000), 20s, true); // 1 Jul 1997
  test_transition(get_sys_time(3124137600), 21s, true); // 1 Jan 1999
  test_transition(get_sys_time(3345062400), 22s, true); // 1 Jan 2006
  test_transition(get_sys_time(3439756800), 23s, true); // 1 Jan 2009
  test_transition(get_sys_time(3550089600), 24s, true); // 1 Jul 2012
  test_transition(get_sys_time(3644697600), 25s, true); // 1 Jul 2015
  test_transition(get_sys_time(3692217600), 26s, true); // 1 Jan 2017
}

// Tests the transition for clocks where the duration's rep is a floating-point type.
static void test_transitions_floating_point() {
  using namespace std::literals::chrono_literals;

  // Based on test_transitions but uses a floating-point duration.
  using F = float;

  auto test_transition = [](std::chrono::sys_seconds sys, std::chrono::seconds elapsed, bool positive) {
    // Note at the time of writing all leap seconds are positive so the else
    // branch is never executed. The private test for this function tests
    // negative leap seconds and uses the else branch.

    std::chrono::utc_seconds utc = std::chrono::utc_seconds{sys.time_since_epoch()} + elapsed;

    using D = std::chrono::duration<F>;
    using S = std::chrono ::time_point<std::chrono::system_clock, D>;
    using U = std::chrono ::time_point<std::chrono::utc_clock, D>;

    S s{sys.time_since_epoch()};
    bool is_leap_second = s.time_since_epoch().count() == sys.time_since_epoch().count();
    assert(is_leap_second);

    U u{utc.time_since_epoch()};
    if (positive) {
      test_leap_seconds(u - 1ns, s - 1ns);
      test_leap_seconds(u, s - 1s);
      test_leap_seconds(u + 0ns, s - 1ns);
      test_leap_seconds(u + 1s - 1ns, s - 1ns);
      test_leap_seconds(u + 1s, s);
      test_leap_seconds(u + 1s + 0ns, s + 0ns);
      test_leap_seconds(u + 1s + 1ns, s + 1ns);

      test_leap_seconds(U{D{std::nextafter(u.time_since_epoch().count(), F{0})}},
                        S{D{std::nextafter(s.time_since_epoch().count(), F{0})}});
      test_leap_seconds(u, S{D{s.time_since_epoch().count() - F{1}}});
      test_leap_seconds(U{D{u.time_since_epoch().count() + F{1}}}, s);
      test_leap_seconds(U{D{std::nextafter(u.time_since_epoch().count() + F{1}, std::numeric_limits<F>::max())}},
                        S{D{std::nextafter(s.time_since_epoch().count(), std::numeric_limits<F>::max())}});
    }
  };

  // Transitions from the start of UTC.
  test_transition(get_sys_time(2287785600), 0s, true);  // 1 Jul 1972
  test_transition(get_sys_time(2303683200), 1s, true);  // 1 Jan 1973
  test_transition(get_sys_time(2335219200), 2s, true);  // 1 Jan 1974
  test_transition(get_sys_time(2366755200), 3s, true);  // 1 Jan 1975
  test_transition(get_sys_time(2398291200), 4s, true);  // 1 Jan 1976
  test_transition(get_sys_time(2429913600), 5s, true);  // 1 Jan 1977
  test_transition(get_sys_time(2461449600), 6s, true);  // 1 Jan 1978
  test_transition(get_sys_time(2492985600), 7s, true);  // 1 Jan 1979
  test_transition(get_sys_time(2524521600), 8s, true);  // 1 Jan 1980
  test_transition(get_sys_time(2571782400), 9s, true);  // 1 Jul 1981
  test_transition(get_sys_time(2603318400), 10s, true); // 1 Jul 1982
  test_transition(get_sys_time(2634854400), 11s, true); // 1 Jul 1983
  test_transition(get_sys_time(2698012800), 12s, true); // 1 Jul 1985
  test_transition(get_sys_time(2776982400), 13s, true); // 1 Jan 1988
  test_transition(get_sys_time(2840140800), 14s, true); // 1 Jan 1990
  test_transition(get_sys_time(2871676800), 15s, true); // 1 Jan 1991
  test_transition(get_sys_time(2918937600), 16s, true); // 1 Jul 1992
  test_transition(get_sys_time(2950473600), 17s, true); // 1 Jul 1993
  test_transition(get_sys_time(2982009600), 18s, true); // 1 Jul 1994
  test_transition(get_sys_time(3029443200), 19s, true); // 1 Jan 1996
  test_transition(get_sys_time(3076704000), 20s, true); // 1 Jul 1997
  test_transition(get_sys_time(3124137600), 21s, true); // 1 Jan 1999
  test_transition(get_sys_time(3345062400), 22s, true); // 1 Jan 2006
  test_transition(get_sys_time(3439756800), 23s, true); // 1 Jan 2009
  test_transition(get_sys_time(3550089600), 24s, true); // 1 Jul 2012
  test_transition(get_sys_time(3644697600), 25s, true); // 1 Jul 2015
  test_transition(get_sys_time(3692217600), 26s, true); // 1 Jan 2017
}

// Tests whether the return type is the expected type.
static void test_return_type() {
  namespace cr = std::chrono;
  using namespace std::literals::chrono_literals;

  {
    [[maybe_unused]] std::same_as<cr::sys_time<cr::nanoseconds>> decltype(auto) _ =
        cr::utc_clock::to_sys(cr::utc_time<cr::nanoseconds>{0ns});
  }
  {
    [[maybe_unused]] std::same_as<cr::sys_time<cr::microseconds>> decltype(auto) _ =
        cr::utc_clock::to_sys(cr::utc_time<cr::microseconds>{0us});
  }
  {
    [[maybe_unused]] std::same_as<cr::sys_time<cr::milliseconds>> decltype(auto) _ =
        cr::utc_clock::to_sys(cr::utc_time<cr::milliseconds>{0ms});
  }

  {
    [[maybe_unused]] std::same_as<cr::sys_time<cr::seconds>> decltype(auto) _ =
        cr::utc_clock::to_sys(cr::utc_time<cr::seconds>{cr::seconds{0}});
  }

  {
    [[maybe_unused]] std::same_as<cr::sys_time<cr::seconds>> decltype(auto) _ =
        cr::utc_clock::to_sys(cr::utc_time<cr::minutes>{cr::minutes{0}});
  }
  {
    [[maybe_unused]] std::same_as<cr::sys_time<cr::seconds>> decltype(auto) _ =
        cr::utc_clock::to_sys(cr::utc_time<cr::hours>{cr::hours{0}});
  }
  {
    [[maybe_unused]] std::same_as<cr::sys_time<cr::seconds>> decltype(auto) _ =
        cr::utc_clock::to_sys(cr::utc_time<cr::days>{cr::days{0}});
  }
  {
    [[maybe_unused]] std::same_as<cr::sys_time<cr::seconds>> decltype(auto) _ =
        cr::utc_clock::to_sys(cr::utc_time<cr::weeks>{cr::weeks{0}});
  }
  {
    [[maybe_unused]] std::same_as<cr::sys_time<cr::seconds>> decltype(auto) _ =
        cr::utc_clock::to_sys(cr::utc_time<cr::months>{cr::months{0}});
  }
  {
    [[maybe_unused]] std::same_as<cr::sys_time<cr::seconds>> decltype(auto) _ =
        cr::utc_clock::to_sys(cr::utc_time<cr::years>{cr::years{0}});
  }
}

int main(int, const char**) {
  test_transitions();
  test_transitions_floating_point();
  test_return_type();

  return 0;
}
