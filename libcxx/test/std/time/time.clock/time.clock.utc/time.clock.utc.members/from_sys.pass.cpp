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
// static utc_time<common_type_t<Duration, seconds>>
// from_sys(const sys_time<Duration>& time);

#include <chrono>
#include <cassert>

#include "test_macros.h"
#include "assert_macros.h"
#include "concat_macros.h"

template <class Duration>
static void test_leap_seconds(std::chrono::time_point<std::chrono::system_clock, Duration> time,
                              std::chrono::seconds leap_seconds) {
  auto utc  = std::chrono::utc_clock::from_sys(time);
  auto diff = utc.time_since_epoch() - time.time_since_epoch();
  TEST_REQUIRE(
      diff == leap_seconds,
      TEST_WRITE_CONCATENATED("\tTime: ", time, "\nExpected output ", leap_seconds, "\nActual output   ", diff, '\n'));
}

// This test is based on the example in [time.clock.utc.members]/3
static void test_example_standard() {
  using namespace std::literals::chrono_literals;

  auto t = std::chrono::sys_days{std::chrono::July / 1 / 2015} - 2ns;
  test_leap_seconds(t, 25s);

  t += 1ns;
  test_leap_seconds(t, 25s);

  t += 1ns;
  test_leap_seconds(t, 26s);

  t += 1ns;
  test_leap_seconds(t, 26s);
}

// Tests set of existing database entries at the time of writing.
static void test_transitions() {
  using namespace std::literals::chrono_literals;

  test_leap_seconds(std::chrono::sys_seconds::min(), 0s);
  test_leap_seconds(std::chrono::sys_days::min(), 0s);

  // Epoch transition no transitions.
  test_leap_seconds(std::chrono::sys_seconds{-1s}, 0s);
  test_leap_seconds(std::chrono::sys_seconds{0s}, 0s);
  test_leap_seconds(std::chrono::sys_seconds{1s}, 0s);

  // Transitions from the start of UTC.
  test_leap_seconds(std::chrono::sys_days{std::chrono::January / 1 / 1972} - 1ns, 0s);
  test_leap_seconds(std::chrono::sys_days{std::chrono::January / 1 / 1972}, 0s);
  test_leap_seconds(std::chrono::sys_days{std::chrono::January / 1 / 1972} + 1ns, 0s);

  test_leap_seconds(std::chrono::sys_days{std::chrono::July / 1 / 1972} - 1ns, 0s);
  test_leap_seconds(std::chrono::sys_days{std::chrono::July / 1 / 1972}, 1s);
  test_leap_seconds(std::chrono::sys_days{std::chrono::July / 1 / 1972} + 1ns, 1s);

  test_leap_seconds(std::chrono::sys_days{std::chrono::January / 1 / 1973} - 1ns, 1s);
  test_leap_seconds(std::chrono::sys_days{std::chrono::January / 1 / 1973}, 2s);
  test_leap_seconds(std::chrono::sys_days{std::chrono::January / 1 / 1973} + 1ns, 2s);

  test_leap_seconds(std::chrono::sys_days{std::chrono::January / 1 / 1974} - 1ns, 2s);
  test_leap_seconds(std::chrono::sys_days{std::chrono::January / 1 / 1974}, 3s);
  test_leap_seconds(std::chrono::sys_days{std::chrono::January / 1 / 1974} + 1ns, 3s);

  test_leap_seconds(std::chrono::sys_days{std::chrono::January / 1 / 1975} - 1ns, 3s);
  test_leap_seconds(std::chrono::sys_days{std::chrono::January / 1 / 1975}, 4s);
  test_leap_seconds(std::chrono::sys_days{std::chrono::January / 1 / 1975} + 1ns, 4s);

  test_leap_seconds(std::chrono::sys_days{std::chrono::January / 1 / 1976} - 1ns, 4s);
  test_leap_seconds(std::chrono::sys_days{std::chrono::January / 1 / 1976}, 5s);
  test_leap_seconds(std::chrono::sys_days{std::chrono::January / 1 / 1976} + 1ns, 5s);

  test_leap_seconds(std::chrono::sys_days{std::chrono::January / 1 / 1977} - 1ns, 5s);
  test_leap_seconds(std::chrono::sys_days{std::chrono::January / 1 / 1977}, 6s);
  test_leap_seconds(std::chrono::sys_days{std::chrono::January / 1 / 1977} + 1ns, 6s);

  test_leap_seconds(std::chrono::sys_days{std::chrono::January / 1 / 1978} - 1ns, 6s);
  test_leap_seconds(std::chrono::sys_days{std::chrono::January / 1 / 1978}, 7s);
  test_leap_seconds(std::chrono::sys_days{std::chrono::January / 1 / 1978} + 1ns, 7s);

  test_leap_seconds(std::chrono::sys_days{std::chrono::January / 1 / 1979} - 1ns, 7s);
  test_leap_seconds(std::chrono::sys_days{std::chrono::January / 1 / 1979}, 8s);
  test_leap_seconds(std::chrono::sys_days{std::chrono::January / 1 / 1979} + 1ns, 8s);

  test_leap_seconds(std::chrono::sys_days{std::chrono::January / 1 / 1980} - 1ns, 8s);
  test_leap_seconds(std::chrono::sys_days{std::chrono::January / 1 / 1980}, 9s);
  test_leap_seconds(std::chrono::sys_days{std::chrono::January / 1 / 1980} + 1ns, 9s);

  test_leap_seconds(std::chrono::sys_days{std::chrono::July / 1 / 1981} - 1ns, 9s);
  test_leap_seconds(std::chrono::sys_days{std::chrono::July / 1 / 1981}, 10s);
  test_leap_seconds(std::chrono::sys_days{std::chrono::July / 1 / 1981} + 1ns, 10s);

  test_leap_seconds(std::chrono::sys_days{std::chrono::July / 1 / 1982} - 1ns, 10s);
  test_leap_seconds(std::chrono::sys_days{std::chrono::July / 1 / 1982}, 11s);
  test_leap_seconds(std::chrono::sys_days{std::chrono::July / 1 / 1982} + 1ns, 11s);

  test_leap_seconds(std::chrono::sys_days{std::chrono::July / 1 / 1983} - 1ns, 11s);
  test_leap_seconds(std::chrono::sys_days{std::chrono::July / 1 / 1983}, 12s);
  test_leap_seconds(std::chrono::sys_days{std::chrono::July / 1 / 1983} + 1ns, 12s);

  test_leap_seconds(std::chrono::sys_days{std::chrono::July / 1 / 1985} - 1ns, 12s);
  test_leap_seconds(std::chrono::sys_days{std::chrono::July / 1 / 1985}, 13s);
  test_leap_seconds(std::chrono::sys_days{std::chrono::July / 1 / 1985} + 1ns, 13s);

  test_leap_seconds(std::chrono::sys_days{std::chrono::January / 1 / 1988} - 1ns, 13s);
  test_leap_seconds(std::chrono::sys_days{std::chrono::January / 1 / 1988}, 14s);
  test_leap_seconds(std::chrono::sys_days{std::chrono::January / 1 / 1988} + 1ns, 14s);

  test_leap_seconds(std::chrono::sys_days{std::chrono::January / 1 / 1990} - 1ns, 14s);
  test_leap_seconds(std::chrono::sys_days{std::chrono::January / 1 / 1990}, 15s);
  test_leap_seconds(std::chrono::sys_days{std::chrono::January / 1 / 1990} + 1ns, 15s);

  test_leap_seconds(std::chrono::sys_days{std::chrono::January / 1 / 1991} - 1ns, 15s);
  test_leap_seconds(std::chrono::sys_days{std::chrono::January / 1 / 1991}, 16s);
  test_leap_seconds(std::chrono::sys_days{std::chrono::January / 1 / 1991} + 1ns, 16s);

  test_leap_seconds(std::chrono::sys_days{std::chrono::July / 1 / 1992} - 1ns, 16s);
  test_leap_seconds(std::chrono::sys_days{std::chrono::July / 1 / 1992}, 17s);
  test_leap_seconds(std::chrono::sys_days{std::chrono::July / 1 / 1992} + 1ns, 17s);

  test_leap_seconds(std::chrono::sys_days{std::chrono::July / 1 / 1993} - 1ns, 17s);
  test_leap_seconds(std::chrono::sys_days{std::chrono::July / 1 / 1993}, 18s);
  test_leap_seconds(std::chrono::sys_days{std::chrono::July / 1 / 1993} + 1ns, 18s);

  test_leap_seconds(std::chrono::sys_days{std::chrono::July / 1 / 1994} - 1ns, 18s);
  test_leap_seconds(std::chrono::sys_days{std::chrono::July / 1 / 1994}, 19s);
  test_leap_seconds(std::chrono::sys_days{std::chrono::July / 1 / 1994} + 1ns, 19s);

  test_leap_seconds(std::chrono::sys_days{std::chrono::January / 1 / 1996} - 1ns, 19s);
  test_leap_seconds(std::chrono::sys_days{std::chrono::January / 1 / 1996}, 20s);
  test_leap_seconds(std::chrono::sys_days{std::chrono::January / 1 / 1996} + 1ns, 20s);

  test_leap_seconds(std::chrono::sys_days{std::chrono::July / 1 / 1997} - 1ns, 20s);
  test_leap_seconds(std::chrono::sys_days{std::chrono::July / 1 / 1997}, 21s);
  test_leap_seconds(std::chrono::sys_days{std::chrono::July / 1 / 1997} + 1ns, 21s);

  test_leap_seconds(std::chrono::sys_days{std::chrono::January / 1 / 1999} - 1ns, 21s);
  test_leap_seconds(std::chrono::sys_days{std::chrono::January / 1 / 1999}, 22s);
  test_leap_seconds(std::chrono::sys_days{std::chrono::January / 1 / 1999} + 1ns, 22s);

  test_leap_seconds(std::chrono::sys_days{std::chrono::January / 1 / 2006} - 1ns, 22s);
  test_leap_seconds(std::chrono::sys_days{std::chrono::January / 1 / 2006}, 23s);
  test_leap_seconds(std::chrono::sys_days{std::chrono::January / 1 / 2006} + 1ns, 23s);

  test_leap_seconds(std::chrono::sys_days{std::chrono::January / 1 / 2009} - 1ns, 23s);
  test_leap_seconds(std::chrono::sys_days{std::chrono::January / 1 / 2009}, 24s);
  test_leap_seconds(std::chrono::sys_days{std::chrono::January / 1 / 2009} + 1ns, 24s);

  test_leap_seconds(std::chrono::sys_days{std::chrono::July / 1 / 2012} - 1ns, 24s);
  test_leap_seconds(std::chrono::sys_days{std::chrono::July / 1 / 2012}, 25s);
  test_leap_seconds(std::chrono::sys_days{std::chrono::July / 1 / 2012} + 1ns, 25s);

  test_leap_seconds(std::chrono::sys_days{std::chrono::July / 1 / 2015} - 1ns, 25s);
  test_leap_seconds(std::chrono::sys_days{std::chrono::July / 1 / 2015}, 26s);
  test_leap_seconds(std::chrono::sys_days{std::chrono::July / 1 / 2015} + 1ns, 26s);

  test_leap_seconds(std::chrono::sys_days{std::chrono::January / 1 / 2017} - 1ns, 26s);
  test_leap_seconds(std::chrono::sys_days{std::chrono::January / 1 / 2017}, 27s);
  test_leap_seconds(std::chrono::sys_days{std::chrono::January / 1 / 2017} + 1ns, 27s);

  // This validates status when the tests were written.
  // It's not possible to test the future; there might be additional leap
  // seconds in the future.
  test_leap_seconds(std::chrono::sys_days{std::chrono::January / 1 / 2024} - 1ns, 27s);
  test_leap_seconds(std::chrono::sys_days{std::chrono::January / 1 / 2024}, 27s);
  test_leap_seconds(std::chrono::sys_days{std::chrono::January / 1 / 2024} + 1ns, 27s);
}

// Tests whether the return type is the expected type.
static void test_return_type() {
  namespace cr = std::chrono;
  using namespace std::literals::chrono_literals;

  {
    [[maybe_unused]] std::same_as<cr::utc_time<cr::nanoseconds>> decltype(auto) _ =
        cr::utc_clock::from_sys(cr::sys_time<cr::nanoseconds>{0ns});
  }
  {
    [[maybe_unused]] std::same_as<cr::utc_time<cr::microseconds>> decltype(auto) _ =
        cr::utc_clock::from_sys(cr::sys_time<cr::microseconds>{0us});
  }
  {
    [[maybe_unused]] std::same_as<cr::utc_time<cr::milliseconds>> decltype(auto) _ =
        cr::utc_clock::from_sys(cr::sys_time<cr::milliseconds>{0ms});
  }

  {
    [[maybe_unused]] std::same_as<cr::utc_time<cr::seconds>> decltype(auto) _ =
        cr::utc_clock::from_sys(cr::sys_time<cr::seconds>{cr::seconds{0}});
  }

  {
    [[maybe_unused]] std::same_as<cr::utc_time<cr::seconds>> decltype(auto) _ =
        cr::utc_clock::from_sys(cr::sys_time<cr::minutes>{cr::minutes{0}});
  }
  {
    [[maybe_unused]] std::same_as<cr::utc_time<cr::seconds>> decltype(auto) _ =
        cr::utc_clock::from_sys(cr::sys_time<cr::hours>{cr::hours{0}});
  }
  {
    [[maybe_unused]] std::same_as<cr::utc_time<cr::seconds>> decltype(auto) _ =
        cr::utc_clock::from_sys(cr::sys_time<cr::days>{cr::days{0}});
  }
  {
    [[maybe_unused]] std::same_as<cr::utc_time<cr::seconds>> decltype(auto) _ =
        cr::utc_clock::from_sys(cr::sys_time<cr::weeks>{cr::weeks{0}});
  }
  {
    [[maybe_unused]] std::same_as<cr::utc_time<cr::seconds>> decltype(auto) _ =
        cr::utc_clock::from_sys(cr::sys_time<cr::months>{cr::months{0}});
  }
  {
    [[maybe_unused]] std::same_as<cr::utc_time<cr::seconds>> decltype(auto) _ =
        cr::utc_clock::from_sys(cr::sys_time<cr::years>{cr::years{0}});
  }
}

int main(int, const char**) {
  test_example_standard();
  test_transitions();
  test_return_type();

  return 0;
}
