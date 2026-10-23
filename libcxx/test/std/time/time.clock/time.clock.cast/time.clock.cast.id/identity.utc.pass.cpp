//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++20
// UNSUPPORTED: no-filesystem, no-localization, no-tzdb

// XFAIL: libcpp-has-no-experimental-tzdb
// XFAIL: availability-tzdb-missing

// <chrono>

// template<>
// struct clock_time_conversion<utc_clock, utc_clock> {
//   template<class Duration>
//     utc_time<Duration>
//       operator()(const utc_time<Duration>& t) const;
// };

#include <cassert>
#include <chrono>
#include <concepts>
#include <ratio>

template <class Clock, class Duration>
void test(Duration d) {
  using TimePoint = std::chrono::time_point<Clock, Duration>;
  const std::chrono::clock_time_conversion<Clock, Clock> conversion{};
  const TimePoint t{d};

  std::same_as<TimePoint> decltype(auto) result = conversion(t);
  assert(result == t);
}

template <class Clock>
void test_durations() {
  using namespace std::literals::chrono_literals;

  test<Clock>(std::chrono::nanoseconds{-1});
  test<Clock>(1'234'567'890ns);
  test<Clock>(42ms);
  test<Clock>(-42s);
  test<Clock>(std::chrono::duration<double, std::milli>{1.5});
  test<Clock>(std::chrono::duration<short, std::ratio<7>>{3});
  test<Clock>(std::chrono::days{20'000});
}

int main(int, char**) {
  test_durations<std::chrono::utc_clock>();
  test_durations<std::chrono::tai_clock>();
  test_durations<std::chrono::gps_clock>();

  return 0;
}
