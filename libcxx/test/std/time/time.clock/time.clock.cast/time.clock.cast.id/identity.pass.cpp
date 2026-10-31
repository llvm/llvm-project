//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++20

// <chrono>

// template<class Clock>
// struct clock_time_conversion<Clock, Clock> {
//   template<class Duration>
//     time_point<Clock, Duration>
//       operator()(const time_point<Clock, Duration>& t) const;
// };
//
// template<>
// struct clock_time_conversion<system_clock, system_clock> {
//   template<class Duration>
//     sys_time<Duration>
//       operator()(const sys_time<Duration>& t) const;
// };

#include <cassert>
#include <chrono>
#include <concepts>
#include <ratio>

struct TestClock {
  using rep                       = long long;
  using period                    = std::ratio<1, 3>;
  using duration                  = std::chrono::duration<rep, period>;
  using time_point                = std::chrono::time_point<TestClock>;
  static constexpr bool is_steady = false;

  static time_point now();
};

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
  test_durations<std::chrono::system_clock>();
  test_durations<std::chrono::file_clock>();
  test_durations<std::chrono::local_t>();
  test_durations<TestClock>();

  return 0;
}
