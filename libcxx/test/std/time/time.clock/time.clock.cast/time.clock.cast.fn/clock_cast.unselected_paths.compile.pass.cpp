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

// template<class DestClock, class SourceClock, class Duration>
//   auto clock_cast(const time_point<SourceClock, Duration>& t);

// Determining which conversions are possible must not instantiate the call
// operators of clock_time_conversion. Otherwise clocks whose conversions
// violate the Mandates: of a conversion that is not selected cause a hard error.

#include <chrono>
#include <concepts>
#include <ratio>

template <class Clock>
struct ClockBase {
  using rep                       = long long;
  using period                    = std::milli;
  using duration                  = std::chrono::duration<rep, period>;
  using time_point                = std::chrono::time_point<Clock, duration>;
  static constexpr bool is_steady = false;

  static time_point now();
};

// to_utc does not return a utc_time.
struct BadToUtcClock : ClockBase<BadToUtcClock> {
  template <class Duration>
  static std::chrono::sys_time<Duration> to_sys(const std::chrono::time_point<BadToUtcClock, Duration>&);
  template <class Duration>
  static std::chrono::sys_time<Duration> to_utc(const std::chrono::time_point<BadToUtcClock, Duration>&);
};

// from_utc does not return a time_point of BadFromUtcClock.
struct BadFromUtcClock : ClockBase<BadFromUtcClock> {
  template <class Duration>
  static std::chrono::time_point<BadFromUtcClock, Duration> from_sys(const std::chrono::sys_time<Duration>&);
  template <class Duration>
  static int from_utc(const std::chrono::utc_time<Duration>&);
};

void test() {
  namespace cr = std::chrono;

  // The conversion through system_clock is selected. Converting through
  // utc_clock isn't possible, since file_clock has no from_utc and
  // clock_time_conversion<system_clock, utc_clock> needs a utc_time.
  {
    std::same_as<cr::file_time<cr::milliseconds>> decltype(auto) r =
        cr::clock_cast<cr::file_clock>(BadToUtcClock::time_point{});
    (void)r;
  }

  // The conversion through system_clock is selected over the one through
  // system_clock and then utc_clock.
  {
    std::same_as<BadFromUtcClock::time_point> decltype(auto) r =
        cr::clock_cast<BadFromUtcClock>(cr::file_time<cr::milliseconds>{});
    (void)r;
  }
}
