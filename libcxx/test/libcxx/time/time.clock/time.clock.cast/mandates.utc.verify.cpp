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

// template<class SourceClock>
// struct clock_time_conversion<utc_clock, SourceClock>;
//   Mandates: SourceClock::to_utc(t) returns a utc_time<Duration2> for some type Duration2.
//
// template<class DestClock>
// struct clock_time_conversion<DestClock, utc_clock>;
//   Mandates: DestClock::from_utc(t) returns a time_point<DestClock, Duration2> for some type Duration2.

#include <chrono>
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

struct ToUtcReturnsInt : ClockBase<ToUtcReturnsInt> {
  template <class Duration>
  static int to_utc(const std::chrono::time_point<ToUtcReturnsInt, Duration>&);
};

struct ToUtcReturnsSysTime : ClockBase<ToUtcReturnsSysTime> {
  template <class Duration>
  static std::chrono::sys_time<Duration> to_utc(const std::chrono::time_point<ToUtcReturnsSysTime, Duration>&);
};

struct FromUtcReturnsInt : ClockBase<FromUtcReturnsInt> {
  template <class Duration>
  static int from_utc(const std::chrono::utc_time<Duration>&);
};

struct FromUtcReturnsWrongClock : ClockBase<FromUtcReturnsWrongClock> {
  template <class Duration>
  static std::chrono::utc_time<Duration> from_utc(const std::chrono::utc_time<Duration>&);
};

void test() {
  namespace cr = std::chrono;

  // expected-error-re@*:* {{static assertion failed{{.*}}SourceClock::to_utc(t) must return a utc_time}}
  (void)cr::clock_time_conversion<cr::utc_clock, ToUtcReturnsInt>{}(ToUtcReturnsInt::time_point{});

  // expected-error-re@*:* {{static assertion failed{{.*}}SourceClock::to_utc(t) must return a utc_time}}
  (void)cr::clock_time_conversion<cr::utc_clock, ToUtcReturnsSysTime>{}(ToUtcReturnsSysTime::time_point{});

  // expected-error-re@*:* {{static assertion failed{{.*}}DestClock::from_utc(t) must return a time_point of DestClock}}
  (void)cr::clock_time_conversion<FromUtcReturnsInt, cr::utc_clock>{}(cr::utc_seconds{});

  // expected-error-re@*:* {{static assertion failed{{.*}}DestClock::from_utc(t) must return a time_point of DestClock}}
  (void)cr::clock_time_conversion<FromUtcReturnsWrongClock, cr::utc_clock>{}(cr::utc_seconds{});
}
