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
//
// Mandates: Among the well-formed expressions [...], there is at least one
// best expression that involves the fewest calls to operator() of
// clock_time_conversion specializations, and that expression is unique.

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

// A clock that can be converted through both system_clock and utc_clock.
template <int N>
struct BothClock : ClockBase<BothClock<N>> {
  template <class Duration>
  static std::chrono::sys_time<Duration> to_sys(const std::chrono::time_point<BothClock, Duration>&);
  template <class Duration>
  static std::chrono::time_point<BothClock, Duration> from_sys(const std::chrono::sys_time<Duration>&);
  template <class Duration>
  static std::chrono::utc_time<Duration> to_utc(const std::chrono::time_point<BothClock, Duration>&);
  template <class Duration>
  static std::chrono::time_point<BothClock, Duration> from_utc(const std::chrono::utc_time<Duration>&);
};

using TwoSeconds = std::chrono::duration<long long, std::ratio<2>>;

// The conversions of these clocks only work in the combinations that go
// through both system_clock and utc_clock: the conversions between these two
// clocks change the duration from TwoSeconds to seconds.
struct TwoSecondsSourceClock : ClockBase<TwoSecondsSourceClock> {
  template <class Duration>
  static std::chrono::sys_time<TwoSeconds> to_sys(const std::chrono::time_point<TwoSecondsSourceClock, Duration>&);
  template <class Duration>
  static std::chrono::utc_time<TwoSeconds> to_utc(const std::chrono::time_point<TwoSecondsSourceClock, Duration>&);
};

struct SecondsDestClock : ClockBase<SecondsDestClock> {
  template <std::same_as<std::chrono::sys_seconds> TimePoint>
  static std::chrono::time_point<SecondsDestClock, std::chrono::seconds> from_sys(const TimePoint&);
  template <std::same_as<std::chrono::utc_seconds> TimePoint>
  static std::chrono::time_point<SecondsDestClock, std::chrono::seconds> from_utc(const TimePoint&);
};

void test() {
  namespace cr = std::chrono;

  // Both BothClock<1> <- system_clock <- BothClock<0> and
  // BothClock<1> <- utc_clock <- BothClock<0> are valid.
  // expected-error-re@*:* {{static assertion failed{{.*}}clock_cast is ambiguous: SourceClock can be converted to DestClock both through system_clock and through utc_clock}}
  (void)cr::clock_cast<BothClock<1>>(BothClock<0>::time_point{});

  // Both SecondsDestClock <- utc_clock <- system_clock <- TwoSecondsSourceClock and
  // SecondsDestClock <- system_clock <- utc_clock <- TwoSecondsSourceClock are valid.
  // expected-error-re@*:* {{static assertion failed{{.*}}clock_cast is ambiguous: SourceClock can be converted to DestClock both through system_clock and then utc_clock, and through utc_clock and then system_clock}}
  (void)cr::clock_cast<SecondsDestClock>(TwoSecondsSourceClock::time_point{});
}
