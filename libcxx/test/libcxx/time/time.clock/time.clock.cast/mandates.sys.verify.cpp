//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++20

// <chrono>

// template<class SourceClock>
// struct clock_time_conversion<system_clock, SourceClock>;
//   Mandates: SourceClock::to_sys(t) returns a sys_time<Duration2> for some type Duration2.
//
// template<class DestClock>
// struct clock_time_conversion<DestClock, system_clock>;
//   Mandates: DestClock::from_sys(t) returns a time_point<DestClock, Duration2> for some type Duration2.

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

struct ToSysReturnsInt : ClockBase<ToSysReturnsInt> {
  template <class Duration>
  static int to_sys(const std::chrono::time_point<ToSysReturnsInt, Duration>&);
};

struct ToSysReturnsWrongClock : ClockBase<ToSysReturnsWrongClock> {
  template <class Duration>
  static std::chrono::time_point<ToSysReturnsWrongClock, Duration>
  to_sys(const std::chrono::time_point<ToSysReturnsWrongClock, Duration>&);
};

struct ToSysReturnsReference : ClockBase<ToSysReturnsReference> {
  template <class Duration>
  static const std::chrono::sys_time<Duration>& to_sys(const std::chrono::time_point<ToSysReturnsReference, Duration>&);
};

struct FromSysReturnsInt : ClockBase<FromSysReturnsInt> {
  template <class Duration>
  static int from_sys(const std::chrono::sys_time<Duration>&);
};

struct FromSysReturnsWrongClock : ClockBase<FromSysReturnsWrongClock> {
  template <class Duration>
  static std::chrono::sys_time<Duration> from_sys(const std::chrono::sys_time<Duration>&);
};

void test() {
  namespace cr = std::chrono;

  // expected-error-re@*:* {{static assertion failed{{.*}}SourceClock::to_sys(t) must return a sys_time}}
  (void)cr::clock_time_conversion<cr::system_clock, ToSysReturnsInt>{}(ToSysReturnsInt::time_point{});

  // expected-error-re@*:* {{static assertion failed{{.*}}SourceClock::to_sys(t) must return a sys_time}}
  (void)cr::clock_time_conversion<cr::system_clock, ToSysReturnsWrongClock>{}(ToSysReturnsWrongClock::time_point{});

  // expected-error-re@*:* {{static assertion failed{{.*}}SourceClock::to_sys(t) must return a sys_time}}
  (void)cr::clock_time_conversion<cr::system_clock, ToSysReturnsReference>{}(ToSysReturnsReference::time_point{});

  // expected-error-re@*:* {{static assertion failed{{.*}}DestClock::from_sys(t) must return a time_point of DestClock}}
  (void)cr::clock_time_conversion<FromSysReturnsInt, cr::system_clock>{}(cr::sys_seconds{});

  // expected-error-re@*:* {{static assertion failed{{.*}}DestClock::from_sys(t) must return a time_point of DestClock}}
  (void)cr::clock_time_conversion<FromSysReturnsWrongClock, cr::system_clock>{}(cr::sys_seconds{});
}
