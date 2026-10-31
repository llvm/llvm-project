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

// This tests the conversions that involve utc_clock.

#include <cassert>
#include <chrono>
#include <concepts>
#include <ratio>

using namespace std::literals::chrono_literals;
namespace cr = std::chrono;

template <class Clock>
struct ClockBase {
  using rep                       = long long;
  using period                    = std::milli;
  using duration                  = cr::duration<rep, period>;
  using time_point                = cr::time_point<Clock, duration>;
  static constexpr bool is_steady = false;

  static time_point now();
};

// A clock that is one hour ahead of system_clock.
struct SysClock : ClockBase<SysClock> {
  template <class Duration>
  static cr::sys_time<Duration> to_sys(const cr::time_point<SysClock, Duration>& t) {
    return cr::sys_time<Duration>{t.time_since_epoch() - 1h};
  }

  template <class Duration>
  static cr::time_point<SysClock, Duration> from_sys(const cr::sys_time<Duration>& t) {
    return cr::time_point<SysClock, Duration>{t.time_since_epoch() + 1h};
  }
};

// A clock that is one hour ahead of utc_clock.
struct UtcClock : ClockBase<UtcClock> {
  template <class Duration>
  static cr::utc_time<Duration> to_utc(const cr::time_point<UtcClock, Duration>& t) {
    return cr::utc_time<Duration>{t.time_since_epoch() - 1h};
  }

  template <class Duration>
  static cr::time_point<UtcClock, Duration> from_utc(const cr::utc_time<Duration>& t) {
    return cr::time_point<UtcClock, Duration>{t.time_since_epoch() + 1h};
  }
};

// A clock that has the same time since epoch as both system_clock and
// utc_clock. Since these two clocks differ by the number of leap seconds, the
// result of a clock_cast shows which conversion was used.
struct BothClock : ClockBase<BothClock> {
  template <class Duration>
  static cr::sys_time<Duration> to_sys(const cr::time_point<BothClock, Duration>& t) {
    return cr::sys_time<Duration>{t.time_since_epoch()};
  }

  template <class Duration>
  static cr::time_point<BothClock, Duration> from_sys(const cr::sys_time<Duration>& t) {
    return cr::time_point<BothClock, Duration>{t.time_since_epoch()};
  }

  template <class Duration>
  static cr::utc_time<Duration> to_utc(const cr::time_point<BothClock, Duration>& t) {
    return cr::utc_time<Duration>{t.time_since_epoch()};
  }

  template <class Duration>
  static cr::time_point<BothClock, Duration> from_utc(const cr::utc_time<Duration>& t) {
    return cr::time_point<BothClock, Duration>{t.time_since_epoch()};
  }
};

// Clocks without to_utc/from_utc; they are convertible to and from utc_clock
// by user-provided specializations of clock_time_conversion. Like UtcClock,
// they are one hour ahead of utc_clock.
struct SpecializedToUtcClock : ClockBase<SpecializedToUtcClock> {};
struct SpecializedFromUtcClock : ClockBase<SpecializedFromUtcClock> {};

template <>
struct std::chrono::clock_time_conversion<std::chrono::utc_clock, SpecializedToUtcClock> {
  template <class Duration>
  cr::utc_time<Duration> operator()(const cr::time_point<SpecializedToUtcClock, Duration>& t) const {
    return cr::utc_time<Duration>{t.time_since_epoch() - 1h};
  }
};

template <>
struct std::chrono::clock_time_conversion<SpecializedFromUtcClock, std::chrono::utc_clock> {
  template <class Duration>
  cr::time_point<SpecializedFromUtcClock, Duration> operator()(const cr::utc_time<Duration>& t) const {
    return cr::time_point<SpecializedFromUtcClock, Duration>{t.time_since_epoch() + 1h};
  }
};

// A clock without any conversions.
struct UnrelatedClock : ClockBase<UnrelatedClock> {};

template <class DestClock, class SourceClock, class Duration = typename SourceClock::duration>
concept HasClockCast = requires(const cr::time_point<SourceClock, Duration>& t) { cr::clock_cast<DestClock>(t); };

static_assert(HasClockCast<cr::utc_clock, cr::utc_clock>);
static_assert(HasClockCast<cr::tai_clock, cr::gps_clock>);
static_assert(HasClockCast<cr::file_clock, cr::tai_clock>);
static_assert(HasClockCast<UtcClock, SysClock>);
static_assert(HasClockCast<SysClock, UtcClock>);
static_assert(!HasClockCast<UnrelatedClock, cr::utc_clock>);
static_assert(!HasClockCast<cr::utc_clock, UnrelatedClock>);
static_assert(!HasClockCast<UnrelatedClock, cr::tai_clock>);
static_assert(!HasClockCast<cr::gps_clock, UnrelatedClock>);

template <class DestClock, class ExpectedDuration, class SourceClock, class Duration>
void test(const cr::time_point<SourceClock, Duration>& t, const cr::time_point<DestClock, ExpectedDuration>& expected) {
  std::same_as<cr::time_point<DestClock, ExpectedDuration>> decltype(auto) result = cr::clock_cast<DestClock>(t);
  assert(result == expected);
}

// The time points at sys, using the standard clocks and the user-defined clocks.
template <class Duration>
struct TimePoints {
  cr::sys_time<Duration> sys;
  cr::utc_time<Duration> utc;
  cr::tai_time<Duration> tai;
  cr::gps_time<Duration> gps;
  cr::file_time<Duration> file;
  cr::time_point<SysClock, Duration> user_sys;
  cr::time_point<UtcClock, Duration> user_utc;

  // leap_seconds is the number of leap seconds inserted between 1970 and sys.
  TimePoints(cr::sys_time<Duration> s, cr::seconds leap_seconds)
      : sys{s},
        utc{s.time_since_epoch() + leap_seconds},
        // [time.clock.tai.overview]/1: 1958-01-01 00:00:00 TAI is equivalent
        // to 1957-12-31 23:59:50 UTC.
        tai{s - cr::sys_days{cr::January / 1 / 1958} + leap_seconds + 10s},
        // [time.clock.gps.overview]/1: 1980-01-06 00:00:00 UTC is equivalent
        // to 1980-01-06 00:00:00 GPS, and GPS does not insert the 9 leap
        // seconds inserted by UTC until then.
        gps{s - cr::sys_days{cr::January / 6 / 1980} + leap_seconds - 9s},
        file{cr::file_clock::from_sys(s)},
        user_sys{s.time_since_epoch() + 1h},
        user_utc{utc.time_since_epoch() + 1h} {}
};

template <class DestClock, class Duration>
void test_from_all(const TimePoints<Duration>& tp, const cr::time_point<DestClock, Duration>& expected) {
  test<DestClock>(tp.sys, expected);
  test<DestClock>(tp.utc, expected);
  test<DestClock>(tp.tai, expected);
  test<DestClock>(tp.gps, expected);
  test<DestClock>(tp.file, expected);
  test<DestClock>(tp.user_sys, expected);
  test<DestClock>(tp.user_utc, expected);
}

template <class Duration>
void test_all_pairs(cr::sys_time<Duration> sys, cr::seconds leap_seconds) {
  const TimePoints<Duration> tp{sys, leap_seconds};

  test_from_all(tp, tp.sys);
  test_from_all(tp, tp.utc);
  test_from_all(tp, tp.tai);
  test_from_all(tp, tp.gps);
  test_from_all(tp, tp.file);
  test_from_all(tp, tp.user_sys);
  test_from_all(tp, tp.user_utc);
}

template <class Duration>
void test_all_pairs() {
  // Before the first leap second.
  test_all_pairs(cr::sys_time<Duration>{cr::sys_days{cr::January / 1 / 1971}}, 0s);
  // [time.clock.tai.overview]/1: by 2000-01-01 there had been 22 positive
  // and 0 negative leap seconds inserted.
  test_all_pairs(cr::sys_time<Duration>{cr::sys_days{cr::January / 1 / 2000}}, 22s);
  // The last leap second was inserted at 2016-12-31 23:59:60.
  test_all_pairs(cr::sys_time<Duration>{cr::sys_days{cr::January / 1 / 2017}} - 1s, 26s);
  test_all_pairs(cr::sys_time<Duration>{cr::sys_days{cr::January / 1 / 2017}}, 27s);
  test_all_pairs(cr::sys_time<Duration>{cr::sys_days{cr::January / 1 / 2017}} + 1234s, 27s);
}

void test_leap_second() {
  // The leap second 2016-12-31 23:59:60 UTC has no sys_time representation;
  // converting it to sys_time gives the last representable value prior to it.
  const cr::sys_seconds sys_2017 = cr::sys_days{cr::January / 1 / 2017};
  const cr::utc_seconds leap     = cr::utc_seconds{sys_2017.time_since_epoch() + 26s};
  const TimePoints<cr::seconds> before{sys_2017 - 1s, 26s};
  const TimePoints<cr::seconds> after{sys_2017, 27s};

  assert(cr::get_leap_second_info(leap).is_leap_second);

  test<cr::utc_clock>(before.utc, leap - 1s);
  test<cr::utc_clock>(after.utc, leap + 1s);
  test<cr::tai_clock>(leap, before.tai + 1s);
  test<cr::gps_clock>(leap, before.gps + 1s);
  test<UtcClock>(leap, before.user_utc + 1s);

  test<cr::system_clock>(leap, before.sys);
  test<cr::system_clock>(before.tai + 1s, before.sys);
  test<cr::system_clock>(before.gps + 1s, before.sys);
  test<cr::system_clock>(before.user_utc + 1s, before.sys);
  // Through utc_clock and then system_clock.
  test<cr::file_clock>(before.tai + 1s, before.file);
  test<SysClock>(before.gps + 1s, before.user_sys);

  {
    const cr::utc_time<cr::milliseconds> leap_ms = leap + 500ms;
    test<cr::system_clock>(leap_ms, cr::sys_time<cr::milliseconds>{sys_2017 - 1ms});
    test<cr::file_clock>(leap_ms, cr::file_clock::from_sys(cr::sys_time<cr::milliseconds>{sys_2017 - 1ms}));
  }
}

void test_result_duration() {
  // The duration of the result is the one returned by the conversions.
  const cr::sys_days sys = cr::sys_days{cr::January / 1 / 2000};
  const TimePoints<cr::seconds> tp{sys, 22s};

  test<cr::system_clock>(sys, sys);
  test<cr::file_clock>(sys, cr::file_clock::from_sys(sys));
  test<cr::utc_clock>(sys, tp.utc);
  test<cr::tai_clock>(sys, tp.tai);
  test<cr::gps_clock>(sys, tp.gps);

  test<cr::utc_clock>(cr::tai_time<cr::days>{cr::days{1}}, cr::utc_seconds{cr::days{1} - 378'691'210s});
  test<cr::system_clock>(cr::time_point<UtcClock, cr::hours>{1h}, cr::sys_seconds{0s});
}

void test_path_selection() {
  const cr::sys_time<cr::milliseconds> sys = cr::sys_days{cr::January / 1 / 2017};
  const TimePoints<cr::milliseconds> tp{sys, 27s};
  const cr::time_point<BothClock, cr::milliseconds> both{sys.time_since_epoch()};
  const cr::time_point<BothClock, cr::milliseconds> both_utc{tp.utc.time_since_epoch()};

  // A direct conversion is preferred over conversions through other clocks.
  // For the identity conversion, both conversions through system_clock and
  // utc_clock would be valid as well.
  test<BothClock>(both, both);
  test<cr::system_clock>(both, sys);
  test<BothClock>(sys, both);
  test<cr::utc_clock>(both_utc, tp.utc);
  test<BothClock>(tp.utc, both_utc);

  // Conversions with two calls to clock_time_conversion::operator() are
  // preferred over conversions with three calls.
  //
  // BothClock <- system_clock <- SysClock is preferred over
  // BothClock <- utc_clock <- system_clock <- SysClock
  test<BothClock>(tp.user_sys, both);
  test<BothClock>(tp.file, both);
  // BothClock <- utc_clock <- UtcClock is preferred over
  // BothClock <- system_clock <- utc_clock <- UtcClock
  test<BothClock>(tp.user_utc, both_utc);
  test<BothClock>(tp.tai, both_utc);
  test<BothClock>(tp.gps, both_utc);

  // The conversions to clocks that don't have to_sys/from_sys or
  // to_utc/from_utc are unambiguous.
  test<cr::file_clock>(both, tp.file);
  test<SysClock>(both, tp.user_sys);
  test<cr::tai_clock>(both_utc, tp.tai);
  test<UtcClock>(both_utc, tp.user_utc);
}

void test_user_specializations() {
  // User-provided specializations of clock_time_conversion are used for the
  // conversions through utc_clock.
  const cr::sys_time<cr::milliseconds> sys = cr::sys_days{cr::January / 1 / 2017} + 1234ms;
  const TimePoints<cr::milliseconds> tp{sys, 27s};
  const cr::time_point<SpecializedToUtcClock, cr::milliseconds> to_utc{tp.user_utc.time_since_epoch()};
  const cr::time_point<SpecializedFromUtcClock, cr::milliseconds> from_utc{tp.user_utc.time_since_epoch()};

  test<cr::utc_clock>(to_utc, tp.utc);
  test<cr::tai_clock>(to_utc, tp.tai);
  test<cr::system_clock>(to_utc, tp.sys);
  test<cr::file_clock>(to_utc, tp.file);
  test<SysClock>(to_utc, tp.user_sys);
  test<SpecializedFromUtcClock>(to_utc, from_utc);

  test<SpecializedFromUtcClock>(tp.utc, from_utc);
  test<SpecializedFromUtcClock>(tp.gps, from_utc);
  test<SpecializedFromUtcClock>(tp.sys, from_utc);
  test<SpecializedFromUtcClock>(tp.file, from_utc);
  test<SpecializedFromUtcClock>(tp.user_sys, from_utc);
}

int main(int, char**) {
  test_all_pairs<cr::seconds>();
  test_all_pairs<cr::milliseconds>();
  test_all_pairs<cr::nanoseconds>();
  test_leap_second();
  test_result_duration();
  test_path_selection();
  test_user_specializations();

  return 0;
}
