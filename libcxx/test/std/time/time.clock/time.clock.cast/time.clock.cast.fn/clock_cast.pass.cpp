//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++20

// <chrono>

// template<class DestClock, class SourceClock, class Duration>
//   auto clock_cast(const time_point<SourceClock, Duration>& t);

// This tests the conversions that don't involve utc_clock.

#include <cassert>
#include <chrono>
#include <concepts>
#include <ratio>

using namespace std::literals::chrono_literals;

// A clock that is Offset ahead of system_clock.
template <int Offset>
struct OffsetClock {
  using rep                       = long long;
  using period                    = std::milli;
  using duration                  = std::chrono::duration<rep, period>;
  using time_point                = std::chrono::time_point<OffsetClock>;
  static constexpr bool is_steady = false;

  static time_point now();

  template <class Duration>
  static std::chrono::sys_time<Duration> to_sys(const std::chrono::time_point<OffsetClock, Duration>& t) {
    return std::chrono::sys_time<Duration>{t.time_since_epoch() - std::chrono::hours{Offset}};
  }

  template <class Duration>
  static std::chrono::time_point<OffsetClock, Duration> from_sys(const std::chrono::sys_time<Duration>& t) {
    return std::chrono::time_point<OffsetClock, Duration>{t.time_since_epoch() + std::chrono::hours{Offset}};
  }
};

using OneHourClock  = OffsetClock<1>;
using TwoHoursClock = OffsetClock<2>;

// A clock with a direct conversion to OneHourClock, as well as a conversion
// through system_clock. The direct conversion is preferred.
struct DirectClock {
  using rep                       = long long;
  using period                    = std::milli;
  using duration                  = std::chrono::duration<rep, period>;
  using time_point                = std::chrono::time_point<DirectClock>;
  static constexpr bool is_steady = false;

  static time_point now();

  template <class Duration>
  static std::chrono::sys_time<Duration> to_sys(const std::chrono::time_point<DirectClock, Duration>& t) {
    return std::chrono::sys_time<Duration>{t.time_since_epoch()};
  }
};

// A clock without to_sys/from_sys; it is convertible to system_clock by a
// user-provided specialization of clock_time_conversion.
struct SpecializedClock {
  using rep                       = long long;
  using period                    = std::milli;
  using duration                  = std::chrono::duration<rep, period>;
  using time_point                = std::chrono::time_point<SpecializedClock>;
  static constexpr bool is_steady = false;

  static time_point now();
};

// A clock that is not convertible to any other clock.
struct UnrelatedClock {
  using rep                       = long long;
  using period                    = std::milli;
  using duration                  = std::chrono::duration<rep, period>;
  using time_point                = std::chrono::time_point<UnrelatedClock>;
  static constexpr bool is_steady = false;

  static time_point now();
};

template <>
struct std::chrono::clock_time_conversion<OneHourClock, DirectClock> {
  template <class Duration>
  OneHourClock::time_point operator()(const std::chrono::time_point<DirectClock, Duration>&) const {
    return OneHourClock::time_point{42ms};
  }
};

template <>
struct std::chrono::clock_time_conversion<std::chrono::system_clock, SpecializedClock> {
  template <class Duration>
  std::chrono::sys_time<Duration> operator()(const std::chrono::time_point<SpecializedClock, Duration>& t) const {
    return std::chrono::sys_time<Duration>{t.time_since_epoch() - 3h};
  }
};

template <class DestClock, class SourceClock, class Duration = typename SourceClock::duration>
concept HasClockCast =
    requires(const std::chrono::time_point<SourceClock, Duration>& t) { std::chrono::clock_cast<DestClock>(t); };

static_assert(HasClockCast<std::chrono::system_clock, std::chrono::system_clock>);
static_assert(HasClockCast<std::chrono::file_clock, std::chrono::system_clock>);
static_assert(HasClockCast<std::chrono::file_clock, OneHourClock>);
static_assert(HasClockCast<std::chrono::file_clock, SpecializedClock>);
static_assert(HasClockCast<UnrelatedClock, UnrelatedClock>);

static_assert(!HasClockCast<UnrelatedClock, std::chrono::file_clock>);
static_assert(!HasClockCast<std::chrono::file_clock, UnrelatedClock>);
static_assert(!HasClockCast<UnrelatedClock, std::chrono::system_clock>);
static_assert(!HasClockCast<std::chrono::system_clock, UnrelatedClock>);
static_assert(!HasClockCast<UnrelatedClock, OneHourClock>);
static_assert(!HasClockCast<DirectClock, std::chrono::system_clock>); // DirectClock has no from_sys
static_assert(!HasClockCast<SpecializedClock, std::chrono::system_clock>);

template <class DestClock, class ExpectedDuration, class SourceClock, class Duration>
void test(const std::chrono::time_point<SourceClock, Duration>& t,
          const std::chrono::time_point<DestClock, ExpectedDuration>& expected) {
  std::same_as<std::chrono::time_point<DestClock, ExpectedDuration>> decltype(auto) result =
      std::chrono::clock_cast<DestClock>(t);
  assert(result == expected);
}

template <class Duration>
void test_identity(Duration d) {
  test<std::chrono::system_clock>(std::chrono::sys_time<Duration>{d}, std::chrono::sys_time<Duration>{d});
  test<std::chrono::file_clock>(std::chrono::file_time<Duration>{d}, std::chrono::file_time<Duration>{d});
  test<UnrelatedClock>(
      std::chrono::time_point<UnrelatedClock, Duration>{d}, std::chrono::time_point<UnrelatedClock, Duration>{d});
  test<OneHourClock>(
      std::chrono::time_point<OneHourClock, Duration>{d}, std::chrono::time_point<OneHourClock, Duration>{d});
}

template <class Duration>
void test_file_clock(Duration d) {
  const std::chrono::sys_time<Duration> sys{d};
  const auto file = std::chrono::file_clock::from_sys(sys);

  test<std::chrono::file_clock>(sys, file);
  test<std::chrono::system_clock>(file, sys);
}

template <class Duration>
void test_user_clocks(Duration d) {
  using std::chrono::time_point;

  const std::chrono::sys_time<Duration> sys{d};
  const auto file = std::chrono::file_clock::from_sys(sys);
  const time_point<OneHourClock, Duration> one{sys.time_since_epoch() + 1h};
  const time_point<TwoHoursClock, Duration> two{sys.time_since_epoch() + 2h};

  // Directly through to_sys/from_sys.
  test<std::chrono::system_clock>(one, sys);
  test<OneHourClock>(sys, one);

  // Through system_clock.
  test<OneHourClock>(two, one);
  test<TwoHoursClock>(one, two);
  test<OneHourClock>(file, one);
  test<std::chrono::file_clock>(one, file);

  // A user-provided specialization of clock_time_conversion is used for the
  // conversion through system_clock.
  const time_point<SpecializedClock, Duration> specialized{sys.time_since_epoch() + 3h};
  test<std::chrono::system_clock>(specialized, sys);
  test<std::chrono::file_clock>(specialized, file);
  test<OneHourClock>(specialized, one);

  // The direct conversion is preferred over the conversion through system_clock.
  const time_point<DirectClock, Duration> direct{sys.time_since_epoch()};
  test<std::chrono::system_clock>(direct, sys);
  test<TwoHoursClock>(direct, two);
  test<OneHourClock>(direct, OneHourClock::time_point{42ms});
}

template <class Duration>
void test_all(Duration d) {
  test_identity(d);
  test_file_clock(d);
  test_user_clocks(d);
}

int main(int, char**) {
  test_all(0ms);
  test_all(-1'234'567ms);
  test_all(1'234'567'890'123ms);
  test_all(std::chrono::minutes{42});

  test_identity(std::chrono::days{20'000});
  test_file_clock(std::chrono::days{20'000});
  test_identity(std::chrono::nanoseconds{-1});
  test_identity(std::chrono::duration<double, std::milli>{1.5});
  test_file_clock(1'234'567'890ns);

  {
    // The duration of the result is the one returned by the conversions.
    using Duration = std::chrono::duration<short, std::ratio<7>>;
    test<std::chrono::file_clock>(std::chrono::sys_time<Duration>{Duration{3}},
                                  std::chrono::file_clock::from_sys(std::chrono::sys_time<Duration>{Duration{3}}));
  }

  return 0;
}
