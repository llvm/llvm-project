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
// struct clock_time_conversion<utc_clock, SourceClock> {
//   template<class Duration>
//     auto operator()(const time_point<SourceClock, Duration>& t) const
//       -> decltype(SourceClock::to_utc(t));
// };
//
// template<class DestClock>
// struct clock_time_conversion<DestClock, utc_clock> {
//   template<class Duration>
//     auto operator()(const utc_time<Duration>& t) const
//       -> decltype(DestClock::from_utc(t));
// };

#include <cassert>
#include <chrono>
#include <concepts>
#include <ratio>
#include <type_traits>

// A clock that is one hour ahead of utc_clock, and whose conversions return a
// coarser duration than the one they are given.
struct OffsetClock {
  using rep                       = long long;
  using period                    = std::milli;
  using duration                  = std::chrono::duration<rep, period>;
  using time_point                = std::chrono::time_point<OffsetClock>;
  static constexpr bool is_steady = false;

  static time_point now();

  template <class Duration>
  static std::chrono::utc_time<std::common_type_t<Duration, std::chrono::seconds>>
  to_utc(const std::chrono::time_point<OffsetClock, Duration>& t) {
    return std::chrono::utc_time<std::common_type_t<Duration, std::chrono::seconds>>{
        t.time_since_epoch() - std::chrono::hours{1}};
  }

  template <class Duration>
  static std::chrono::time_point<OffsetClock, std::common_type_t<Duration, std::chrono::seconds>>
  from_utc(const std::chrono::utc_time<Duration>& t) {
    return std::chrono::time_point<OffsetClock, std::common_type_t<Duration, std::chrono::seconds>>{
        t.time_since_epoch() + std::chrono::hours{1}};
  }
};

// A clock whose conversions only accept time points with a seconds duration.
struct SecondsOnlyClock {
  using rep                       = long long;
  using period                    = std::ratio<1>;
  using duration                  = std::chrono::duration<rep, period>;
  using time_point                = std::chrono::time_point<SecondsOnlyClock>;
  static constexpr bool is_steady = false;

  static time_point now();

  static std::chrono::utc_seconds to_utc(const time_point& t) { return std::chrono::utc_seconds{t.time_since_epoch()}; }
  static time_point from_utc(const std::chrono::utc_seconds& t) { return time_point{t.time_since_epoch()}; }
};

// A clock without any conversions.
struct UnrelatedClock {
  using rep                       = long long;
  using period                    = std::ratio<1>;
  using duration                  = std::chrono::duration<rep, period>;
  using time_point                = std::chrono::time_point<UnrelatedClock>;
  static constexpr bool is_steady = false;

  static time_point now();
};

template <class DestClock, class SourceClock, class Duration>
concept HasConversion = requires(const std::chrono::time_point<SourceClock, Duration>& t) {
  std::chrono::clock_time_conversion<DestClock, SourceClock>{}(t);
};

// Constraints: SourceClock::to_utc(t) / DestClock::from_utc(t) is well-formed.
static_assert(HasConversion<std::chrono::utc_clock, SecondsOnlyClock, std::chrono::seconds>);
static_assert(HasConversion<SecondsOnlyClock, std::chrono::utc_clock, std::chrono::seconds>);
static_assert(HasConversion<std::chrono::utc_clock, SecondsOnlyClock, std::chrono::minutes>);
static_assert(HasConversion<SecondsOnlyClock, std::chrono::utc_clock, std::chrono::minutes>);
static_assert(!HasConversion<std::chrono::utc_clock, SecondsOnlyClock, std::chrono::milliseconds>);
static_assert(!HasConversion<SecondsOnlyClock, std::chrono::utc_clock, std::chrono::milliseconds>);
static_assert(!HasConversion<std::chrono::utc_clock, UnrelatedClock, std::chrono::seconds>);
static_assert(!HasConversion<UnrelatedClock, std::chrono::utc_clock, std::chrono::seconds>);
static_assert(!HasConversion<std::chrono::utc_clock, std::chrono::file_clock, std::chrono::seconds>);
static_assert(!HasConversion<std::chrono::file_clock, std::chrono::utc_clock, std::chrono::seconds>);

template <class Clock, class Duration>
void test_to_utc(Duration d) {
  const std::chrono::clock_time_conversion<std::chrono::utc_clock, Clock> conversion{};
  const std::chrono::time_point<Clock, Duration> t{d};

  std::same_as<decltype(Clock::to_utc(t))> decltype(auto) result = conversion(t);
  assert(result == Clock::to_utc(t));
}

template <class Clock, class Duration>
void test_from_utc(Duration d) {
  const std::chrono::clock_time_conversion<Clock, std::chrono::utc_clock> conversion{};
  const std::chrono::utc_time<Duration> t{d};

  std::same_as<decltype(Clock::from_utc(t))> decltype(auto) result = conversion(t);
  assert(result == Clock::from_utc(t));
}

template <class Clock>
void test_clock() {
  using namespace std::literals::chrono_literals;

  test_to_utc<Clock>(0s);
  test_to_utc<Clock>(1'234'567'890ns);
  test_to_utc<Clock>(42ms);
  test_to_utc<Clock>(std::chrono::days{20'000});

  test_from_utc<Clock>(0s);
  test_from_utc<Clock>(1'234'567'890ns);
  test_from_utc<Clock>(42ms);
  test_from_utc<Clock>(std::chrono::days{20'000});
}

int main(int, char**) {
  using namespace std::literals::chrono_literals;

  test_clock<std::chrono::tai_clock>();
  test_clock<std::chrono::gps_clock>();
  test_clock<OffsetClock>();

  {
    // The result type is the one returned by to_utc, not a utc_time with the source duration.
    const std::chrono::time_point<OffsetClock, std::chrono::milliseconds> t{3'600'500ms};
    std::same_as<std::chrono::utc_time<std::chrono::milliseconds>> decltype(auto) result =
        std::chrono::clock_time_conversion<std::chrono::utc_clock, OffsetClock>{}(t);
    assert(result.time_since_epoch() == 500ms);

    const std::chrono::utc_time<std::chrono::minutes> u{2min};
    std::same_as<std::chrono::time_point<OffsetClock, std::chrono::seconds>> decltype(auto) result2 =
        std::chrono::clock_time_conversion<OffsetClock, std::chrono::utc_clock>{}(u);
    assert(result2.time_since_epoch() == 3'720s);
  }

  {
    const std::chrono::time_point<SecondsOnlyClock, std::chrono::seconds> t{42s};
    std::same_as<std::chrono::utc_seconds> decltype(auto) result =
        std::chrono::clock_time_conversion<std::chrono::utc_clock, SecondsOnlyClock>{}(t);
    assert(result.time_since_epoch() == 42s);

    std::same_as<SecondsOnlyClock::time_point> decltype(auto) result2 =
        std::chrono::clock_time_conversion<SecondsOnlyClock, std::chrono::utc_clock>{}(
            std::chrono::utc_time<std::chrono::minutes>{1min});
    assert(result2.time_since_epoch() == 60s);
  }

  return 0;
}
