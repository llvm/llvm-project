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
// struct clock_time_conversion<system_clock, SourceClock> {
//   template<class Duration>
//     auto operator()(const time_point<SourceClock, Duration>& t) const
//       -> decltype(SourceClock::to_sys(t));
// };
//
// template<class DestClock>
// struct clock_time_conversion<DestClock, system_clock> {
//   template<class Duration>
//     auto operator()(const sys_time<Duration>& t) const
//       -> decltype(DestClock::from_sys(t));
// };

#include <cassert>
#include <chrono>
#include <concepts>
#include <ratio>
#include <type_traits>

// A clock that is one hour ahead of system_clock, and whose conversions
// return a coarser duration than the one they are given.
struct OffsetClock {
  using rep                       = long long;
  using period                    = std::milli;
  using duration                  = std::chrono::duration<rep, period>;
  using time_point                = std::chrono::time_point<OffsetClock>;
  static constexpr bool is_steady = false;

  static time_point now();

  template <class Duration>
  static std::chrono::sys_time<std::common_type_t<Duration, std::chrono::seconds>>
  to_sys(const std::chrono::time_point<OffsetClock, Duration>& t) {
    return std::chrono::sys_time<std::common_type_t<Duration, std::chrono::seconds>>{
        t.time_since_epoch() - std::chrono::hours{1}};
  }

  template <class Duration>
  static std::chrono::time_point<OffsetClock, std::common_type_t<Duration, std::chrono::seconds>>
  from_sys(const std::chrono::sys_time<Duration>& t) {
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

  static std::chrono::sys_seconds to_sys(const time_point& t) { return std::chrono::sys_seconds{t.time_since_epoch()}; }
  static time_point from_sys(const std::chrono::sys_seconds& t) { return time_point{t.time_since_epoch()}; }
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

// Constraints: SourceClock::to_sys(t) / DestClock::from_sys(t) is well-formed.
static_assert(HasConversion<std::chrono::system_clock, SecondsOnlyClock, std::chrono::seconds>);
static_assert(HasConversion<SecondsOnlyClock, std::chrono::system_clock, std::chrono::seconds>);
static_assert(HasConversion<std::chrono::system_clock, SecondsOnlyClock, std::chrono::minutes>);
static_assert(HasConversion<SecondsOnlyClock, std::chrono::system_clock, std::chrono::minutes>);
static_assert(!HasConversion<std::chrono::system_clock, SecondsOnlyClock, std::chrono::milliseconds>);
static_assert(!HasConversion<SecondsOnlyClock, std::chrono::system_clock, std::chrono::milliseconds>);
static_assert(!HasConversion<std::chrono::system_clock, UnrelatedClock, std::chrono::seconds>);
static_assert(!HasConversion<UnrelatedClock, std::chrono::system_clock, std::chrono::seconds>);

template <class Clock, class Duration>
void test_to_sys(Duration d) {
  const std::chrono::clock_time_conversion<std::chrono::system_clock, Clock> conversion{};
  const std::chrono::time_point<Clock, Duration> t{d};

  std::same_as<decltype(Clock::to_sys(t))> decltype(auto) result = conversion(t);
  assert(result == Clock::to_sys(t));
}

template <class Clock, class Duration>
void test_from_sys(Duration d) {
  const std::chrono::clock_time_conversion<Clock, std::chrono::system_clock> conversion{};
  const std::chrono::sys_time<Duration> t{d};

  std::same_as<decltype(Clock::from_sys(t))> decltype(auto) result = conversion(t);
  assert(result == Clock::from_sys(t));
}

template <class Clock>
void test_clock() {
  using namespace std::literals::chrono_literals;

  test_to_sys<Clock>(0s);
  test_to_sys<Clock>(-1'234'567'890ns);
  test_to_sys<Clock>(42ms);
  test_to_sys<Clock>(std::chrono::days{20'000});

  test_from_sys<Clock>(0s);
  test_from_sys<Clock>(-1'234'567'890ns);
  test_from_sys<Clock>(42ms);
  test_from_sys<Clock>(std::chrono::days{20'000});
}

int main(int, char**) {
  using namespace std::literals::chrono_literals;

  test_clock<std::chrono::file_clock>();
  test_clock<OffsetClock>();

  {
    // The result type is the one returned by to_sys, not a sys_time with the source duration.
    const std::chrono::time_point<OffsetClock, std::chrono::milliseconds> t{3'600'500ms};
    std::same_as<std::chrono::sys_time<std::chrono::milliseconds>> decltype(auto) result =
        std::chrono::clock_time_conversion<std::chrono::system_clock, OffsetClock>{}(t);
    assert(result.time_since_epoch() == 500ms);

    const std::chrono::sys_time<std::chrono::minutes> s{2min};
    std::same_as<std::chrono::time_point<OffsetClock, std::chrono::seconds>> decltype(auto) result2 =
        std::chrono::clock_time_conversion<OffsetClock, std::chrono::system_clock>{}(s);
    assert(result2.time_since_epoch() == 3'720s);
  }

  {
    const std::chrono::time_point<SecondsOnlyClock, std::chrono::seconds> t{42s};
    std::same_as<std::chrono::sys_seconds> decltype(auto) result =
        std::chrono::clock_time_conversion<std::chrono::system_clock, SecondsOnlyClock>{}(t);
    assert(result.time_since_epoch() == 42s);

    std::same_as<SecondsOnlyClock::time_point> decltype(auto) result2 =
        std::chrono::clock_time_conversion<SecondsOnlyClock, std::chrono::system_clock>{}(
            std::chrono::sys_time<std::chrono::minutes>{1min});
    assert(result2.time_since_epoch() == 60s);
  }

  return 0;
}
