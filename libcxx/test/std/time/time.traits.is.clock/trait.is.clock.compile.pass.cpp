//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++20

#include <chrono>
#include <ratio>

struct EmptyStruct {};

// Test structs missing required members
struct MissingRep {
  using period                    = std::ratio<1>;
  using duration                  = std::chrono::seconds;
  using time_point                = std::chrono::time_point<MissingRep>;
  static constexpr bool is_steady = false;
  static time_point now();
};

struct MissingPeriod {
  using rep                       = long;
  using duration                  = std::chrono::seconds;
  using time_point                = std::chrono::time_point<MissingPeriod>;
  static constexpr bool is_steady = false;
  static time_point now();
};

struct MissingDuration {
  using rep                       = long;
  using time_point                = long;
  static constexpr bool is_steady = false;
  static time_point now();
};

struct MissingTimePoint {
  using rep                       = long;
  using period                    = std::ratio<1>;
  using duration                  = std::chrono::seconds;
  static constexpr bool is_steady = false;
  static std::chrono::time_point<MissingTimePoint> now();
};

struct MissingIsSteady {
  using rep        = long;
  using period     = std::ratio<1>;
  using duration   = std::chrono::seconds;
  using time_point = std::chrono::time_point<MissingIsSteady>;
  static time_point now();
};

struct MissingNow {
  using rep                       = long;
  using period                    = std::ratio<1>;
  using duration                  = std::chrono::seconds;
  using time_point                = std::chrono::time_point<MissingNow>;
  static constexpr bool is_steady = false;
};

// Valid clock types
struct ValidSteadyClock {
  using rep                       = long long;
  using period                    = std::nano;
  using duration                  = std::chrono::nanoseconds;
  using time_point                = std::chrono::time_point<ValidSteadyClock>;
  static constexpr bool is_steady = true;
  static time_point now();
};

struct ValidSystemClock {
  using rep                       = int64_t;
  using period                    = std::micro;
  using duration                  = std::chrono::microseconds;
  using time_point                = std::chrono::time_point<ValidSystemClock>;
  static constexpr bool is_steady = false;
  static time_point now();
};

int main(int, char**) {
  // Test both is_clock and is_clock_v
  static_assert(std::chrono::is_clock<std::chrono::system_clock>::value);
  static_assert(std::chrono::is_clock_v<std::chrono::system_clock>);

  // Test standard clock types
  static_assert(std::chrono::is_clock_v<std::chrono::system_clock>);
  static_assert(std::chrono::is_clock_v<std::chrono::steady_clock>);
  static_assert(std::chrono::is_clock_v<std::chrono::high_resolution_clock>);

  // Test non-clock types
  static_assert(!std::chrono::is_clock_v<EmptyStruct>);
  static_assert(!std::chrono::is_clock_v<int>);
  static_assert(!std::chrono::is_clock_v<void>);
  static_assert(!std::chrono::is_clock_v<std::chrono::system_clock::time_point>);
  static_assert(!std::chrono::is_clock_v<std::chrono::steady_clock::time_point>);
  static_assert(!std::chrono::is_clock_v<std::chrono::seconds>);
  static_assert(!std::chrono::is_clock_v<std::chrono::milliseconds>);

  // Test structs missing required members
  static_assert(!std::chrono::is_clock_v<MissingRep>);
  static_assert(!std::chrono::is_clock_v<MissingPeriod>);
  static_assert(!std::chrono::is_clock_v<MissingDuration>);
  static_assert(!std::chrono::is_clock_v<MissingTimePoint>);
  static_assert(!std::chrono::is_clock_v<MissingIsSteady>);
  static_assert(!std::chrono::is_clock_v<MissingNow>);

  // Test valid custom clocks
  static_assert(std::chrono::is_clock_v<ValidSteadyClock>);
  static_assert(std::chrono::is_clock_v<ValidSystemClock>);

  // cv-qualified and reference types
  static_assert(std::chrono::is_clock_v<const std::chrono::system_clock>);
  static_assert(std::chrono::is_clock_v<volatile std::chrono::system_clock>);
  static_assert(std::chrono::is_clock_v<const volatile std::chrono::system_clock>);
  static_assert(!std::chrono::is_clock_v<std::chrono::system_clock&>);
  static_assert(!std::chrono::is_clock_v<std::chrono::system_clock&&>);
  static_assert(!std::chrono::is_clock_v<const std::chrono::system_clock&>);

  // array and pointer types
  static_assert(!std::chrono::is_clock_v<std::chrono::system_clock[]>);
  static_assert(!std::chrono::is_clock_v<std::chrono::system_clock[10]>);
  static_assert(!std::chrono::is_clock_v<std::chrono::system_clock*>);
  static_assert(!std::chrono::is_clock_v<std::chrono::system_clock* const>);

  return 0;
}
