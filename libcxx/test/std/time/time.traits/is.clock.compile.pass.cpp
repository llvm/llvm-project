//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++20

// <chrono>
//
// template<class T> struct is_clock;
// template<class T> constexpr bool is_clock_v = is_clock<T>::value;

#include <chrono>
#include <ratio>

#include "test_macros.h"

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
  using rep                       = long long;
  using period                    = std::micro;
  using duration                  = std::chrono::microseconds;
  using time_point                = std::chrono::time_point<ValidSystemClock>;
  static constexpr bool is_steady = false;
  static time_point now();
};

// Test clocks with invalid is_steady type
struct WrongIsSteadyType {
  using rep        = long;
  using period     = std::ratio<1>;
  using duration   = std::chrono::seconds;
  using time_point = std::chrono::time_point<WrongIsSteadyType>;
  static bool is_steady; // Not const bool
  static time_point now();
};

struct WrongIsSteadyNonBool {
  using rep                      = long;
  using period                   = std::ratio<1>;
  using duration                 = std::chrono::seconds;
  using time_point               = std::chrono::time_point<WrongIsSteadyNonBool>;
  static constexpr int is_steady = 1; // Not bool
  static time_point now();
};

// Test clocks with invalid now() return type
struct WrongNowReturnType {
  using rep                       = long;
  using period                    = std::ratio<1>;
  using duration                  = std::chrono::seconds;
  using time_point                = std::chrono::time_point<WrongNowReturnType>;
  static constexpr bool is_steady = false;
  static int now(); // Wrong return type
};

// Test clocks with invalid period type
struct WrongPeriodType {
  using rep                       = long;
  using period                    = int; // Not a ratio
  using duration                  = std::chrono::seconds;
  using time_point                = std::chrono::time_point<WrongPeriodType>;
  static constexpr bool is_steady = false;
  static time_point now();
};

// Test clocks with wrong duration type
struct WrongDurationType {
  using rep                       = long;
  using period                    = std::ratio<1>;
  using duration                  = std::chrono::milliseconds; // Should be duration<long, ratio<1>>
  using time_point                = std::chrono::time_point<WrongDurationType>;
  static constexpr bool is_steady = false;
  static time_point now();
};

// Test clocks with wrong time_point type
struct WrongTimePointType {
  using rep                       = long;
  using period                    = std::ratio<1>;
  using duration                  = std::chrono::duration<long, std::ratio<1>>;
  using time_point                = int; // Not a time_point
  static constexpr bool is_steady = false;
  static time_point now();
};

struct WrongTimePointClock {
  using rep                       = long;
  using period                    = std::ratio<1>;
  using duration                  = std::chrono::duration<long, std::ratio<1>>;
  using time_point                = std::chrono::time_point<ValidSystemClock>; // Wrong clock type
  static constexpr bool is_steady = false;
  static time_point now();
};

// Valid clock with time_point that has matching duration instead of matching clock
struct ValidClockWithDurationMatch {
  using rep                       = int;
  using period                    = std::milli;
  using duration                  = std::chrono::duration<int, std::milli>;
  using time_point                = std::chrono::time_point<ValidSystemClock, duration>; // Valid: matches duration
  static constexpr bool is_steady = false;
  static time_point now();
};

// Test both is_clock and is_clock_v
static_assert(std::chrono::is_clock<std::chrono::system_clock>::value);
static_assert(std::chrono::is_clock_v<std::chrono::system_clock>);

// Test standard clock types
static_assert(std::chrono::is_clock_v<std::chrono::system_clock>);
static_assert(std::chrono::is_clock_v<std::chrono::high_resolution_clock>);

// Test non-clock types
static_assert(!std::chrono::is_clock_v<EmptyStruct>);
static_assert(!std::chrono::is_clock_v<int>);
static_assert(!std::chrono::is_clock_v<void>);
static_assert(!std::chrono::is_clock_v<std::chrono::system_clock::time_point>);
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
static_assert(std::chrono::is_clock_v<ValidClockWithDurationMatch>);

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

// The Standard defined a minimum set of checks and allowed implementation to perform stricter checks. The following
// static asserts are implementation specific and a conforming standard library implementation doesn't have to produce
// the same outcome.

// Test clocks with invalid is_steady type
LIBCPP_STATIC_ASSERT(!std::chrono::is_clock_v<WrongIsSteadyType>);    // is_steady not const bool
LIBCPP_STATIC_ASSERT(!std::chrono::is_clock_v<WrongIsSteadyNonBool>); // is_steady not bool type

// Test clocks with invalid now() return type
LIBCPP_STATIC_ASSERT(!std::chrono::is_clock_v<WrongNowReturnType>); // now() doesn't return time_point

// Test clocks with invalid period type
LIBCPP_STATIC_ASSERT(!std::chrono::is_clock_v<WrongPeriodType>); // period is not a ratio

// Test clocks with wrong duration type
LIBCPP_STATIC_ASSERT(!std::chrono::is_clock_v<WrongDurationType>); // duration doesn't match duration<rep, period>

// Test clocks with wrong time_point type
LIBCPP_STATIC_ASSERT(!std::chrono::is_clock_v<WrongTimePointType>);  // time_point is not a time_point
LIBCPP_STATIC_ASSERT(!std::chrono::is_clock_v<WrongTimePointClock>); // time_point has wrong clock and wrong duration
