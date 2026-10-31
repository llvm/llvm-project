//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++20

// <chrono>

// template<class DestClock, class SourceClock>
//   struct clock_time_conversion {};

#include <chrono>
#include <ratio>
#include <type_traits>

struct UnrelatedClock {
  using rep                       = long long;
  using period                    = std::ratio<1>;
  using duration                  = std::chrono::duration<rep, period>;
  using time_point                = std::chrono::time_point<UnrelatedClock>;
  static constexpr bool is_steady = false;

  static time_point now();
};

struct OtherUnrelatedClock {
  using rep                       = long long;
  using period                    = std::ratio<1>;
  using duration                  = std::chrono::duration<rep, period>;
  using time_point                = std::chrono::time_point<OtherUnrelatedClock>;
  static constexpr bool is_steady = false;

  static time_point now();
};

template <class DestClock, class SourceClock, class Duration = typename SourceClock::duration>
concept HasConversion = requires(const std::chrono::time_point<SourceClock, Duration>& t) {
  std::chrono::clock_time_conversion<DestClock, SourceClock>{}(t);
};

// The primary template is an empty class, and so are the partial
// specializations that have no call operator for the given clocks.
static_assert(std::is_empty_v<std::chrono::clock_time_conversion<OtherUnrelatedClock, std::chrono::system_clock>>);
static_assert(std::is_empty_v<std::chrono::clock_time_conversion<UnrelatedClock, std::chrono::system_clock>>);
static_assert(std::is_empty_v<std::chrono::clock_time_conversion<int, long>>);
static_assert(std::is_trivially_default_constructible_v<
              std::chrono::clock_time_conversion<OtherUnrelatedClock, std::chrono::system_clock>>);

// The primary template has no call operator.
static_assert(!HasConversion<OtherUnrelatedClock, std::chrono::system_clock>);
static_assert(!HasConversion<std::chrono::system_clock, OtherUnrelatedClock>);
static_assert(!HasConversion<std::chrono::file_clock, OtherUnrelatedClock>);
static_assert(!HasConversion<UnrelatedClock, OtherUnrelatedClock>);

// A clock without to_sys/from_sys does not match the system_clock specializations.
static_assert(!HasConversion<std::chrono::system_clock, UnrelatedClock>);
static_assert(!HasConversion<UnrelatedClock, std::chrono::system_clock>);

// The standard provides specializations for these.
static_assert(HasConversion<OtherUnrelatedClock, OtherUnrelatedClock>);
static_assert(HasConversion<UnrelatedClock, UnrelatedClock>);
static_assert(HasConversion<std::chrono::system_clock, std::chrono::system_clock>);
static_assert(HasConversion<std::chrono::system_clock, std::chrono::file_clock>);
static_assert(HasConversion<std::chrono::file_clock, std::chrono::system_clock>);
