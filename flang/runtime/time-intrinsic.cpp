//===-- runtime/time-intrinsic.cpp ----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Implements time-related intrinsic subroutines.

#include "time-intrinsic.h"

#include <ctime>

// CPU_TIME (Fortran 2018 16.9.57)
// SYSTEM_CLOCK (Fortran 2018 16.9.168)
//
// We can use std::clock() from the <ctime> header as a fallback implementation
// that should be available everywhere. This may not provide the best resolution
// and is particularly troublesome on (some?) POSIX systems where CLOCKS_PER_SEC
// is defined as 10^6 regardless of the actual precision of std::clock().
// Therefore, we will usually prefer platform-specific alternatives when they
// are available.
//
// We can use SFINAE to choose a platform-specific alternative. To do so, we
// introduce a helper function template, whose overload set will contain only
// implementations relying on interfaces which are actually available. Each
// overload will have a dummy parameter whose type indicates whether or not it
// should be preferred. Any other parameters required for SFINAE should have
// default values provided.
namespace {
// Types for the dummy parameter indicating the priority of a given overload.
// We will invoke our helper with an integer literal argument, so the overload
// with the highest priority should have the type int.
using fallback_implementation = double;
using preferred_implementation = int;

// This is the fallback implementation, which should work everywhere.
template <typename Unused = void> double GetCpuTime(fallback_implementation) {
  std::clock_t timestamp{std::clock()};
  if (timestamp != static_cast<std::clock_t>(-1)) {
    return static_cast<double>(timestamp) / CLOCKS_PER_SEC;
  }

  // Return some negative value to represent failure.
  return -1.0;
}

// POSIX implementation using clock_gettime. This is only enabled if
// clock_gettime is available.
template <typename T = int, typename U = struct timespec>
double GetCpuTime(preferred_implementation,
    // We need some dummy parameters to pass to decltype(clock_gettime).
    T ClockId = 0, U *Timespec = nullptr,
    decltype(clock_gettime(ClockId, Timespec)) *Enabled = nullptr) {
#if defined CLOCK_THREAD_CPUTIME_ID
#define CLOCKID CLOCK_THREAD_CPUTIME_ID
#elif defined CLOCK_PROCESS_CPUTIME_ID
#define CLOCKID CLOCK_PROCESS_CPUTIME_ID
#elif defined CLOCK_MONOTONIC
#define CLOCKID CLOCK_MONOTONIC
#else
#define CLOCKID CLOCK_REALTIME
#endif
  struct timespec tspec;
  if (clock_gettime(CLOCKID, &tspec) == 0) {
    return tspec.tv_nsec * 1.0e-9 + tspec.tv_sec;
  }

  // Return some negative value to represent failure.
  return -1.0;
}

using count_t =
    Fortran::runtime::CppTypeFor<Fortran::common::TypeCategory::Integer, 8>;

// This is the fallback implementation, which should work everywhere. Note that
// in general we can't recover after std::clock has reached its maximum value.
template <typename Unused = void>
count_t GetSystemClockCount(fallback_implementation) {
  std::clock_t timestamp{std::clock()};
  if (timestamp == static_cast<std::clock_t>(-1)) {
    // Return -HUGE() to represent failure.
    return -std::numeric_limits<count_t>::max();
  }

  // If our return type is large enough to hold any value returned by
  // std::clock, our work is done. Otherwise, we have to wrap around.
  static constexpr auto max{std::numeric_limits<count_t>::max()};
  if constexpr (std::numeric_limits<std::clock_t>::max() <= max) {
    return static_cast<count_t>(timestamp);
  } else {
    // Since std::clock_t could be a floating point type, we can't just use the
    // % operator, so we have to wrap around manually.
    return static_cast<count_t>(timestamp - max * std::floor(timestamp / max));
  }
}

template <typename Unused = void>
count_t GetSystemClockCountRate(fallback_implementation) {
  return CLOCKS_PER_SEC;
}

template <typename Unused = void>
count_t GetSystemClockCountMax(fallback_implementation) {
  static constexpr auto max_clock_t = std::numeric_limits<std::clock_t>::max();
  static constexpr auto max_count_t = std::numeric_limits<count_t>::max();
  if constexpr (max_clock_t < max_count_t) {
    return static_cast<count_t>(max_clock_t);
  } else {
    return max_count_t;
  }
}

constexpr count_t NSECS_PER_SEC{1'000'000'000};

// POSIX implementation using clock_gettime. This is only enabled if
// clock_gettime is available.
template <typename T = int, typename U = struct timespec>
count_t GetSystemClockCount(preferred_implementation,
    // We need some dummy parameters to pass to decltype(clock_gettime).
    T ClockId = 0, U *Timespec = nullptr,
    decltype(clock_gettime(ClockId, Timespec)) *Enabled = nullptr) {
#if defined CLOCK_THREAD_CPUTIME_ID
#define CLOCKID CLOCK_THREAD_CPUTIME_ID
#elif defined CLOCK_PROCESS_CPUTIME_ID
#define CLOCKID CLOCK_PROCESS_CPUTIME_ID
#elif defined CLOCK_MONOTONIC
#define CLOCKID CLOCK_MONOTONIC
#else
#define CLOCKID CLOCK_REALTIME
#endif
  struct timespec tspec;
  if (clock_gettime(CLOCKID, &tspec) != 0) {
    // Return -HUGE() to represent failure.
    return -std::numeric_limits<count_t>::max();
  }

  // Wrap around to avoid overflows.
  constexpr count_t max_secs{
      std::numeric_limits<count_t>::max() / NSECS_PER_SEC};
  count_t wrapped_secs{tspec.tv_sec % max_secs};

  // At this point, wrapped_secs < max_secs, and max_secs has already been
  // truncated by the division. Therefore, we should still have enough room to
  // add tv_nsec, since it is < NSECS_PER_SEC.
  return tspec.tv_nsec + wrapped_secs * NSECS_PER_SEC;
}

template <typename T = int, typename U = struct timespec>
count_t GetSystemClockCountRate(preferred_implementation,
    // We need some dummy parameters to pass to decltype(clock_gettime).
    T ClockId = 0, U *Timespec = nullptr,
    decltype(clock_gettime(ClockId, Timespec)) *Enabled = nullptr) {
  return NSECS_PER_SEC;
}

template <typename T = int, typename U = struct timespec>
count_t GetSystemClockCountMax(preferred_implementation,
    // We need some dummy parameters to pass to decltype(clock_gettime).
    T ClockId = 0, U *Timespec = nullptr,
    decltype(clock_gettime(ClockId, Timespec)) *Enabled = nullptr) {
  count_t max_secs{std::numeric_limits<count_t>::max() / NSECS_PER_SEC};
  return max_secs * NSECS_PER_SEC - 1;
}
} // anonymous namespace

namespace Fortran::runtime {
extern "C" {

double RTNAME(CpuTime)() { return GetCpuTime(0); }

CppTypeFor<TypeCategory::Integer, 8> RTNAME(SystemClockCount)() {
  return GetSystemClockCount(0);
}

CppTypeFor<TypeCategory::Integer, 8> RTNAME(SystemClockCountRate)() {
  return GetSystemClockCountRate(0);
}

CppTypeFor<TypeCategory::Integer, 8> RTNAME(SystemClockCountMax)() {
  return GetSystemClockCountMax(0);
}
} // extern "C"
} // namespace Fortran::runtime
