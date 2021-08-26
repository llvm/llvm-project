//===-- runtime/time-intrinsic.cpp ----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Implements time-related intrinsic subroutines.

#include "time-intrinsic.h"

#include "descriptor.h"
#include "terminator.h"
#include "tools.h"
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#ifndef _WIN32
#include <sys/time.h> // gettimeofday
#endif

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

// DATE_AND_TIME (Fortran 2018 16.9.59)

// Helper to store integer value in result[at].
template <int KIND> struct StoreIntegerAt {
  void operator()(const Fortran::runtime::Descriptor &result, std::size_t at,
      std::int64_t value) const {
    *result.ZeroBasedIndexedElement<Fortran::runtime::CppTypeFor<
        Fortran::common::TypeCategory::Integer, KIND>>(at) = value;
  }
};

// Helper to set an integer value to -HUGE
template <int KIND> struct StoreNegativeHugeAt {
  void operator()(
      const Fortran::runtime::Descriptor &result, std::size_t at) const {
    *result.ZeroBasedIndexedElement<Fortran::runtime::CppTypeFor<
        Fortran::common::TypeCategory::Integer, KIND>>(at) =
        -std::numeric_limits<Fortran::runtime::CppTypeFor<
            Fortran::common::TypeCategory::Integer, KIND>>::max();
  }
};

// Default implementation when date and time information is not available (set
// strings to blanks and values to -HUGE as defined by the standard).
void DateAndTimeUnavailable(Fortran::runtime::Terminator &terminator,
    char *date, std::size_t dateChars, char *time, std::size_t timeChars,
    char *zone, std::size_t zoneChars,
    const Fortran::runtime::Descriptor *values) {
  if (date) {
    std::memset(date, static_cast<int>(' '), dateChars);
  }
  if (time) {
    std::memset(time, static_cast<int>(' '), timeChars);
  }
  if (zone) {
    std::memset(zone, static_cast<int>(' '), zoneChars);
  }
  if (values) {
    auto typeCode{values->type().GetCategoryAndKind()};
    RUNTIME_CHECK(terminator,
        values->rank() == 1 && values->GetDimension(0).Extent() >= 8 &&
            typeCode &&
            typeCode->first == Fortran::common::TypeCategory::Integer);
    // DATE_AND_TIME values argument must have decimal range > 4. Do not accept
    // KIND 1 here.
    int kind{typeCode->second};
    RUNTIME_CHECK(terminator, kind != 1);
    for (std::size_t i = 0; i < 8; ++i) {
      Fortran::runtime::ApplyIntegerKind<StoreNegativeHugeAt, void>(
          kind, terminator, *values, i);
    }
  }
}

#ifndef _WIN32

// SFINAE helper to return the struct tm.tm_gmtoff which is not a POSIX standard
// field.
template <int KIND, typename TM = struct tm>
Fortran::runtime::CppTypeFor<Fortran::common::TypeCategory::Integer, KIND>
GetGmtOffset(const TM &tm, preferred_implementation,
    decltype(tm.tm_gmtoff) *Enabled = nullptr) {
  // Returns the GMT offset in minutes.
  return tm.tm_gmtoff / 60;
}
template <int KIND, typename TM = struct tm>
Fortran::runtime::CppTypeFor<Fortran::common::TypeCategory::Integer, KIND>
GetGmtOffset(const TM &tm, fallback_implementation) {
  // tm.tm_gmtoff is not available, there may be platform dependent alternatives
  // (such as using timezone from <time.h> when available), but so far just
  // return -HUGE to report that this information is not available.
  return -std::numeric_limits<Fortran::runtime::CppTypeFor<
      Fortran::common::TypeCategory::Integer, KIND>>::max();
}
template <typename TM = struct tm> struct GmtOffsetHelper {
  template <int KIND> struct StoreGmtOffset {
    void operator()(const Fortran::runtime::Descriptor &result, std::size_t at,
        TM &tm) const {
      *result.ZeroBasedIndexedElement<Fortran::runtime::CppTypeFor<
          Fortran::common::TypeCategory::Integer, KIND>>(at) =
          GetGmtOffset<KIND>(tm, 0);
    }
  };
};

// Dispatch to posix implemetation when gettimeofday and localtime_r are
// available.
void GetDateAndTime(Fortran::runtime::Terminator &terminator, char *date,
    std::size_t dateChars, char *time, std::size_t timeChars, char *zone,
    std::size_t zoneChars, const Fortran::runtime::Descriptor *values) {

  timeval t;
  if (gettimeofday(&t, nullptr) != 0) {
    DateAndTimeUnavailable(
        terminator, date, dateChars, time, timeChars, zone, zoneChars, values);
    return;
  }
  time_t timer{t.tv_sec};
  tm localTime;
  localtime_r(&timer, &localTime);
  std::intmax_t ms{t.tv_usec / 1000};

  static constexpr std::size_t buffSize{16};
  char buffer[buffSize];
  auto copyBufferAndPad{
      [&](char *dest, std::size_t destChars, std::size_t len) {
        auto copyLen{std::min(len, destChars)};
        std::memcpy(dest, buffer, copyLen);
        for (auto i{copyLen}; i < destChars; ++i) {
          dest[i] = ' ';
        }
      }};
  if (date) {
    auto len = std::strftime(buffer, buffSize, "%Y%m%d", &localTime);
    copyBufferAndPad(date, dateChars, len);
  }
  if (time) {
    auto len{std::snprintf(buffer, buffSize, "%02d%02d%02d.%03jd",
        localTime.tm_hour, localTime.tm_min, localTime.tm_sec, ms)};
    copyBufferAndPad(time, timeChars, len);
  }
  if (zone) {
    // Note: this may leave the buffer empty on many platforms. Classic flang
    // has a much more complex way of doing this (see __io_timezone in classic
    // flang).
    auto len{std::strftime(buffer, buffSize, "%z", &localTime)};
    copyBufferAndPad(zone, zoneChars, len);
  }
  if (values) {
    auto typeCode{values->type().GetCategoryAndKind()};
    RUNTIME_CHECK(terminator,
        values->rank() == 1 && values->GetDimension(0).Extent() >= 8 &&
            typeCode &&
            typeCode->first == Fortran::common::TypeCategory::Integer);
    // DATE_AND_TIME values argument must have decimal range > 4. Do not accept
    // KIND 1 here.
    int kind{typeCode->second};
    RUNTIME_CHECK(terminator, kind != 1);
    auto storeIntegerAt = [&](std::size_t atIndex, std::int64_t value) {
      Fortran::runtime::ApplyIntegerKind<StoreIntegerAt, void>(
          kind, terminator, *values, atIndex, value);
    };
    storeIntegerAt(0, localTime.tm_year + 1900);
    storeIntegerAt(1, localTime.tm_mon + 1);
    storeIntegerAt(2, localTime.tm_mday);
    Fortran::runtime::ApplyIntegerKind<
        GmtOffsetHelper<struct tm>::StoreGmtOffset, void>(
        kind, terminator, *values, 3, localTime);
    storeIntegerAt(4, localTime.tm_hour);
    storeIntegerAt(5, localTime.tm_min);
    storeIntegerAt(6, localTime.tm_sec);
    storeIntegerAt(7, ms);
  }
}

#else
// Fallback implementation when gettimeofday or localtime_r is not available
// (e.g. windows).
void GetDateAndTime(Fortran::runtime::Terminator &terminator, char *date,
    std::size_t dateChars, char *time, std::size_t timeChars, char *zone,
    std::size_t zoneChars, const Fortran::runtime::Descriptor *values) {
  // TODO: An actual implementation for non Posix system should be added.
  // So far, implement as if the date and time is not available on those
  // platforms.
  DateAndTimeUnavailable(
      terminator, date, dateChars, time, timeChars, zone, zoneChars, values);
}
#endif
} // anonymous namespace

namespace Fortran::runtime {
extern "C" {

double RTNAME(CpuTime)() { return GetCpuTime(0); }

CppTypeFor<Fortran::common::TypeCategory::Integer, 8> RTNAME(
    SystemClockCount)() {
  return GetSystemClockCount(0);
}

CppTypeFor<Fortran::common::TypeCategory::Integer, 8> RTNAME(
    SystemClockCountRate)() {
  return GetSystemClockCountRate(0);
}

CppTypeFor<Fortran::common::TypeCategory::Integer, 8> RTNAME(
    SystemClockCountMax)() {
  return GetSystemClockCountMax(0);
}

void RTNAME(DateAndTime)(char *date, std::size_t dateChars, char *time,
    std::size_t timeChars, char *zone, std::size_t zoneChars,
    const char *source, int line, const Descriptor *values) {
  Fortran::runtime::Terminator terminator{source, line};
  return GetDateAndTime(
      terminator, date, dateChars, time, timeChars, zone, zoneChars, values);
}

} // extern "C"
} // namespace Fortran::runtime
