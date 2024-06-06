//===-- runtime/time-intrinsic.cpp ----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Implements time-related intrinsic subroutines.

#include "flang/Runtime/time-intrinsic.h"
#include "terminator.h"
#include "tools.h"
#include "flang/Runtime/cpp-type.h"
#include "flang/Runtime/descriptor.h"
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#ifdef _WIN32
#include "flang/Common/windows-include.h"
#else
#include <sys/time.h> // gettimeofday
#include <sys/times.h>
#include <unistd.h>
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

#if defined __MINGW32__
// clock_gettime is implemented in the pthread library for MinGW.
// Using it here would mean that all programs that link libFortranRuntime are
// required to also link to pthread. Instead, don't use the function.
#undef CLOCKID
#elif defined CLOCK_PROCESS_CPUTIME_ID
#define CLOCKID CLOCK_PROCESS_CPUTIME_ID
#elif defined CLOCK_THREAD_CPUTIME_ID
#define CLOCKID CLOCK_THREAD_CPUTIME_ID
#elif defined CLOCK_MONOTONIC
#define CLOCKID CLOCK_MONOTONIC
#elif defined CLOCK_REALTIME
#define CLOCKID CLOCK_REALTIME
#else
#undef CLOCKID
#endif

#ifdef CLOCKID
// POSIX implementation using clock_gettime. This is only enabled where
// clock_gettime is available.
template <typename T = int, typename U = struct timespec>
double GetCpuTime(preferred_implementation,
    // We need some dummy parameters to pass to decltype(clock_gettime).
    T ClockId = 0, U *Timespec = nullptr,
    decltype(clock_gettime(ClockId, Timespec)) *Enabled = nullptr) {
  struct timespec tspec;
  if (clock_gettime(CLOCKID, &tspec) == 0) {
    return tspec.tv_nsec * 1.0e-9 + tspec.tv_sec;
  }
  // Return some negative value to represent failure.
  return -1.0;
}
#endif

using count_t = std::int64_t;
using unsigned_count_t = std::uint64_t;

// Computes HUGE(INT(0,kind)) as an unsigned integer value.
static constexpr inline unsigned_count_t GetHUGE(int kind) {
  if (kind > 8) {
    kind = 8;
  }
  return (unsigned_count_t{1} << ((8 * kind) - 1)) - 1;
}

// This is the fallback implementation, which should work everywhere. Note that
// in general we can't recover after std::clock has reached its maximum value.
template <typename Unused = void>
count_t GetSystemClockCount(int kind, fallback_implementation) {
  std::clock_t timestamp{std::clock()};
  if (timestamp == static_cast<std::clock_t>(-1)) {
    // Return -HUGE(COUNT) to represent failure.
    return -static_cast<count_t>(GetHUGE(kind));
  }
  // Convert the timestamp to std::uint64_t with wrap-around. The timestamp is
  // most likely a floating-point value (since C'11), so compute the modulus
  // carefully when one is required.
  constexpr auto maxUnsignedCount{std::numeric_limits<unsigned_count_t>::max()};
  if constexpr (std::numeric_limits<std::clock_t>::max() > maxUnsignedCount) {
    timestamp -= maxUnsignedCount * std::floor(timestamp / maxUnsignedCount);
  }
  unsigned_count_t unsignedCount{static_cast<unsigned_count_t>(timestamp)};
  // Return the modulus of the unsigned integral count with HUGE(COUNT)+1.
  // The result is a signed integer but never negative.
  return static_cast<count_t>(unsignedCount % (GetHUGE(kind) + 1));
}

template <typename Unused = void>
count_t GetSystemClockCountRate(int kind, fallback_implementation) {
  return CLOCKS_PER_SEC;
}

template <typename Unused = void>
count_t GetSystemClockCountMax(int kind, fallback_implementation) {
  constexpr auto max_clock_t{std::numeric_limits<std::clock_t>::max()};
  unsigned_count_t maxCount{GetHUGE(kind)};
  return max_clock_t <= maxCount ? static_cast<count_t>(max_clock_t)
                                 : static_cast<count_t>(maxCount);
}

// POSIX implementation using clock_gettime where available.  The clock_gettime
// result is in nanoseconds, which is converted as necessary to
//  - deciseconds for kind 1
//  - milliseconds for kinds 2, 4
//  - nanoseconds for kinds 8, 16
constexpr unsigned_count_t DS_PER_SEC{10u};
constexpr unsigned_count_t MS_PER_SEC{1'000u};
constexpr unsigned_count_t NS_PER_SEC{1'000'000'000u};

#ifdef CLOCKID
template <typename T = int, typename U = struct timespec>
count_t GetSystemClockCount(int kind, preferred_implementation,
    // We need some dummy parameters to pass to decltype(clock_gettime).
    T ClockId = 0, U *Timespec = nullptr,
    decltype(clock_gettime(ClockId, Timespec)) *Enabled = nullptr) {
  struct timespec tspec;
  const unsigned_count_t huge{GetHUGE(kind)};
  if (clock_gettime(CLOCKID, &tspec) != 0) {
    return -huge; // failure
  }
  unsigned_count_t sec{static_cast<unsigned_count_t>(tspec.tv_sec)};
  unsigned_count_t nsec{static_cast<unsigned_count_t>(tspec.tv_nsec)};
  if (kind >= 8) {
    return (sec * NS_PER_SEC + nsec) % (huge + 1);
  } else if (kind >= 2) {
    return (sec * MS_PER_SEC + (nsec / (NS_PER_SEC / MS_PER_SEC))) % (huge + 1);
  } else { // kind == 1
    return (sec * DS_PER_SEC + (nsec / (NS_PER_SEC / DS_PER_SEC))) % (huge + 1);
  }
}
#endif

template <typename T = int, typename U = struct timespec>
count_t GetSystemClockCountRate(int kind, preferred_implementation,
    // We need some dummy parameters to pass to decltype(clock_gettime).
    T ClockId = 0, U *Timespec = nullptr,
    decltype(clock_gettime(ClockId, Timespec)) *Enabled = nullptr) {
  return kind >= 8 ? NS_PER_SEC : kind >= 2 ? MS_PER_SEC : DS_PER_SEC;
}

template <typename T = int, typename U = struct timespec>
count_t GetSystemClockCountMax(int kind, preferred_implementation,
    // We need some dummy parameters to pass to decltype(clock_gettime).
    T ClockId = 0, U *Timespec = nullptr,
    decltype(clock_gettime(ClockId, Timespec)) *Enabled = nullptr) {
  return GetHUGE(kind);
}

// DATE_AND_TIME (Fortran 2018 16.9.59)

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
static void DateAndTimeUnavailable(Fortran::runtime::Terminator &terminator,
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

// Dispatch to posix implementation where gettimeofday and localtime_r are
// available.
static void GetDateAndTime(Fortran::runtime::Terminator &terminator, char *date,
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
      Fortran::runtime::ApplyIntegerKind<Fortran::runtime::StoreIntegerAt,
          void>(kind, terminator, *values, atIndex, value);
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
// Fallback implementation where gettimeofday or localtime_r are not both
// available (e.g. windows).
static void GetDateAndTime(Fortran::runtime::Terminator &terminator, char *date,
    std::size_t dateChars, char *time, std::size_t timeChars, char *zone,
    std::size_t zoneChars, const Fortran::runtime::Descriptor *values) {
  // TODO: An actual implementation for non Posix system should be added.
  // So far, implement as if the date and time is not available on those
  // platforms.
  DateAndTimeUnavailable(
      terminator, date, dateChars, time, timeChars, zone, zoneChars, values);
}
#endif
} // namespace

namespace Fortran::runtime {
extern "C" {

double RTNAME(CpuTime)() { return GetCpuTime(0); }

std::int64_t RTNAME(SystemClockCount)(int kind) {
  return GetSystemClockCount(kind, 0);
}

std::int64_t RTNAME(SystemClockCountRate)(int kind) {
  return GetSystemClockCountRate(kind, 0);
}

std::int64_t RTNAME(SystemClockCountMax)(int kind) {
  return GetSystemClockCountMax(kind, 0);
}

void RTNAME(DateAndTime)(char *date, std::size_t dateChars, char *time,
    std::size_t timeChars, char *zone, std::size_t zoneChars,
    const char *source, int line, const Descriptor *values) {
  Fortran::runtime::Terminator terminator{source, line};
  return GetDateAndTime(
      terminator, date, dateChars, time, timeChars, zone, zoneChars, values);
}

void RTNAME(Etime)(const Descriptor *values, const Descriptor *time,
    const char *sourceFile, int line) {
  Fortran::runtime::Terminator terminator{sourceFile, line};

  double usrTime = -1.0, sysTime = -1.0, realTime = -1.0;

#ifdef _WIN32
  FILETIME creationTime;
  FILETIME exitTime;
  FILETIME kernelTime;
  FILETIME userTime;

  if (GetProcessTimes(GetCurrentProcess(), &creationTime, &exitTime,
          &kernelTime, &userTime) == 0) {
    ULARGE_INTEGER userSystemTime;
    ULARGE_INTEGER kernelSystemTime;

    memcpy(&userSystemTime, &userTime, sizeof(FILETIME));
    memcpy(&kernelSystemTime, &kernelTime, sizeof(FILETIME));

    usrTime = ((double)(userSystemTime.QuadPart)) / 10000000.0;
    sysTime = ((double)(kernelSystemTime.QuadPart)) / 10000000.0;
    realTime = usrTime + sysTime;
  }
#else
  struct tms tms;
  if (times(&tms) != (clock_t)-1) {
    usrTime = ((double)(tms.tms_utime)) / sysconf(_SC_CLK_TCK);
    sysTime = ((double)(tms.tms_stime)) / sysconf(_SC_CLK_TCK);
    realTime = usrTime + sysTime;
  }
#endif

  if (values) {
    auto typeCode{values->type().GetCategoryAndKind()};
    // ETIME values argument must have decimal range == 2.
    RUNTIME_CHECK(terminator,
        values->rank() == 1 && values->GetDimension(0).Extent() == 2 &&
            typeCode && typeCode->first == Fortran::common::TypeCategory::Real);
    // Only accept KIND=4 here.
    int kind{typeCode->second};
    RUNTIME_CHECK(terminator, kind == 4);

    ApplyFloatingPointKind<StoreFloatingPointAt, void>(
        kind, terminator, *values, /* atIndex = */ 0, usrTime);
    ApplyFloatingPointKind<StoreFloatingPointAt, void>(
        kind, terminator, *values, /* atIndex = */ 1, sysTime);
  }

  if (time) {
    auto typeCode{time->type().GetCategoryAndKind()};
    // ETIME time argument must have decimal range == 0.
    RUNTIME_CHECK(terminator,
        time->rank() == 0 && typeCode &&
            typeCode->first == Fortran::common::TypeCategory::Real);
    // Only accept KIND=4 here.
    int kind{typeCode->second};
    RUNTIME_CHECK(terminator, kind == 4);

    ApplyFloatingPointKind<StoreFloatingPointAt, void>(
        kind, terminator, *time, /* atIndex = */ 0, realTime);
  }
}

} // extern "C"
} // namespace Fortran::runtime
