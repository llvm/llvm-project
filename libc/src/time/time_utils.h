//===-- Collection of utils for mktime and friends --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_TIME_TIME_UTILS_H
#define LLVM_LIBC_SRC_TIME_TIME_UTILS_H

#include <stddef.h> // For size_t.

#include "include/errno.h"

#include "src/errno/llvmlibc_errno.h"
#include "src/time/mktime.h"

#include <stdint.h>

namespace __llvm_libc {
namespace time_utils {

struct TimeConstants {
  static constexpr int SecondsPerMin = 60;
  static constexpr int SecondsPerHour = 3600;
  static constexpr int SecondsPerDay = 86400;
  static constexpr int DaysPerWeek = 7;
  static constexpr int MonthsPerYear = 12;
  static constexpr int DaysPerNonLeapYear = 365;
  static constexpr int DaysPerLeapYear = 366;
  static constexpr int TimeYearBase = 1900;
  static constexpr int EpochYear = 1970;
  static constexpr int EpochWeekDay = 4;
  static constexpr int NumberOfSecondsInLeapYear =
      (DaysPerNonLeapYear + 1) * SecondsPerDay;

  // For asctime the behavior is undefined if struct tm's tm_wday or tm_mon are
  // not within the normal ranges as defined in <time.h>, or if struct tm's
  // tm_year exceeds {INT_MAX}-1990, or if the below asctime_internal algorithm
  // would attempt to generate more than 26 bytes of output (including the
  // terminating null).
  static constexpr int AsctimeBufferSize = 256;
  static constexpr int AsctimeMaxBytes = 26;

  /* 2000-03-01 (mod 400 year, immediately after feb29 */
  static constexpr int64_t SecondsUntil2000MarchFirst =
      (946684800LL + SecondsPerDay * (31 + 29));
  static constexpr int WeekDayOf2000MarchFirst = 3;

  static constexpr int DaysPer400Years =
      (DaysPerNonLeapYear * 400 + (400 / 4) - 3);
  static constexpr int DaysPer100Years =
      (DaysPerNonLeapYear * 100 + (100 / 4) - 1);
  static constexpr int DaysPer4Years = (DaysPerNonLeapYear * 4 + 1);

  // The latest time that can be represented in this form is 03:14:07 UTC on
  // Tuesday, 19 January 2038 (corresponding to 2,147,483,647 seconds since the
  // start of the epoch). This means that systems using a 32-bit time_t type are
  // susceptible to the Year 2038 problem.
  static constexpr int EndOf32BitEpochYear = 2038;

  static constexpr time_t OutOfRangeReturnValue = -1;
};

// Update the "tm" structure's year, month, etc. members from seconds.
// "total_seconds" is the number of seconds since January 1st, 1970.
extern int64_t UpdateFromSeconds(int64_t total_seconds, struct tm *tm);

// POSIX.1-2017 requires this.
static inline time_t OutOfRange() {
  llvmlibc_errno = EOVERFLOW;
  return static_cast<time_t>(-1);
}

static inline void InvalidValue() { llvmlibc_errno = EINVAL; }

static inline char *asctime(const struct tm *timeptr, char *buffer,
                            size_t bufferLength) {
  if (timeptr == nullptr || buffer == nullptr) {
    InvalidValue();
    return nullptr;
  }
  if (timeptr->tm_wday < 0 ||
      timeptr->tm_wday > (TimeConstants::DaysPerWeek - 1)) {
    InvalidValue();
    return nullptr;
  }
  if (timeptr->tm_mon < 0 ||
      timeptr->tm_mon > (TimeConstants::MonthsPerYear - 1)) {
    InvalidValue();
    return nullptr;
  }

  // TODO(rtenneti): i18n the following strings.
  static const char *WeekDaysName[TimeConstants::DaysPerWeek] = {
      "Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"};

  static const char *MonthsName[TimeConstants::MonthsPerYear] = {
      "Jan", "Feb", "Mar", "Apr", "May", "Jun",
      "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"};
  int written_size = __builtin_snprintf(
      buffer, bufferLength, "%.3s %.3s%3d %.2d:%.2d:%.2d %d\n",
      WeekDaysName[timeptr->tm_wday], MonthsName[timeptr->tm_mon],
      timeptr->tm_mday, timeptr->tm_hour, timeptr->tm_min, timeptr->tm_sec,
      TimeConstants::TimeYearBase + timeptr->tm_year);
  if (written_size < 0)
    return nullptr;
  if (static_cast<size_t>(written_size) >= bufferLength) {
    OutOfRange();
    return nullptr;
  }
  return buffer;
}

static inline struct tm *gmtime_internal(const time_t *timer,
                                         struct tm *result) {
  int64_t seconds = *timer;
  // Update the tm structure's year, month, day, etc. from seconds.
  if (UpdateFromSeconds(seconds, result) < 0) {
    OutOfRange();
    return nullptr;
  }

  return result;
}

} // namespace time_utils
} // namespace __llvm_libc

#endif // LLVM_LIBC_SRC_TIME_TIME_UTILS_H
