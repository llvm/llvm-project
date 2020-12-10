//===-- Implementation of mktime function ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "include/errno.h"

#include "src/__support/common.h"
#include "src/errno/llvmlibc_errno.h"
#include "src/time/mktime.h"

namespace __llvm_libc {

constexpr int SecondsPerMin = 60;
constexpr int MinutesPerHour = 60;
constexpr int HoursPerDay = 24;
constexpr int DaysPerWeek = 7;
constexpr int MonthsPerYear = 12;
constexpr int DaysPerNonLeapYear = 365;
constexpr int TimeYearBase = 1900;
constexpr int EpochYear = 1970;
constexpr int EpochWeekDay = 4;
// The latest time that can be represented in this form is 03:14:07 UTC on
// Tuesday, 19 January 2038 (corresponding to 2,147,483,647 seconds since the
// start of the epoch). This means that systems using a 32-bit time_t type are
// susceptible to the Year 2038 problem.
constexpr int EndOf32BitEpochYear = 2038;

constexpr int NonLeapYearDaysInMonth[] = {31 /* Jan */, 28, 31, 30, 31, 30,
                                          31,           31, 30, 31, 30, 31};

constexpr bool isLeapYear(const time_t year) {
  return (((year) % 4) == 0 && (((year) % 100) != 0 || ((year) % 400) == 0));
}

// POSIX.1-2017 requires this.
static inline time_t outOfRange() {
  llvmlibc_errno = EOVERFLOW;
  return static_cast<time_t>(-1);
}

time_t LLVM_LIBC_ENTRYPOINT(mktime)(struct tm *t1) {
  // Unlike most C Library functions, mktime doesn't just die on bad input.
  // TODO(rtenneti); Handle leap seconds. Handle out of range time and date
  // values that don't overflow or underflow.
  // TODO (rtenneti): Implement the following suggestion Siva: "As we start
  // accumulating the seconds, we should be able to check if the next amount of
  // seconds to be added can lead to an overflow. If it does, return the
  // overflow value. If not keep accumulating. The benefit is that, we don't
  // have to validate every input, and also do not need the special cases for
  // sizeof(time_t) == 4".
  if (t1->tm_sec < 0 || t1->tm_sec > (SecondsPerMin - 1))
    return outOfRange();
  if (t1->tm_min < 0 || t1->tm_min > (MinutesPerHour - 1))
    return outOfRange();
  if (t1->tm_hour < 0 || t1->tm_hour > (HoursPerDay - 1))
    return outOfRange();
  time_t tmYearFromBase = t1->tm_year + TimeYearBase;

  if (tmYearFromBase < EpochYear)
    return outOfRange();

  // 32-bit end-of-the-world is 03:14:07 UTC on 19 January 2038.
  if (sizeof(time_t) == 4 && tmYearFromBase >= EndOf32BitEpochYear) {
    if (tmYearFromBase > EndOf32BitEpochYear)
      return outOfRange();
    if (t1->tm_mon > 0)
      return outOfRange();
    if (t1->tm_mday > 19)
      return outOfRange();
    if (t1->tm_hour > 3)
      return outOfRange();
    if (t1->tm_min > 14)
      return outOfRange();
    if (t1->tm_sec > 7)
      return outOfRange();
  }

  // Years are ints.  A 32-bit year will fit into a 64-bit time_t.
  // A 64-bit year will not.
  static_assert(sizeof(int) == 4,
                "ILP64 is unimplemented.  This implementation requires "
                "32-bit integers.");

  if (t1->tm_mon < 0 || t1->tm_mon > (MonthsPerYear - 1))
    return outOfRange();
  bool tmYearIsLeap = isLeapYear(tmYearFromBase);
  time_t daysInMonth = NonLeapYearDaysInMonth[t1->tm_mon];
  // Add one day if it is a leap year and the month is February.
  if (tmYearIsLeap && t1->tm_mon == 1)
    ++daysInMonth;
  if (t1->tm_mday < 1 || t1->tm_mday > daysInMonth)
    return outOfRange();

  time_t totalDays = t1->tm_mday - 1;
  for (int i = 0; i < t1->tm_mon; ++i)
    totalDays += NonLeapYearDaysInMonth[i];
  // Add one day if it is a leap year and the month is after February.
  if (tmYearIsLeap && t1->tm_mon > 1)
    totalDays++;
  t1->tm_yday = totalDays;
  totalDays += (tmYearFromBase - EpochYear) * DaysPerNonLeapYear;

  // Add an extra day for each leap year, starting with 1972
  for (time_t year = EpochYear + 2; year < tmYearFromBase;) {
    if (isLeapYear(year)) {
      totalDays += 1;
      year += 4;
    } else {
      year++;
    }
  }

  t1->tm_wday = (EpochWeekDay + totalDays) % DaysPerWeek;
  if (t1->tm_wday < 0)
    t1->tm_wday += DaysPerWeek;
  // TODO(rtenneti): Need to handle timezone and update of tm_isdst.
  return t1->tm_sec + t1->tm_min * SecondsPerMin +
         t1->tm_hour * MinutesPerHour * SecondsPerMin +
         totalDays * HoursPerDay * MinutesPerHour * SecondsPerMin;
}

} // namespace __llvm_libc
