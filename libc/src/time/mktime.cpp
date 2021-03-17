//===-- Implementation of mktime function ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/time/mktime.h"
#include "src/__support/common.h"
#include "src/time/time_utils.h"

#include <limits.h>

namespace __llvm_libc {

using __llvm_libc::time_utils::TimeConstants;

static constexpr int NonLeapYearDaysInMonth[] = {31, 28, 31, 30, 31, 30,
                                                 31, 31, 30, 31, 30, 31};

// Returns number of years from (1, year).
static constexpr int64_t getNumOfLeapYearsBefore(int64_t year) {
  return (year / 4) - (year / 100) + (year / 400);
}

// Returns True if year is a leap year.
static constexpr bool isLeapYear(const int64_t year) {
  return (((year) % 4) == 0 && (((year) % 100) != 0 || ((year) % 400) == 0));
}

LLVM_LIBC_FUNCTION(time_t, mktime, (struct tm * tm_out)) {
  // Unlike most C Library functions, mktime doesn't just die on bad input.
  // TODO(rtenneti); Handle leap seconds.
  int64_t tmYearFromBase = tm_out->tm_year + TimeConstants::TimeYearBase;

  // 32-bit end-of-the-world is 03:14:07 UTC on 19 January 2038.
  if (sizeof(time_t) == 4 &&
      tmYearFromBase >= TimeConstants::EndOf32BitEpochYear) {
    if (tmYearFromBase > TimeConstants::EndOf32BitEpochYear)
      return time_utils::OutOfRange();
    if (tm_out->tm_mon > 0)
      return time_utils::OutOfRange();
    if (tm_out->tm_mday > 19)
      return time_utils::OutOfRange();
    if (tm_out->tm_hour > 3)
      return time_utils::OutOfRange();
    if (tm_out->tm_min > 14)
      return time_utils::OutOfRange();
    if (tm_out->tm_sec > 7)
      return time_utils::OutOfRange();
  }

  // Years are ints.  A 32-bit year will fit into a 64-bit time_t.
  // A 64-bit year will not.
  static_assert(sizeof(int) == 4,
                "ILP64 is unimplemented.  This implementation requires "
                "32-bit integers.");

  // Calculate number of months and years from tm_mon.
  int64_t month = tm_out->tm_mon;
  if (month < 0 || month >= TimeConstants::MonthsPerYear - 1) {
    int64_t years = month / 12;
    month %= 12;
    if (month < 0) {
      years--;
      month += 12;
    }
    tmYearFromBase += years;
  }
  bool tmYearIsLeap = isLeapYear(tmYearFromBase);

  // Calculate total number of days based on the month and the day (tm_mday).
  int64_t totalDays = tm_out->tm_mday - 1;
  for (int64_t i = 0; i < month; ++i)
    totalDays += NonLeapYearDaysInMonth[i];
  // Add one day if it is a leap year and the month is after February.
  if (tmYearIsLeap && month > 1)
    totalDays++;

  // Calculate total numbers of days based on the year.
  totalDays += (tmYearFromBase - TimeConstants::EpochYear) *
               TimeConstants::DaysPerNonLeapYear;
  if (tmYearFromBase >= TimeConstants::EpochYear) {
    totalDays += getNumOfLeapYearsBefore(tmYearFromBase - 1) -
                 getNumOfLeapYearsBefore(TimeConstants::EpochYear);
  } else if (tmYearFromBase >= 1) {
    totalDays -= getNumOfLeapYearsBefore(TimeConstants::EpochYear) -
                 getNumOfLeapYearsBefore(tmYearFromBase - 1);
  } else {
    // Calculate number of leap years until 0th year.
    totalDays -= getNumOfLeapYearsBefore(TimeConstants::EpochYear) -
                 getNumOfLeapYearsBefore(0);
    if (tmYearFromBase <= 0) {
      totalDays -= 1; // Subtract 1 for 0th year.
      // Calculate number of leap years until -1 year
      if (tmYearFromBase < 0) {
        totalDays -= getNumOfLeapYearsBefore(-tmYearFromBase) -
                     getNumOfLeapYearsBefore(1);
      }
    }
  }

  // TODO(rtenneti): Need to handle timezone and update of tm_isdst.
  int64_t seconds = tm_out->tm_sec +
                    tm_out->tm_min * TimeConstants::SecondsPerMin +
                    tm_out->tm_hour * TimeConstants::SecondsPerHour +
                    totalDays * TimeConstants::SecondsPerDay;

  // Update the tm structure's year, month, day, etc. from seconds.
  if (time_utils::UpdateFromSeconds(seconds, tm_out) < 0)
    return time_utils::OutOfRange();

  return static_cast<time_t>(seconds);
}

} // namespace __llvm_libc
