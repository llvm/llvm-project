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

static int64_t computeRemainingYears(int64_t daysPerYears,
                                     int64_t quotientYears,
                                     int64_t *remainingDays) {
  int64_t years = *remainingDays / daysPerYears;
  if (years == quotientYears)
    years--;
  *remainingDays -= years * daysPerYears;
  return years;
}

// Update the "tm" structure's year, month, etc. members from seconds.
// "total_seconds" is the number of seconds since January 1st, 1970.
//
// First, divide "total_seconds" by the number of seconds in a day to get the
// number of days since Jan 1 1970. The remainder will be used to calculate the
// number of Hours, Minutes and Seconds.
//
// Then, adjust that number of days by a constant to be the number of days
// since Mar 1 2000. Year 2000 is a multiple of 400, the leap year cycle. This
// makes it easier to count how many leap years have passed using division.
//
// While calculating numbers of years in the days, the following algorithm
// subdivides the days into the number of 400 years, the number of 100 years and
// the number of 4 years. These numbers of cycle years are used in calculating
// leap day. This is similar to the algorithm used in  getNumOfLeapYearsBefore()
// and isLeapYear(). Then compute the total number of years in days from these
// subdivided units.
//
// Compute the number of months from the remaining days. Finally, adjust years
// to be 1900 and months to be from January.
static int64_t updateFromSeconds(int64_t total_seconds, struct tm *tm) {
  // Days in month starting from March in the year 2000.
  static const char daysInMonth[] = {31 /* Mar */, 30, 31, 30, 31, 31,
                                     30,           31, 30, 31, 31, 29};

  if (sizeof(time_t) == 4) {
    if (total_seconds < 0x80000000)
      return time_utils::OutOfRange();
    if (total_seconds > 0x7FFFFFFF)
      return time_utils::OutOfRange();
  } else {
    if (total_seconds <
            INT_MIN * static_cast<int64_t>(
                          TimeConstants::NumberOfSecondsInLeapYear) ||
        total_seconds > INT_MAX * static_cast<int64_t>(
                                      TimeConstants::NumberOfSecondsInLeapYear))
      return time_utils::OutOfRange();
  }

  int64_t seconds = total_seconds - TimeConstants::SecondsUntil2000MarchFirst;
  int64_t days = seconds / TimeConstants::SecondsPerDay;
  int64_t remainingSeconds = seconds % TimeConstants::SecondsPerDay;
  if (remainingSeconds < 0) {
    remainingSeconds += TimeConstants::SecondsPerDay;
    days--;
  }

  int64_t wday = (TimeConstants::WeekDayOf2000MarchFirst + days) %
                 TimeConstants::DaysPerWeek;
  if (wday < 0)
    wday += TimeConstants::DaysPerWeek;

  // Compute the number of 400 year cycles.
  int64_t numOfFourHundredYearCycles = days / TimeConstants::DaysPer400Years;
  int64_t remainingDays = days % TimeConstants::DaysPer400Years;
  if (remainingDays < 0) {
    remainingDays += TimeConstants::DaysPer400Years;
    numOfFourHundredYearCycles--;
  }

  // The reminder number of years after computing number of
  // "four hundred year cycles" will be 4 hundred year cycles or less in 400
  // years.
  int64_t numOfHundredYearCycles =
      computeRemainingYears(TimeConstants::DaysPer100Years, 4, &remainingDays);

  // The reminder number of years after computing number of
  // "hundred year cycles" will be 25 four year cycles or less in 100 years.
  int64_t numOfFourYearCycles =
      computeRemainingYears(TimeConstants::DaysPer4Years, 25, &remainingDays);

  // The reminder number of years after computing number of "four year cycles"
  // will be 4 one year cycles or less in 4 years.
  int64_t remainingYears = computeRemainingYears(
      TimeConstants::DaysPerNonLeapYear, 4, &remainingDays);

  // Calculate number of years from year 2000.
  int64_t years = remainingYears + 4 * numOfFourYearCycles +
                  100 * numOfHundredYearCycles +
                  400LL * numOfFourHundredYearCycles;

  int leapDay =
      !remainingYears && (numOfFourYearCycles || !numOfHundredYearCycles);

  int64_t yday = remainingDays + 31 + 28 + leapDay;
  if (yday >= TimeConstants::DaysPerNonLeapYear + leapDay)
    yday -= TimeConstants::DaysPerNonLeapYear + leapDay;

  int64_t months = 0;
  while (daysInMonth[months] <= remainingDays) {
    remainingDays -= daysInMonth[months];
    months++;
  }

  if (months >= TimeConstants::MonthsPerYear - 2) {
    months -= TimeConstants::MonthsPerYear;
    years++;
  }

  if (years > INT_MAX || years < INT_MIN)
    return time_utils::OutOfRange();

  // All the data (years, month and remaining days) was calculated from
  // March, 2000. Thus adjust the data to be from January, 1900.
  tm->tm_year = years + 2000 - TimeConstants::TimeYearBase;
  tm->tm_mon = months + 2;
  tm->tm_mday = remainingDays + 1;
  tm->tm_wday = wday;
  tm->tm_yday = yday;

  tm->tm_hour = remainingSeconds / TimeConstants::SecondsPerHour;
  tm->tm_min = remainingSeconds / TimeConstants::SecondsPerMin %
               TimeConstants::SecondsPerMin;
  tm->tm_sec = remainingSeconds % TimeConstants::SecondsPerMin;

  return 0;
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
  if (updateFromSeconds(seconds, tm_out) < 0)
    return time_utils::OutOfRange();

  return static_cast<time_t>(seconds);
}

} // namespace __llvm_libc
