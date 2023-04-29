//===-- Implementation of mktime function ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/time/time_utils.h"
#include "src/__support/common.h"

#include <limits.h>

namespace __llvm_libc {
namespace time_utils {

using __llvm_libc::time_utils::TimeConstants;

static int64_t computeRemainingYears(int64_t daysPerYears,
                                     int64_t quotientYears,
                                     int64_t *remainingDays) {
  int64_t years = *remainingDays / daysPerYears;
  if (years == quotientYears)
    years--;
  *remainingDays -= years * daysPerYears;
  return years;
}

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
int64_t update_from_seconds(int64_t total_seconds, struct tm *tm) {
  // Days in month starting from March in the year 2000.
  static const char daysInMonth[] = {31 /* Mar */, 30, 31, 30, 31, 31,
                                     30,           31, 30, 31, 31, 29};

  if (sizeof(time_t) == 4) {
    if (total_seconds < 0x80000000)
      return time_utils::out_of_range();
    if (total_seconds > 0x7FFFFFFF)
      return time_utils::out_of_range();
  } else {
    if (total_seconds <
            INT_MIN * static_cast<int64_t>(
                          TimeConstants::NUMBER_OF_SECONDS_IN_LEAP_YEAR) ||
        total_seconds >
            INT_MAX * static_cast<int64_t>(
                          TimeConstants::NUMBER_OF_SECONDS_IN_LEAP_YEAR))
      return time_utils::out_of_range();
  }

  int64_t seconds =
      total_seconds - TimeConstants::SECONDS_UNTIL2000_MARCH_FIRST;
  int64_t days = seconds / TimeConstants::SECONDS_PER_DAY;
  int64_t remainingSeconds = seconds % TimeConstants::SECONDS_PER_DAY;
  if (remainingSeconds < 0) {
    remainingSeconds += TimeConstants::SECONDS_PER_DAY;
    days--;
  }

  int64_t wday = (TimeConstants::WEEK_DAY_OF2000_MARCH_FIRST + days) %
                 TimeConstants::DAYS_PER_WEEK;
  if (wday < 0)
    wday += TimeConstants::DAYS_PER_WEEK;

  // Compute the number of 400 year cycles.
  int64_t numOfFourHundredYearCycles = days / TimeConstants::DAYS_PER400_YEARS;
  int64_t remainingDays = days % TimeConstants::DAYS_PER400_YEARS;
  if (remainingDays < 0) {
    remainingDays += TimeConstants::DAYS_PER400_YEARS;
    numOfFourHundredYearCycles--;
  }

  // The remaining number of years after computing the number of
  // "four hundred year cycles" will be 4 hundred year cycles or less in 400
  // years.
  int64_t numOfHundredYearCycles = computeRemainingYears(
      TimeConstants::DAYS_PER100_YEARS, 4, &remainingDays);

  // The remaining number of years after computing the number of
  // "hundred year cycles" will be 25 four year cycles or less in 100 years.
  int64_t numOfFourYearCycles =
      computeRemainingYears(TimeConstants::DAYS_PER4_YEARS, 25, &remainingDays);

  // The remaining number of years after computing the number of
  // "four year cycles" will be 4 one year cycles or less in 4 years.
  int64_t remainingYears = computeRemainingYears(
      TimeConstants::DAYS_PER_NON_LEAP_YEAR, 4, &remainingDays);

  // Calculate number of years from year 2000.
  int64_t years = remainingYears + 4 * numOfFourYearCycles +
                  100 * numOfHundredYearCycles +
                  400LL * numOfFourHundredYearCycles;

  int leapDay =
      !remainingYears && (numOfFourYearCycles || !numOfHundredYearCycles);

  // We add 31 and 28 for the number of days in January and February, since our
  // starting point was March 1st.
  int64_t yday = remainingDays + 31 + 28 + leapDay;
  if (yday >= TimeConstants::DAYS_PER_NON_LEAP_YEAR + leapDay)
    yday -= TimeConstants::DAYS_PER_NON_LEAP_YEAR + leapDay;

  int64_t months = 0;
  while (daysInMonth[months] <= remainingDays) {
    remainingDays -= daysInMonth[months];
    months++;
  }

  if (months >= TimeConstants::MONTHS_PER_YEAR - 2) {
    months -= TimeConstants::MONTHS_PER_YEAR;
    years++;
  }

  if (years > INT_MAX || years < INT_MIN)
    return time_utils::out_of_range();

  // All the data (years, month and remaining days) was calculated from
  // March, 2000. Thus adjust the data to be from January, 1900.
  tm->tm_year = years + 2000 - TimeConstants::TIME_YEAR_BASE;
  tm->tm_mon = months + 2;
  tm->tm_mday = remainingDays + 1;
  tm->tm_wday = wday;
  tm->tm_yday = yday;

  tm->tm_hour = remainingSeconds / TimeConstants::SECONDS_PER_HOUR;
  tm->tm_min = remainingSeconds / TimeConstants::SECONDS_PER_MIN %
               TimeConstants::SECONDS_PER_MIN;
  tm->tm_sec = remainingSeconds % TimeConstants::SECONDS_PER_MIN;
  // TODO(rtenneti): Need to handle timezone and update of tm_isdst.
  tm->tm_isdst = 0;

  return 0;
}

} // namespace time_utils
} // namespace __llvm_libc
