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

static constexpr int NON_LEAP_YEAR_DAYS_IN_MONTH[] = {31, 28, 31, 30, 31, 30,
                                                      31, 31, 30, 31, 30, 31};

// Returns number of years from (1, year).
static constexpr int64_t get_num_of_leap_years_before(int64_t year) {
  return (year / 4) - (year / 100) + (year / 400);
}

// Returns True if year is a leap year.
static constexpr bool is_leap_year(const int64_t year) {
  return (((year) % 4) == 0 && (((year) % 100) != 0 || ((year) % 400) == 0));
}

LLVM_LIBC_FUNCTION(time_t, mktime, (struct tm * tm_out)) {
  // Unlike most C Library functions, mktime doesn't just die on bad input.
  // TODO(rtenneti); Handle leap seconds.
  int64_t tm_year_from_base = tm_out->tm_year + TimeConstants::TIME_YEAR_BASE;

  // 32-bit end-of-the-world is 03:14:07 UTC on 19 January 2038.
  if (sizeof(time_t) == 4 &&
      tm_year_from_base >= TimeConstants::END_OF32_BIT_EPOCH_YEAR) {
    if (tm_year_from_base > TimeConstants::END_OF32_BIT_EPOCH_YEAR)
      return time_utils::out_of_range();
    if (tm_out->tm_mon > 0)
      return time_utils::out_of_range();
    if (tm_out->tm_mday > 19)
      return time_utils::out_of_range();
    if (tm_out->tm_hour > 3)
      return time_utils::out_of_range();
    if (tm_out->tm_min > 14)
      return time_utils::out_of_range();
    if (tm_out->tm_sec > 7)
      return time_utils::out_of_range();
  }

  // Years are ints.  A 32-bit year will fit into a 64-bit time_t.
  // A 64-bit year will not.
  static_assert(sizeof(int) == 4,
                "ILP64 is unimplemented.  This implementation requires "
                "32-bit integers.");

  // Calculate number of months and years from tm_mon.
  int64_t month = tm_out->tm_mon;
  if (month < 0 || month >= TimeConstants::MONTHS_PER_YEAR - 1) {
    int64_t years = month / 12;
    month %= 12;
    if (month < 0) {
      years--;
      month += 12;
    }
    tm_year_from_base += years;
  }
  bool tm_year_is_leap = is_leap_year(tm_year_from_base);

  // Calculate total number of days based on the month and the day (tm_mday).
  int64_t total_days = tm_out->tm_mday - 1;
  for (int64_t i = 0; i < month; ++i)
    total_days += NON_LEAP_YEAR_DAYS_IN_MONTH[i];
  // Add one day if it is a leap year and the month is after February.
  if (tm_year_is_leap && month > 1)
    total_days++;

  // Calculate total numbers of days based on the year.
  total_days += (tm_year_from_base - TimeConstants::EPOCH_YEAR) *
                TimeConstants::DAYS_PER_NON_LEAP_YEAR;
  if (tm_year_from_base >= TimeConstants::EPOCH_YEAR) {
    total_days += get_num_of_leap_years_before(tm_year_from_base - 1) -
                  get_num_of_leap_years_before(TimeConstants::EPOCH_YEAR);
  } else if (tm_year_from_base >= 1) {
    total_days -= get_num_of_leap_years_before(TimeConstants::EPOCH_YEAR) -
                  get_num_of_leap_years_before(tm_year_from_base - 1);
  } else {
    // Calculate number of leap years until 0th year.
    total_days -= get_num_of_leap_years_before(TimeConstants::EPOCH_YEAR) -
                  get_num_of_leap_years_before(0);
    if (tm_year_from_base <= 0) {
      total_days -= 1; // Subtract 1 for 0th year.
      // Calculate number of leap years until -1 year
      if (tm_year_from_base < 0) {
        total_days -= get_num_of_leap_years_before(-tm_year_from_base) -
                      get_num_of_leap_years_before(1);
      }
    }
  }

  // TODO(rtenneti): Need to handle timezone and update of tm_isdst.
  int64_t seconds = tm_out->tm_sec +
                    tm_out->tm_min * TimeConstants::SECONDS_PER_MIN +
                    tm_out->tm_hour * TimeConstants::SECONDS_PER_HOUR +
                    total_days * TimeConstants::SECONDS_PER_DAY;

  // Update the tm structure's year, month, day, etc. from seconds.
  if (time_utils::update_from_seconds(seconds, tm_out) < 0)
    return time_utils::out_of_range();

  return static_cast<time_t>(seconds);
}

} // namespace __llvm_libc
