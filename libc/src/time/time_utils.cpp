//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Implementation of internal time utility functions.
///
//===----------------------------------------------------------------------===//

#include "src/time/time_utils.h"
#include "hdr/stdint_proxy.h"
#include "src/__support/CPP/limits.h" // INT_MIN, INT_MAX
#include "src/__support/common.h"
#include "src/__support/libc_assert.h"
#include "src/__support/macros/config.h"
#include "src/__support/uint128.h"
#include "src/time/time_constants.h"

namespace LIBC_NAMESPACE_DECL {
namespace time_utils {

cpp::optional<time_t> mktime_internal(const tm *tm_out) {
  // Unlike most C Library functions, mktime doesn't just die on bad input.
  // TODO(rtenneti); Handle leap seconds.
  // POSIX §4.16: each day is accounted for by exactly 86400 seconds.

  // Normalize year and month from tm_mon.
  int64_t total_months = tm_out->tm_mon;
  int64_t year =
      tm_out->tm_year + static_cast<int64_t>(time_constants::TIME_YEAR_BASE);
  int64_t month = total_months % 12;
  year += total_months / 12;
  if (month < 0) {
    month += 12;
    year--;
  }
  month += 1; // 1-12 range
  int64_t day = tm_out->tm_mday;

  // Inverse date algorithm from Ben Joffe, "The Julian Map" (Nov 2025).
  // https://www.benjoffe.com/fast-date#inverse
  //
  // Calculates the number of days since 1970-01-01 (Unix epoch).
  //
  // Key constants:
  //   S = 14700: biases years positive to avoid negative division.
  //     YEAR_SHIFT = 400*S, RATA_SHIFT = 719468 + 146097*S + 1
  //     where 719468 = days from 0000-02-29 to 1970-01-01.
  //   979/32: Neri-Schneider EAF approximation for cumulative month days.
  //   phase (-2919 or 8829): month offset selecting March-start vs
  //     January-start, depending on whether month <= 2 ("bump").
  constexpr int64_t S = 14700;
  constexpr int64_t YEAR_SHIFT = 400 * S;
  constexpr int64_t RATA_SHIFT = 719468 + 146097 * S + 1;

  bool bump = (month <= 2);
  int64_t y = year + YEAR_SHIFT - bump;
  int64_t cent = y / 100;
  int64_t phase = bump ? 8829 : -2919;

  int64_t y_days = y * 365 + (y / 4) - cent + (cent / 4);
  int64_t m_days = (979 * month + phase) / 32;
  int64_t total_days = y_days + m_days + day - RATA_SHIFT;

  // TODO: https://github.com/llvm/llvm-project/issues/121962
  // Need to handle timezone and update of tm_isdst.
  time_t seconds = static_cast<time_t>(
      tm_out->tm_sec + tm_out->tm_min * time_constants::SECONDS_PER_MIN +
      tm_out->tm_hour * time_constants::SECONDS_PER_HOUR +
      total_days * time_constants::SECONDS_PER_DAY);
  return seconds;
}

// Update the tm structure's year, month, etc. members from seconds.
// total_seconds is the number of seconds since January 1st, 1970.
//
// Uses Ben Joffe's "Very Fast 64-Bit" date algorithm (Article 3, Nov 2025).
// https://www.benjoffe.com/fast-date-64
//
// The Neri-Schneider "EAF" technique is used for month/day determination:
// https://onlinelibrary.wiley.com/doi/full/10.1002/spe.3172
//
// This uses the proleptic Gregorian calendar: Gregorian leap-year rules are
// extended to all dates, including those before the calendar's adoption in
// 1582.
ErrorOr<int> update_from_seconds(time_t total_seconds, tm *tm) {
  // Range check for valid time_t values
  constexpr time_t time_min =
      INT_MIN *
      static_cast<int64_t>(time_constants::NUMBER_OF_SECONDS_IN_LEAP_YEAR);
  constexpr time_t time_max =
      INT_MAX *
      static_cast<int64_t>(time_constants::NUMBER_OF_SECONDS_IN_LEAP_YEAR);

  if (total_seconds < time_min || total_seconds > time_max)
    return cpp::unexpected(TIME_OVERFLOW);

  // Step 1: Convert seconds to days + remaining seconds
  // Handle negative timestamps correctly (before Unix epoch)
  int64_t days = total_seconds / time_constants::SECONDS_PER_DAY;
  int64_t remaining_seconds = total_seconds % time_constants::SECONDS_PER_DAY;
  if (remaining_seconds < 0) {
    remaining_seconds += time_constants::SECONDS_PER_DAY;
    days--;
  }

  // Save Unix epoch days for wday calculation later
  const int64_t unix_days = days;

  // See pseudocode lines 1-29 at https://www.benjoffe.com/fast-date-64
  //
  // Key idea: count years backwards from a far-future epoch so that both
  // the 4-year and 400-year cycles start with a leap/long period. This
  // eliminates the "+3" offset terms from traditional algorithms and
  // enables pure multiply-shift division (4 multiplications, 0 hardware
  // divisions).

  // ERAS: number of 400-year eras to shift into the future; chosen to
  // maximize the symmetric range around the Unix epoch in 64-bit.
  constexpr int64_t ERAS = 4726498270LL;
  // D_SHIFT: reversed day count from the epoch alignment point 0000-02-29.
  // 146097 = days per 400-year era; 719469 = days from 0000-02-29 to
  // 1970-01-01 (one day earlier than the 719468 used by forward algorithms).
  constexpr int64_t D_SHIFT = 146097LL * ERAS - 719469LL;
  // Y_SHIFT: converts reversed year count back to a forward year.
  constexpr int64_t Y_SHIFT = 400LL * ERAS - 1;
  // C1-C3: fixed-point reciprocals for multiply-shift division.
  // The >>64 bit-shift is "free" on 64-bit CPUs (just reads the high
  // register of the 128-bit multiplication result).
  constexpr uint64_t C1 = 505054698555331ULL;   // floor(2^64 * 4 / 146097)
  constexpr uint64_t C2 = 50504432782230121ULL; // ceil(2^64 * 4 / 1461)
  constexpr uint64_t C3 = 8619973866219416ULL;  // floor(2^64 / 2140)

  // Pseudocode lines 9-11: Adjust for 100/400 leap year rule (Julian Map).
  int64_t rev = D_SHIFT - unix_days;
  int64_t cen = static_cast<int64_t>((static_cast<UInt128>(rev) * C1) >> 64);
  int64_t jul = rev + cen - (cen / 4);

  // Pseudocode lines 14-17: Determine year and year-part.
  UInt128 num = static_cast<UInt128>(jul) * C2;
  int64_t yrs = Y_SHIFT - static_cast<int64_t>(num >> 64);
  uint64_t low = static_cast<uint64_t>(num);
  // 782432 scales the fractional year into a "year-part" (ypt) that
  // encodes day-of-year, deliberately skipping the explicit day-of-year
  // step from Neri-Schneider and merging it into ypt.
  uint64_t ypt =
      static_cast<uint64_t>((static_cast<UInt128>(low) * 782432ULL) >> 64);

  // Pseudocode lines 19-20: Detect Jan/Feb and select month offset.
  // 126464: ypt threshold for Jan/Feb (lowest values in the reversed,
  // March-based computational year).
  bool bump = ypt < 126464ULL;
  // 191360 and 977792 differ by exactly 12 * 2^16 = 786432, shifting
  // the month by 12 without a conditional subtraction.
  int64_t shift = bump ? 191360LL : 977792LL;

  // Pseudocode lines 24-25: Year-modulo-bitshift for leap years.
  // N packs month (high 16 bits) and day-part (low 16 bits).
  // (yrs % 4) * 512 corrects a 1/4-day-per-year drift introduced by
  // skipping the explicit day-of-year step. 512 = 2^16 / 32 / 4, i.e.
  // one quarter of a "fake 32-day month" in 16-bit space.
  int64_t n_val = (yrs % 4) * 512 + shift - static_cast<int64_t>(ypt);
  int64_t d_val =
      static_cast<int64_t>((static_cast<UInt128>(n_val & 65535) * C3) >> 64);

  const int d = static_cast<int>(d_val + 1);
  const int month = static_cast<int>(n_val >> 16);
  const int64_t year_full = yrs + (bump ? 1 : 0);

  if (year_full > INT_MAX || year_full < INT_MIN)
    return cpp::unexpected(TIME_OVERFLOW);
  const int year = static_cast<int>(year_full);

  // Step 4: Calculate day of year (yday) in January-based calendar [0, 365]
  // Use cumulative-days lookup table for O(1) computation instead of a loop.
  LIBC_ASSERT(month >= 1 && month <= 12);
  const bool is_leap = time_utils::is_leap_year(year);
  int yday = time_constants::CUMULATIVE_DAYS_BEFORE_MONTH[month] + d - 1;
  if (is_leap && month > 2)
    yday++;

  // Step 5: Calculate day of week [0=Sun, 1=Mon, ..., 6=Sat]
  // Unix epoch 1970-01-01 was Thursday (4)
  int wday = static_cast<int>((unix_days + 4) % 7);
  if (wday < 0)
    wday += 7;

  // Step 6: Populate tm structure with all calculated values
  tm->tm_year = year - time_constants::TIME_YEAR_BASE; // Years since 1900
  tm->tm_mon = month - 1;                              // Months [0, 11]
  tm->tm_mday = d;                                     // Day of month [1, 31]
  tm->tm_wday = wday;                                  // Day of week [0, 6]
  tm->tm_yday = yday;                                  // Day of year [0, 365]

  // Calculate time components from remaining seconds
  tm->tm_hour =
      static_cast<int>(remaining_seconds / time_constants::SECONDS_PER_HOUR);
  tm->tm_min =
      static_cast<int>(remaining_seconds / time_constants::SECONDS_PER_MIN %
                       time_constants::SECONDS_PER_MIN);
  tm->tm_sec =
      static_cast<int>(remaining_seconds % time_constants::SECONDS_PER_MIN);
  tm->tm_isdst = 0; // Daylight saving time flag (not implemented)

  return 0;
}

} // namespace time_utils
} // namespace LIBC_NAMESPACE_DECL
