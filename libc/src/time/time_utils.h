//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Collection of utils for mktime and friends.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_TIME_TIME_UTILS_H
#define LLVM_LIBC_SRC_TIME_TIME_UTILS_H

#include "hdr/errno_macros.h"
#include "hdr/stdint_proxy.h"
#include "hdr/types/size_t.h"
#include "hdr/types/struct_tm.h"
#include "hdr/types/time_t.h"
#include "src/__support/CPP/optional.h"
#include "src/__support/CPP/string_view.h"
#include "src/__support/common.h"
#include "src/__support/error_or.h"
#include "src/__support/macros/config.h"
#include "src/time/time_constants.h"

namespace LIBC_NAMESPACE_DECL {
namespace time_utils {

/// Calculates the seconds from the epoch for tm_in.
/// Does not update the struct, call update_from_seconds for that.
///
/// \param tm_out Pointer to the tm structure to read.
/// \return The epoch time on success, or nullopt on failure.
cpp::optional<time_t> mktime_internal(const tm *tm_out);

// EOVERFLOW is a POSIX extension and might not be defined on targets that only
// support the C standard (e.g. generic baremetal). The C standard does not
// require setting errno on overflow for time functions, but POSIX does.
// We map it to ERANGE on non-POSIX targets to provide consistent error
// reporting while remaining compliant.
#ifdef EOVERFLOW
constexpr int TIME_OVERFLOW = EOVERFLOW;
#else
constexpr int TIME_OVERFLOW = ERANGE;
#endif

/// Update the "tm" structure's year, month, etc. members from seconds.
///
/// \param total_seconds The number of seconds since January 1st, 1970.
/// \param tm Pointer to the tm structure to update.
/// \return void on success, or error code on failure.
ErrorOr<void> update_from_seconds(time_t total_seconds, tm *tm);

LIBC_INLINE ErrorOr<tm *> gmtime_internal(const time_t *timer, tm *result) {
  time_t seconds = *timer;
  auto status = update_from_seconds(seconds, result);
  if (!status) {
    return cpp::unexpected(status.error());
  }

#if defined(__linux__)
  result->tm_gmtoff = 0;
  result->tm_zone = "GMT";
#endif

  return result;
}

LIBC_INLINE ErrorOr<tm *> localtime_internal(const time_t *timer, tm *result) {
  time_t seconds = *timer;
  auto status = update_from_seconds(seconds, result);
  if (!status) {
    return cpp::unexpected(status.error());
  }

  // TODO(zimirza): implement timezone database
#if defined(__linux__)
  result->tm_gmtoff = 0;
  result->tm_zone = "UTC";
#endif

  return result;
}

LIBC_INLINE ErrorOr<tm *> localtime(const time_t *t_ptr) {
  if (t_ptr == nullptr)
    return cpp::unexpected(EINVAL);
  static tm result;
  return time_utils::localtime_internal(t_ptr, &result);
}

// Returns True if year is a leap year.
// Uses the Drepper-Neri-Schneider algorithm (version 3).
// https://www.benjoffe.com/fast-leap-year#drepper-neri-schneider
//
// % 25 is faster than % 100 on modern compilers. False positives (years
// divisible by 25 but not 100) are harmless: the & 15 check only differs
// from & 3 when the year truly is a century year, and non-century years
// that happen to be divisible by 25 always pass both checks identically.
LIBC_INLINE constexpr bool is_leap_year(const int64_t year) {
  const bool is_mul_25 = (year % 25 == 0);
  return (year & (is_mul_25 ? 15 : 3)) == 0;
}

LIBC_INLINE constexpr int get_days_in_year(const int year) {
  return is_leap_year(year) ? time_constants::DAYS_PER_LEAP_YEAR
                            : time_constants::DAYS_PER_NON_LEAP_YEAR;
}

// Returns the day of the week [0=Sun, 1=Mon, ..., 6=Sat] for a count of days
// since the Unix epoch (1970-01-01, a Thursday), i.e. (unix_days + 4) mod 7
// with a non-negative result.
//
// Uses Ben Joffe's "Unreasonably Fast Mul-Add-Shift" algorithm (Article 5).
// https://www.benjoffe.com/fast-day-of-week
//
// Since 7 = 2^3 - 1 is a Mersenne number, x mod 7 == floor(x * 8 / 7) mod 8.
// Multiplying by M = ceil(2^64 / 7) and keeping the low 64 bits places
// floor(x * 8 / 7) mod 8 in the top 3 bits; the final mod 8 is free. Adding Z
// before the shift rotates the result so that day 0 maps to Thursday (4).
//
// Valid for unix_days in [-230471710930685136, 230696890912053658], which is
// about +/-631 trillion years and far exceeds the range accepted by
// update_from_seconds (about +/-7.9e11 days).
LIBC_INLINE constexpr int get_weekday(const int64_t unix_days) {
  // M = ceil(2^64 / 7).
  constexpr uint64_t M = (UINT64_MAX / time_constants::DAYS_PER_WEEK) + 1;
  // Z rotates the result so that day 0 maps to Thursday: its top 3 bits
  // (0x9492 >> 13 == 4) encode Thursday, and the remaining low bits are a
  // rounding bias (from Joffe's article) that positions the window of exact
  // results to give the valid range documented above.
  constexpr uint64_t Z = 0x9492ULL << 48;
  return static_cast<int>((static_cast<uint64_t>(unix_days) * M + Z) >> 61);
}

// Hour, minute and second within a day.
struct TimeOfDay {
  // Hour of the day [0, 23].
  int hour;
  // Minute of the hour [0, 59].
  int minute;
  // Second of the minute [0, 59].
  int second;
};

// Splits the number of seconds elapsed within a day [0, 86399] into hours,
// minutes and seconds.
//
// Uses Ben Joffe's "V2 - Fixed-Point Hi/Low Bits" algorithm (Article 6).
// https://www.benjoffe.com/fast-time-of-day
//
// Instead of the sequential chain hour -> remainder -> minute -> second, both
// fixed-point products are computed directly from seconds_of_day so they can
// run in parallel:
//   - The high 32 bits of seconds_of_day * ceil(2^32 / 3600) are the hour,
//     and the low 32 bits are the fraction of the current hour elapsed;
//     scaling that fraction by 60 yields the minute.
//   - The low 32 bits of seconds_of_day * ceil(2^32 / 60) are the fraction of
//     the current minute elapsed; scaling it by 60 yields the second.
//
// This compiles to four multiplications (two pairs that can overlap) and no
// division.
// Valid for seconds_of_day in [0, 2255818] (up to 626:36:58), which covers a
// full day with a wide margin.
LIBC_INLINE constexpr TimeOfDay get_time_of_day(const uint32_t seconds_of_day) {
  // H_MUL = ceil(2^32 / 3600) = 1193047.
  constexpr uint64_t H_MUL =
      ((1ULL << 32) / time_constants::SECONDS_PER_HOUR) + 1;
  // M_MUL = ceil(2^32 / 60) = 71582789.
  constexpr uint64_t M_MUL =
      ((1ULL << 32) / time_constants::SECONDS_PER_MIN) + 1;
  const uint64_t HOUR_PRODUCT = static_cast<uint64_t>(seconds_of_day) * H_MUL;
  const uint32_t MINUTE_FRACTION = static_cast<uint32_t>(HOUR_PRODUCT);
  const uint32_t SECOND_FRACTION =
      static_cast<uint32_t>(seconds_of_day * M_MUL);
  return {static_cast<int>(HOUR_PRODUCT >> 32),
          static_cast<int>((static_cast<uint64_t>(MINUTE_FRACTION) *
                            time_constants::MINUTES_PER_HOUR) >>
                           32),
          static_cast<int>((static_cast<uint64_t>(SECOND_FRACTION) *
                            time_constants::SECONDS_PER_MIN) >>
                           32)};
}

// This is a helper class that takes a struct tm and lets you inspect its
// values. Where relevant, results are bounds checked and returned as optionals.
/// Class to read fields from tm structure.
///
/// This class does not, however, do data normalization except where necessary.
/// It will faithfully return a date of 9999-99-99, even though that makes no
/// sense.
class TMReader final {
  const tm *timeptr;

  template <size_t N>
  LIBC_INLINE constexpr cpp::optional<cpp::string_view>
  bounds_check(const cpp::array<cpp::string_view, N> &arr, int index) const {
    if (index >= 0 && index < static_cast<int>(arr.size()))
      return arr[index];
    return cpp::nullopt;
  }

public:
  LIBC_INLINE constexpr explicit TMReader(const tm *tmptr) : timeptr(tmptr) {}

  // Strings
  LIBC_INLINE constexpr cpp::optional<cpp::string_view>
  get_weekday_short_name() const {
    return bounds_check(time_constants::WEEK_DAY_NAMES, timeptr->tm_wday);
  }

  LIBC_INLINE constexpr cpp::optional<cpp::string_view>
  get_weekday_full_name() const {
    return bounds_check(time_constants::WEEK_DAY_FULL_NAMES, timeptr->tm_wday);
  }

  LIBC_INLINE constexpr cpp::optional<cpp::string_view>
  get_month_short_name() const {
    return bounds_check(time_constants::MONTH_NAMES, timeptr->tm_mon);
  }

  LIBC_INLINE constexpr cpp::optional<cpp::string_view>
  get_month_full_name() const {
    return bounds_check(time_constants::MONTH_FULL_NAMES, timeptr->tm_mon);
  }

  LIBC_INLINE constexpr cpp::string_view get_am_pm() const {
    if (timeptr->tm_hour < 12)
      return "AM";
    return "PM";
  }

  /// Get timezone name.
  ///
  /// \return Timezone name string, or empty string if not set.
  LIBC_INLINE constexpr cpp::string_view get_timezone_name() const {
#if defined(__linux__)
    return timeptr->tm_zone ? timeptr->tm_zone : "";
#else
    return "UTC";
#endif
  }

  // Numbers
  LIBC_INLINE constexpr int get_sec() const { return timeptr->tm_sec; }
  LIBC_INLINE constexpr int get_min() const { return timeptr->tm_min; }
  LIBC_INLINE constexpr int get_hour() const { return timeptr->tm_hour; }
  LIBC_INLINE constexpr int get_mday() const { return timeptr->tm_mday; }
  LIBC_INLINE constexpr int get_mon() const { return timeptr->tm_mon; }
  LIBC_INLINE constexpr int get_yday() const { return timeptr->tm_yday; }
  LIBC_INLINE constexpr int get_wday() const { return timeptr->tm_wday; }
  LIBC_INLINE constexpr int get_isdst() const { return timeptr->tm_isdst; }

  // returns the year, counting from 1900
  LIBC_INLINE constexpr int get_year_raw() const { return timeptr->tm_year; }
  // returns the year, counting from 0
  LIBC_INLINE constexpr int get_year() const {
    return timeptr->tm_year + time_constants::TIME_YEAR_BASE;
  }

  LIBC_INLINE constexpr int is_leap_year() const {
    return time_utils::is_leap_year(get_year());
  }

  LIBC_INLINE constexpr int get_iso_wday() const {
    using time_constants::DAYS_PER_WEEK;
    using time_constants::MONDAY;
    // ISO uses a week that starts on Monday, but struct tm starts its week on
    // Sunday. This function normalizes the weekday so that it always returns a
    // value 0-6
    const int NORMALIZED_WDAY = timeptr->tm_wday % DAYS_PER_WEEK;
    return (NORMALIZED_WDAY + (DAYS_PER_WEEK - MONDAY)) % DAYS_PER_WEEK;
  }

  // returns the week of the current year, with weeks starting on start_day.
  LIBC_INLINE constexpr int get_week(time_constants::WeekDay start_day) const {
    using time_constants::DAYS_PER_WEEK;
    // The most recent start_day. The rest of the days into the current week
    // don't count, so ignore them.
    // Also add 7 to handle start_day > tm_wday
    const int start_of_cur_week =
        timeptr->tm_yday -
        ((timeptr->tm_wday + DAYS_PER_WEEK - start_day) % DAYS_PER_WEEK);

    // The original formula is ceil((start_of_cur_week + 1) / DAYS_PER_WEEK)
    // That becomes (start_of_cur_week + 1 + DAYS_PER_WEEK - 1) / DAYS_PER_WEEK)
    // Which simplifies to (start_of_cur_week + DAYS_PER_WEEK) / DAYS_PER_WEEK
    const int ceil_weeks_since_start =
        (start_of_cur_week + DAYS_PER_WEEK) / DAYS_PER_WEEK;

    return ceil_weeks_since_start;
  }

  LIBC_INLINE constexpr int get_iso_week() const {
    using time_constants::DAYS_PER_WEEK;
    using time_constants::ISO_FIRST_DAY_OF_YEAR;
    using time_constants::MONDAY;
    using time_constants::WeekDay;
    using time_constants::WEEKS_PER_YEAR;

    constexpr WeekDay START_DAY = MONDAY;

    // The most recent start_day. The rest of the days into the current week
    // don't count, so ignore them.
    // Also add 7 to handle start_day > tm_wday
    const int start_of_cur_week =
        timeptr->tm_yday -
        ((timeptr->tm_wday + DAYS_PER_WEEK - START_DAY) % DAYS_PER_WEEK);

    // if the week starts in the previous year, and also if the 4th of this year
    // is not in this week.
    if (start_of_cur_week < -3) {
      const int days_into_prev_year =
          get_days_in_year(get_year() - 1) + start_of_cur_week;
      // Each year has at least 52 weeks, but a year's last week will be 53 if
      // its first week starts in the previous year and its last week ends
      // in the next year. We know get_year() - 1 must extend into get_year(),
      // so here we check if it also extended into get_year() - 2 and add 1 week
      // if it does.
      return WEEKS_PER_YEAR +
             ((days_into_prev_year % DAYS_PER_WEEK) > ISO_FIRST_DAY_OF_YEAR);
    }

    // subtract 1 to account for yday being 0 indexed
    const int days_until_end_of_year =
        get_days_in_year(get_year()) - start_of_cur_week - 1;

    // if there are less than 3 days from the start of this week to the end of
    // the year, then there must be 4 days in this week in the next year, which
    // means that this week is the first week of that year.
    if (days_until_end_of_year < 3)
      return 1;

    // else just calculate the current week like normal.
    const int ceil_weeks_since_start =
        (start_of_cur_week + DAYS_PER_WEEK) / DAYS_PER_WEEK;

    // add 1 if this year's first week starts in the previous year.
    const int WEEK_STARTS_IN_PREV_YEAR =
        ((start_of_cur_week + time_constants::DAYS_PER_WEEK) %
         time_constants::DAYS_PER_WEEK) > time_constants::ISO_FIRST_DAY_OF_YEAR;
    return ceil_weeks_since_start + WEEK_STARTS_IN_PREV_YEAR;
  }

  LIBC_INLINE constexpr int get_iso_year() const {
    const int BASE_YEAR = get_year();
    // The ISO year is the same as a standard year for all dates after the start
    // of the first week and before the last week. Since the first ISO week of a
    // year starts on the 4th, anything after that is in this year.
    if (timeptr->tm_yday >= time_constants::ISO_FIRST_DAY_OF_YEAR &&
        timeptr->tm_yday < time_constants::DAYS_PER_NON_LEAP_YEAR -
                               time_constants::DAYS_PER_WEEK)
      return BASE_YEAR;

    const int ISO_WDAY = get_iso_wday();
    // The first week of the ISO year is defined as the week containing the
    // 4th day of January.

    // first week
    if (timeptr->tm_yday < time_constants::ISO_FIRST_DAY_OF_YEAR) {
      /*
      If jan 4 is in this week, then we're in BASE_YEAR, else we're in the
      previous year. The formula's been rearranged so here's the derivation:

              +--------+-- days until jan 4
              |        |
       wday + (4 - yday) < 7
       |               |
       +---------------+-- weekday of jan 4

       rearranged to get all the constants on one side:

       wday - yday < 7 - 4
      */
      const int IS_CUR_YEAR = (ISO_WDAY - timeptr->tm_yday <
                               time_constants::DAYS_PER_WEEK -
                                   time_constants::ISO_FIRST_DAY_OF_YEAR);
      return BASE_YEAR - !IS_CUR_YEAR;
    }

    // last week
    const int DAYS_LEFT_IN_YEAR =
        get_days_in_year(get_year()) - timeptr->tm_yday;
    /*
    Similar to above, we're checking if jan 4 (of next year) is in this week. If
    it is, this is in the next year. Note that this also handles the case of
    yday > days in year gracefully.

           +------------------+-- days until jan 4 (of next year)
           |                  |
    wday + (4 + remaining days) < 7
    |                         |
    +-------------------------+-- weekday of jan 4

    rearranging we get:

    wday + remaining days < 7 - 4
    */
    const int IS_NEXT_YEAR =
        (ISO_WDAY + DAYS_LEFT_IN_YEAR <
         time_constants::DAYS_PER_WEEK - time_constants::ISO_FIRST_DAY_OF_YEAR);
    return BASE_YEAR + IS_NEXT_YEAR;
  }

  /// Get epoch time.
  ///
  /// \return Epoch time in seconds on success, or TIME_OVERFLOW on failure.
  LIBC_INLINE ErrorOr<time_t> get_epoch() const {
    auto seconds = mktime_internal(timeptr);
    if (!seconds)
      return cpp::unexpected(TIME_OVERFLOW);
    return *seconds;
  }

  /// Get timezone offset.
  ///
  /// The offset is returned in microwave time: (hours * 100) + minutes.
  /// For example, -04:30 is returned as -430.
  ///
  /// \return Timezone offset in microwave time.
  LIBC_INLINE constexpr int get_timezone_offset() const {
#if defined(__linux__)
    // TODO: This relies on tm_gmtoff which is currently initialized to 0
    // by localtime/gmtime until timezone database is implemented.
    int64_t seconds = timeptr->tm_gmtoff;
    int64_t hours = seconds / 3600;
    int64_t minutes = (seconds % 3600) / 60;
    return static_cast<int>((hours * 100) + minutes);
#else
    return 0;
#endif
  }
};

} // namespace time_utils
} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC_TIME_TIME_UTILS_H
