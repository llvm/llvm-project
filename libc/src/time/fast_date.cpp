//===-- Fast date conversion implementation -------------------------------===//
//
// Implementation of Ben Joffe's "Century-February-Padding" algorithm
// Reference: https://www.benjoffe.com/fast-date
//
//===----------------------------------------------------------------------===//

#include "fast_date.h"

namespace fast_date {

// Core Joffe algorithm: Convert days since 0000-01-01 to year/month/day
void days_to_ymd_joffe(int64_t days, int &year, int &month, int &day, int &yday) {
  // Based on Howard Hinnant's civil_from_days
  // Shift to March-based year (makes Feb the last month)
  days -= MARCH_SHIFT_DAYS;  // Shift from 0000-01-01 to 0000-03-01 (60 days: 31 Jan + 29 Feb)
  
  const int64_t era = (days >= 0 ? days : days - (DAYS_PER_ERA - 1)) / DAYS_PER_ERA;
  const int64_t doe = days - era * DAYS_PER_ERA;  // day of era [0, 146096]
  const int64_t yoe = (doe - doe / DAYS_PER_4_YEARS + doe / DAYS_PER_CENTURY - doe / DAYS_PER_ERA) / 365;  // year of era [0, 399]
  const int y = static_cast<int>(yoe + era * YEARS_PER_ERA);
  const int64_t doy = doe - (365 * yoe + yoe / 4 - yoe / YEARS_PER_CENTURY);  // day of year [0, 365]
  const int64_t mp = (MONTH_CYCLE_MONTHS * doy + 2) / MONTH_CYCLE_DAYS;  // month [0, 11]
  const int d = static_cast<int>(doy - (MONTH_CYCLE_DAYS * mp + 2) / MONTH_CYCLE_MONTHS + 1);  // day [1, 31]
  
  month = static_cast<int>(mp < 10 ? mp + 3 : mp - 9);
  year = y + (mp >= 10);
  day = d;
  
  // Calculate yday (0-indexed from Jan 1)
  const bool is_leap = (year % 4 == 0) && ((year % YEARS_PER_CENTURY != 0) || (year % YEARS_PER_ERA == 0));
  if (mp < 10) {
    yday = static_cast<int>(doy + (is_leap ? MARCH_SHIFT_DAYS : MARCH_SHIFT_DAYS - 1));
  } else {
    yday = static_cast<int>(doy - DAYS_BEFORE_MARCH);
  }
}

// Optimized inverse: Convert year/month/day to days since 0000-01-01
int64_t ymd_to_days_joffe(int year, int month, int day) {
  // Based on Howard Hinnant's days_from_civil algorithm
  // Adjust to March-based year
  year -= (month <= 2);
  
  // Calculate era (400-year periods)
  int64_t era = (year >= 0 ? year : year - (YEARS_PER_ERA - 1)) / YEARS_PER_ERA;
  int64_t yoe = year - era * YEARS_PER_ERA;  // year of era [0, 399]
  
  // Day of year, with March 1 = 0
  int64_t doy = (MONTH_CYCLE_DAYS * (month + (month > 2 ? -3 : 9)) + 2) / MONTH_CYCLE_MONTHS + day - 1;
  
  // Day of era
  int64_t doe = yoe * 365 + yoe / 4 - yoe / YEARS_PER_CENTURY + doy;
  
  // Days since March 1, year 0
  int64_t days_since_march_1 = era * DAYS_PER_ERA + doe;
  
  // Adjust to Jan 1, year 0 epoch
  // days_since_march_1 is from 0000-03-01, add 60 to get from 0000-01-01
  return days_since_march_1 + MARCH_SHIFT_DAYS;
}

// Main function: Convert Unix timestamp to date
DateResult unix_to_date_fast(int64_t timestamp) {
  DateResult result = {0};
  
  // Calculate days and remaining seconds
  int64_t days = timestamp / SECONDS_PER_DAY;
  int64_t remaining = timestamp % SECONDS_PER_DAY;
  
  // Handle negative remainders
  if (remaining < 0) {
    remaining += SECONDS_PER_DAY;
    days--;
  }
  
  // Convert Unix days to days since 0000-01-01
  // 1970-01-01 is 719162 days after 0001-01-01 (Rata Die)
  // Plus 366 days for year 0 (leap year in proleptic Gregorian)
  // = 719528 days since 0000-01-01
  days += UNIX_EPOCH_DAYS;
  
  // Use Joffe algorithm to get year/month/day
  days_to_ymd_joffe(days, result.year, result.month, result.day, result.yday);
  
  // Calculate time components
  result.hour = static_cast<int>(remaining / SECONDS_PER_HOUR);
  remaining %= SECONDS_PER_HOUR;
  result.minute = static_cast<int>(remaining / SECONDS_PER_MINUTE);
  result.second = static_cast<int>(remaining % SECONDS_PER_MINUTE);
  
  // Calculate day of week
  // Unix epoch (1970-01-01) was a Thursday (4)
  int64_t total_days = timestamp / SECONDS_PER_DAY;
  result.wday = static_cast<int>((total_days + UNIX_EPOCH_WDAY) % 7);
  if (result.wday < 0) result.wday += 7;
  
  result.valid = true;
  return result;
}

// Inverse function: Convert date to Unix timestamp
int64_t date_to_unix_fast(int year, int month, int day, 
                          int hour, int minute, int second) {
  // Validate inputs
  if (month < 1 || month > 12) return -1;
  if (day < 1 || day > 31) return -1;
  if (hour < 0 || hour > 23) return -1;
  if (minute < 0 || minute > 59) return -1;
  if (second < 0 || second > 59) return -1;
  
  // Convert to days since 0000-01-01
  int64_t days = ymd_to_days_joffe(year, month, day);
  
  // Adjust to Unix epoch
  days -= UNIX_EPOCH_DAYS;
  
  // Convert to seconds
  int64_t total_seconds = days * SECONDS_PER_DAY;
  total_seconds += hour * SECONDS_PER_HOUR;
  total_seconds += minute * SECONDS_PER_MINUTE;
  total_seconds += second;
  
  return total_seconds;
}

} // namespace fast_date
