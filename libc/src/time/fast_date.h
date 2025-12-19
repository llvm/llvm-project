//===-- Fast date conversion using Joffe algorithm ---------------*- C++ -*-===//
//
// Implementation of Ben Joffe's "Century-February-Padding" algorithm
// Reference: https://www.benjoffe.com/fast-date
//
// This algorithm achieves 2-11% performance improvement over traditional
// date conversion by mapping the Gregorian calendar to Julian calendar
// (by padding with fake Feb 29s every 100 years except 400 years).
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_TIME_FAST_DATE_H
#define LLVM_LIBC_SRC_TIME_FAST_DATE_H

#include <cstdint>

namespace fast_date {

// Result structure for date conversion
struct DateResult {
  int year;
  int month;  // 1-12 (January = 1)
  int day;    // 1-31
  int yday;   // Day of year (0-365, Jan 1 = 0)
  int wday;   // Day of week (0-6, Sunday = 0)
  int hour;
  int minute;
  int second;
  bool valid; // false if date is out of range
};

// Convert Unix timestamp (seconds since 1970-01-01 00:00:00 UTC) to date
// This is the fast algorithm using Century-February-Padding technique
DateResult unix_to_date_fast(int64_t timestamp);

// Convert year/month/day to days since epoch (inverse function)
// Returns -1 if date is invalid
int64_t date_to_unix_fast(int year, int month, int day, 
                          int hour = 0, int minute = 0, int second = 0);

// Helper: Convert days since epoch (Jan 1, 0000 = 0) to year/month/day
// This is the core Joffe algorithm
void days_to_ymd_joffe(int64_t days, int &year, int &month, int &day, int &yday);

// Helper: Convert year/month/day to days since epoch (0000-01-01 = 0)
// This is the optimized inverse from the article
int64_t ymd_to_days_joffe(int year, int month, int day);

// Constants
constexpr int64_t SECONDS_PER_DAY = 86400;
constexpr int64_t SECONDS_PER_HOUR = 3600;
constexpr int64_t SECONDS_PER_MINUTE = 60;

// Unix epoch (1970-01-01) as days since 0000-01-01
// Calculated as: 719162 (Rata Die for 1970-01-01) + 366 (year 0 is leap year)
constexpr int64_t UNIX_EPOCH_DAYS = 719528;

// Shift from Jan 1 to March 1 (31 Jan + 29 Feb in leap year 0)
constexpr int64_t MARCH_SHIFT_DAYS = 60;

// Gregorian calendar cycle constants
constexpr int64_t DAYS_PER_ERA = 146097;     // Days in 400-year cycle
constexpr int64_t DAYS_PER_CENTURY = 36524;  // Days in 100-year cycle (non-leap)
constexpr int64_t DAYS_PER_4_YEARS = 1461;   // Days in 4-year cycle (with leap)
constexpr int64_t YEARS_PER_ERA = 400;
constexpr int64_t YEARS_PER_CENTURY = 100;

// Magic constants for month calculation (based on 153-day 5-month cycles)
constexpr int64_t MONTH_CYCLE_DAYS = 153;
constexpr int64_t MONTH_CYCLE_MONTHS = 5;

// Day of week constant (Unix epoch was Thursday)
constexpr int UNIX_EPOCH_WDAY = 4;

// Days before March 1 in a March-based year
constexpr int DAYS_BEFORE_MARCH = 306;

} // namespace fast_date

#endif // LLVM_LIBC_SRC_TIME_FAST_DATE_H
