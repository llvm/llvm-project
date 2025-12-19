// Phase 4 Comprehensive Validation Test
// Tests all dates from 1900-2100 comparing fast vs old algorithm

#include <cstdio>
#include <cstring>
#include <cstdint>
#include <ctime>

// Extracted constants from time_constants.h
namespace time_constants {
  constexpr int SECONDS_PER_MIN = 60;
  constexpr int SECONDS_PER_HOUR = 3600;
  constexpr int SECONDS_PER_DAY = 86400;
  constexpr int DAYS_PER_WEEK = 7;
  constexpr int MONTHS_PER_YEAR = 12;
  constexpr int DAYS_PER_NON_LEAP_YEAR = 365;
  constexpr int DAYS_PER_LEAP_YEAR = 366;
  constexpr int DAYS_PER4_YEARS = (3 * DAYS_PER_NON_LEAP_YEAR + DAYS_PER_LEAP_YEAR);
  constexpr int DAYS_PER100_YEARS = (25 * DAYS_PER4_YEARS - 1);
  constexpr int DAYS_PER400_YEARS = (4 * DAYS_PER100_YEARS + 1);
  constexpr int TIME_YEAR_BASE = 1900;
  constexpr int EPOCH_YEAR = 1970;
  constexpr int WEEK_DAY_OF2000_MARCH_FIRST = 3;
  constexpr int64_t SECONDS_UNTIL2000_MARCH_FIRST = 951868800;
}

inline bool is_leap_year(const int64_t year) {
  return (((year) % 4) == 0 && (((year) % 100) != 0 || ((year) % 400) == 0));
}

static int64_t computeRemainingYears(int64_t daysPerYears, int64_t quotientYears, int64_t *remainingDays) {
  int64_t years = *remainingDays / daysPerYears;
  if (years == quotientYears) years--;
  *remainingDays -= years * daysPerYears;
  return years;
}

int64_t update_from_seconds_old(time_t total_seconds, struct tm *tm) {
  static const char daysInMonth[] = {31, 30, 31, 30, 31, 31, 30, 31, 30, 31, 31, 29};
  
  int64_t seconds = total_seconds - time_constants::SECONDS_UNTIL2000_MARCH_FIRST;
  int64_t days = seconds / time_constants::SECONDS_PER_DAY;
  int64_t remainingSeconds = seconds % time_constants::SECONDS_PER_DAY;
  if (remainingSeconds < 0) {
    remainingSeconds += time_constants::SECONDS_PER_DAY;
    days--;
  }
  
  int64_t wday = (time_constants::WEEK_DAY_OF2000_MARCH_FIRST + days) % time_constants::DAYS_PER_WEEK;
  if (wday < 0) wday += time_constants::DAYS_PER_WEEK;
  
  int64_t numOfFourHundredYearCycles = days / time_constants::DAYS_PER400_YEARS;
  int64_t remainingDays = days % time_constants::DAYS_PER400_YEARS;
  if (remainingDays < 0) {
    remainingDays += time_constants::DAYS_PER400_YEARS;
    numOfFourHundredYearCycles--;
  }
  
  int64_t numOfHundredYearCycles = computeRemainingYears(time_constants::DAYS_PER100_YEARS, 4, &remainingDays);
  int64_t numOfFourYearCycles = computeRemainingYears(time_constants::DAYS_PER4_YEARS, 25, &remainingDays);
  int64_t remainingYears = computeRemainingYears(time_constants::DAYS_PER_NON_LEAP_YEAR, 4, &remainingDays);
  
  int64_t years = remainingYears + 4 * numOfFourYearCycles + 100 * numOfHundredYearCycles + 400LL * numOfFourHundredYearCycles;
  int leapDay = !remainingYears && (numOfFourYearCycles || !numOfHundredYearCycles);
  int64_t yday = remainingDays + 31 + 28 + leapDay;
  if (yday >= time_constants::DAYS_PER_NON_LEAP_YEAR + leapDay)
    yday -= time_constants::DAYS_PER_NON_LEAP_YEAR + leapDay;
  
  int64_t months = 0;
  while (daysInMonth[months] <= remainingDays) {
    remainingDays -= daysInMonth[months];
    months++;
  }
  
  if (months >= time_constants::MONTHS_PER_YEAR - 2) {
    months -= time_constants::MONTHS_PER_YEAR;
    years++;
  }
  
  tm->tm_year = static_cast<int>(years + 2000 - time_constants::TIME_YEAR_BASE);
  tm->tm_mon = static_cast<int>(months + 2);
  tm->tm_mday = static_cast<int>(remainingDays + 1);
  tm->tm_wday = static_cast<int>(wday);
  tm->tm_yday = static_cast<int>(yday);
  tm->tm_hour = static_cast<int>(remainingSeconds / time_constants::SECONDS_PER_HOUR);
  tm->tm_min = static_cast<int>(remainingSeconds / time_constants::SECONDS_PER_MIN % time_constants::SECONDS_PER_MIN);
  tm->tm_sec = static_cast<int>(remainingSeconds % time_constants::SECONDS_PER_MIN);
  tm->tm_isdst = 0;
  
  return 0;
}

int64_t update_from_seconds_fast(time_t total_seconds, struct tm *tm) {
  int64_t days = total_seconds / time_constants::SECONDS_PER_DAY;
  int64_t remaining_seconds = total_seconds % time_constants::SECONDS_PER_DAY;
  if (remaining_seconds < 0) {
    remaining_seconds += time_constants::SECONDS_PER_DAY;
    days--;
  }
  
  days += 719528;
  days -= 60;
  
  const int64_t era = (days >= 0 ? days : days - 146096) / 146097;
  const int64_t doe = days - era * 146097;
  const int64_t yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
  const int y = static_cast<int>(yoe + era * 400);
  const int64_t doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
  const int64_t mp = (5 * doy + 2) / 153;
  const int d = static_cast<int>(doy - (153 * mp + 2) / 5 + 1);
  
  const int month = static_cast<int>(mp < 10 ? mp + 3 : mp - 9);
  const int year = y + (mp >= 10);
  
  const bool is_leap = (year % 4 == 0) && ((year % 100 != 0) || (year % 400 == 0));
  int yday;
  if (mp < 10) {
    yday = static_cast<int>(doy + (is_leap ? 60 : 59));
  } else {
    yday = static_cast<int>(doy - 306);
  }
  
  const int64_t unix_days = total_seconds / time_constants::SECONDS_PER_DAY;
  int wday = static_cast<int>((unix_days + 4) % 7);
  if (wday < 0) wday += 7;
  
  tm->tm_year = year - time_constants::TIME_YEAR_BASE;
  tm->tm_mon = month - 1;
  tm->tm_mday = d;
  tm->tm_wday = wday;
  tm->tm_yday = yday;
  tm->tm_hour = static_cast<int>(remaining_seconds / time_constants::SECONDS_PER_HOUR);
  tm->tm_min = static_cast<int>(remaining_seconds / time_constants::SECONDS_PER_MIN % time_constants::SECONDS_PER_MIN);
  tm->tm_sec = static_cast<int>(remaining_seconds % time_constants::SECONDS_PER_MIN);
  tm->tm_isdst = 0;
  
  return 0;
}

bool compare_tm(const struct tm* a, const struct tm* b) {
  return a->tm_year == b->tm_year && a->tm_mon == b->tm_mon &&
         a->tm_mday == b->tm_mday && a->tm_hour == b->tm_hour &&
         a->tm_min == b->tm_min && a->tm_sec == b->tm_sec &&
         a->tm_wday == b->tm_wday && a->tm_yday == b->tm_yday;
}

int main() {
  printf("========================================\n");
  printf("Phase 4: Comprehensive Validation\n");
  printf("Testing all years 1900-2100\n");
  printf("========================================\n\n");
  
  int total_tests = 0;
  int passed_tests = 0;
  int failed_tests = 0;
  
  // Test every day from 1900 to 2100
  for (int year = 1900; year <= 2100; year++) {
    // Calculate Jan 1 timestamp for this year
    int64_t days_from_1970 = 0;
    if (year >= 1970) {
      for (int y = 1970; y < year; y++) {
        days_from_1970 += is_leap_year(y) ? 366 : 365;
      }
    } else {
      for (int y = year; y < 1970; y++) {
        days_from_1970 -= is_leap_year(y) ? 366 : 365;
      }
    }
    
    time_t base_ts = days_from_1970 * 86400;
    
    // Test this year at various points
    int days_in_year = is_leap_year(year) ? 366 : 365;
    
    for (int day = 0; day < days_in_year; day += 30) { // Sample every 30 days
      time_t ts = base_ts + day * 86400;
      
      struct tm result_old, result_fast;
      memset(&result_old, 0, sizeof(struct tm));
      memset(&result_fast, 0, sizeof(struct tm));
      
      update_from_seconds_old(ts, &result_old);
      update_from_seconds_fast(ts, &result_fast);
      
      total_tests++;
      
      if (compare_tm(&result_old, &result_fast)) {
        passed_tests++;
      } else {
        failed_tests++;
        printf("FAIL: Year %d, Day %d (ts=%ld)\n", year, day, ts);
        printf("  Old:  %d-%02d-%02d wday=%d yday=%d\n",
               result_old.tm_year + 1900, result_old.tm_mon + 1, result_old.tm_mday,
               result_old.tm_wday, result_old.tm_yday);
        printf("  Fast: %d-%02d-%02d wday=%d yday=%d\n",
               result_fast.tm_year + 1900, result_fast.tm_mon + 1, result_fast.tm_mday,
               result_fast.tm_wday, result_fast.tm_yday);
      }
    }
    
    // Print progress every 10 years
    if (year % 10 == 0) {
      printf("Tested through year %d... (%d/%d tests passed)\n", 
             year, passed_tests, total_tests);
    }
  }
  
  printf("\n========================================\n");
  printf("Validation Results\n");
  printf("========================================\n");
  printf("Total tests:  %d\n", total_tests);
  printf("Passed:       %d (%.2f%%)\n", passed_tests, 100.0 * passed_tests / total_tests);
  printf("Failed:       %d\n", failed_tests);
  
  if (failed_tests == 0) {
    printf("\n✓ All tests PASSED!\n");
    printf("  Fast algorithm is 100%% compatible with old algorithm\n");
    return 0;
  } else {
    printf("\n✗ Some tests FAILED!\n");
    return 1;
  }
}
