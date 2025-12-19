//===-- Unit tests for fast date algorithm -------------------------------===//
//
// Comprehensive tests for the Joffe fast date conversion algorithm
//
//===----------------------------------------------------------------------===//

#include "fast_date.h"
#include <cstdio>
#include <ctime>
#include <cstring>
#include <cassert>

using namespace fast_date;

// Test counter
int tests_passed = 0;
int tests_failed = 0;

#define TEST(name) void test_##name()
#define RUN_TEST(name) do { \
  printf("Running %s...\n", #name); \
  test_##name(); \
} while(0)

#define ASSERT_EQ(a, b) do { \
  if ((a) != (b)) { \
    printf("  FAIL: %s:%d: %s != %s (%lld != %lld)\n", __FILE__, __LINE__, #a, #b, (long long)(a), (long long)(b)); \
    tests_failed++; \
    return; \
  } \
  tests_passed++; \
} while(0)

#define ASSERT_TRUE(cond) do { \
  if (!(cond)) { \
    printf("  FAIL: %s:%d: %s is false\n", __FILE__, __LINE__, #cond); \
    tests_failed++; \
    return; \
  } \
  tests_passed++; \
} while(0)

// Helper to compare with system gmtime
bool compare_with_system(int64_t timestamp, int &year, int &month, int &day, 
                         int &hour, int &minute, int &second, int &wday, int &yday) {
  time_t t = static_cast<time_t>(timestamp);
  struct tm* sys = gmtime(&t);
  if (!sys) return false;
  
  year = sys->tm_year + 1900;
  month = sys->tm_mon + 1;
  day = sys->tm_mday;
  hour = sys->tm_hour;
  minute = sys->tm_min;
  second = sys->tm_sec;
  wday = sys->tm_wday;
  yday = sys->tm_yday;
  return true;
}

TEST(unix_epoch) {
  DateResult result = unix_to_date_fast(0);
  ASSERT_TRUE(result.valid);
  ASSERT_EQ(result.year, 1970);
  ASSERT_EQ(result.month, 1);
  ASSERT_EQ(result.day, 1);
  ASSERT_EQ(result.hour, 0);
  ASSERT_EQ(result.minute, 0);
  ASSERT_EQ(result.second, 0);
  ASSERT_EQ(result.wday, 4); // Thursday
  ASSERT_EQ(result.yday, 0);
}

TEST(y2k) {
  DateResult result = unix_to_date_fast(946684800); // 2000-01-01 00:00:00
  ASSERT_TRUE(result.valid);
  ASSERT_EQ(result.year, 2000);
  ASSERT_EQ(result.month, 1);
  ASSERT_EQ(result.day, 1);
  ASSERT_EQ(result.hour, 0);
  ASSERT_EQ(result.minute, 0);
  ASSERT_EQ(result.second, 0);
  ASSERT_EQ(result.wday, 6); // Saturday
  ASSERT_EQ(result.yday, 0);
}

TEST(leap_day_2000) {
  DateResult result = unix_to_date_fast(951782400); // 2000-02-29 00:00:00
  ASSERT_TRUE(result.valid);
  ASSERT_EQ(result.year, 2000);
  ASSERT_EQ(result.month, 2);
  ASSERT_EQ(result.day, 29);
  ASSERT_EQ(result.yday, 59);
}

TEST(leap_day_2004) {
  DateResult result = unix_to_date_fast(1078012800); // 2004-02-29 00:00:00
  ASSERT_TRUE(result.valid);
  ASSERT_EQ(result.year, 2004);
  ASSERT_EQ(result.month, 2);
  ASSERT_EQ(result.day, 29);
}

TEST(non_leap_year_2001) {
  DateResult result = unix_to_date_fast(983318400); // 2001-02-28 00:00:00
  ASSERT_TRUE(result.valid);
  ASSERT_EQ(result.year, 2001);
  ASSERT_EQ(result.month, 2);
  ASSERT_EQ(result.day, 28);
  
  // Next day should be March 1
  result = unix_to_date_fast(983404800); // 2001-03-01 00:00:00
  ASSERT_TRUE(result.valid);
  ASSERT_EQ(result.year, 2001);
  ASSERT_EQ(result.month, 3);
  ASSERT_EQ(result.day, 1);
}

TEST(year_1900_not_leap) {
  // 1900 is NOT a leap year (divisible by 100 but not 400)
  DateResult result = unix_to_date_fast(-2203977600); // 1900-02-28 00:00:00
  ASSERT_TRUE(result.valid);
  ASSERT_EQ(result.year, 1900);
  ASSERT_EQ(result.month, 2);
  ASSERT_EQ(result.day, 28);
  
  // Next day should be March 1
  result = unix_to_date_fast(-2203891200); // 1900-03-01 00:00:00
  ASSERT_TRUE(result.valid);
  ASSERT_EQ(result.year, 1900);
  ASSERT_EQ(result.month, 3);
  ASSERT_EQ(result.day, 1);
}

TEST(year_2100_not_leap) {
  // 2100 is NOT a leap year (divisible by 100 but not 400)
  DateResult result = unix_to_date_fast(4107456000); // 2100-02-28 00:00:00
  ASSERT_TRUE(result.valid);
  ASSERT_EQ(result.year, 2100);
  ASSERT_EQ(result.month, 2);
  ASSERT_EQ(result.day, 28);
  
  // Next day should be March 1
  result = unix_to_date_fast(4107542400); // 2100-03-01 00:00:00
  ASSERT_TRUE(result.valid);
  ASSERT_EQ(result.year, 2100);
  ASSERT_EQ(result.month, 3);
  ASSERT_EQ(result.day, 1);
}

TEST(year_2400_is_leap) {
  // 2400 IS a leap year (divisible by 400)
  DateResult result = unix_to_date_fast(13574563200); // 2400-02-29 00:00:00
  ASSERT_TRUE(result.valid);
  ASSERT_EQ(result.year, 2400);
  ASSERT_EQ(result.month, 2);
  ASSERT_EQ(result.day, 29);
}

TEST(32bit_limit) {
  DateResult result = unix_to_date_fast(2147483647); // 2038-01-19 03:14:07
  ASSERT_TRUE(result.valid);
  ASSERT_EQ(result.year, 2038);
  ASSERT_EQ(result.month, 1);
  ASSERT_EQ(result.day, 19);
  ASSERT_EQ(result.hour, 3);
  ASSERT_EQ(result.minute, 14);
  ASSERT_EQ(result.second, 7);
}

TEST(negative_timestamp) {
  DateResult result = unix_to_date_fast(-86400); // 1969-12-31 00:00:00
  ASSERT_TRUE(result.valid);
  ASSERT_EQ(result.year, 1969);
  ASSERT_EQ(result.month, 12);
  ASSERT_EQ(result.day, 31);
  ASSERT_EQ(result.yday, 364);
}

TEST(far_past) {
  DateResult result = unix_to_date_fast(-2208988800); // 1900-01-01 00:00:00
  ASSERT_TRUE(result.valid);
  ASSERT_EQ(result.year, 1900);
  ASSERT_EQ(result.month, 1);
  ASSERT_EQ(result.day, 1);
}

TEST(far_future) {
  DateResult result = unix_to_date_fast(4102444800); // 2100-01-01 00:00:00
  ASSERT_TRUE(result.valid);
  ASSERT_EQ(result.year, 2100);
  ASSERT_EQ(result.month, 1);
  ASSERT_EQ(result.day, 1);
}

TEST(time_components) {
  // Test various times of day
  DateResult result = unix_to_date_fast(946731245); // 2000-01-01 12:54:05
  ASSERT_TRUE(result.valid);
  ASSERT_EQ(result.year, 2000);
  ASSERT_EQ(result.month, 1);
  ASSERT_EQ(result.day, 1);
  ASSERT_EQ(result.hour, 12);
  ASSERT_EQ(result.minute, 54);
  ASSERT_EQ(result.second, 5);
}

TEST(all_months) {
  // Test each month of a year
  int64_t base = 946684800; // 2000-01-01 00:00:00
  int days_per_month[] = {31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31}; // 2000 is leap
  
  int64_t timestamp = base;
  for (int m = 1; m <= 12; m++) {
    DateResult result = unix_to_date_fast(timestamp);
    ASSERT_TRUE(result.valid);
    ASSERT_EQ(result.year, 2000);
    ASSERT_EQ(result.month, m);
    ASSERT_EQ(result.day, 1);
    
    // Move to next month
    timestamp += days_per_month[m-1] * 86400;
  }
}

TEST(day_of_week) {
  // Known dates and their weekdays
  struct TestCase {
    int64_t timestamp;
    int expected_wday;
  } cases[] = {
    {0, 4},              // 1970-01-01 Thursday
    {946684800, 6},      // 2000-01-01 Saturday
    {1234567890, 5},     // 2009-02-13 Friday
    {1609459200, 5},     // 2021-01-01 Friday
  };
  
  for (const auto& tc : cases) {
    DateResult result = unix_to_date_fast(tc.timestamp);
    ASSERT_TRUE(result.valid);
    ASSERT_EQ(result.wday, tc.expected_wday);
  }
}

TEST(inverse_function_basic) {
  // Test date_to_unix_fast
  int64_t timestamp = date_to_unix_fast(2000, 1, 1, 0, 0, 0);
  ASSERT_EQ(timestamp, 946684800);
  
  timestamp = date_to_unix_fast(1970, 1, 1, 0, 0, 0);
  ASSERT_EQ(timestamp, 0);
  
  timestamp = date_to_unix_fast(2038, 1, 19, 3, 14, 7);
  ASSERT_EQ(timestamp, 2147483647);
}

TEST(round_trip) {
  // Test that converting timestamp->date->timestamp gives original value
  int64_t timestamps[] = {
    0,
    946684800,
    951868800,
    1234567890,
    2147483647,
    -86400,
    -2208988800,
  };
  
  for (int64_t ts : timestamps) {
    DateResult date = unix_to_date_fast(ts);
    ASSERT_TRUE(date.valid);
    
    int64_t ts2 = date_to_unix_fast(date.year, date.month, date.day,
                                     date.hour, date.minute, date.second);
    ASSERT_EQ(ts, ts2);
  }
}

TEST(compare_with_system_gmtime) {
  // Test various timestamps against system gmtime
  int64_t timestamps[] = {
    0,
    946684800,
    951868800,
    1234567890,
    1609459200,
    -86400,
  };
  
  for (int64_t ts : timestamps) {
    DateResult fast = unix_to_date_fast(ts);
    ASSERT_TRUE(fast.valid);
    
    int sys_year, sys_month, sys_day, sys_hour, sys_min, sys_sec, sys_wday, sys_yday;
    bool sys_ok = compare_with_system(ts, sys_year, sys_month, sys_day,
                                      sys_hour, sys_min, sys_sec, sys_wday, sys_yday);
    
    if (sys_ok) {
      ASSERT_EQ(fast.year, sys_year);
      ASSERT_EQ(fast.month, sys_month);
      ASSERT_EQ(fast.day, sys_day);
      ASSERT_EQ(fast.hour, sys_hour);
      ASSERT_EQ(fast.minute, sys_min);
      ASSERT_EQ(fast.second, sys_sec);
      ASSERT_EQ(fast.wday, sys_wday);
      ASSERT_EQ(fast.yday, sys_yday);
    }
  }
}

TEST(edge_cases_end_of_month) {
  // Test last day of each month
  struct TestCase {
    int year;
    int month;
    int day;
  } cases[] = {
    {2000, 1, 31},
    {2000, 2, 29},
    {2000, 3, 31},
    {2000, 4, 30},
    {2000, 5, 31},
    {2000, 6, 30},
    {2000, 7, 31},
    {2000, 8, 31},
    {2000, 9, 30},
    {2000, 10, 31},
    {2000, 11, 30},
    {2000, 12, 31},
  };
  
  for (const auto& tc : cases) {
    int64_t ts = date_to_unix_fast(tc.year, tc.month, tc.day, 0, 0, 0);
    DateResult result = unix_to_date_fast(ts);
    
    ASSERT_TRUE(result.valid);
    ASSERT_EQ(result.year, tc.year);
    ASSERT_EQ(result.month, tc.month);
    ASSERT_EQ(result.day, tc.day);
  }
}

TEST(century_boundaries) {
  // Test dates around century boundaries
  struct TestCase {
    int64_t timestamp;
    int year;
    int month;
    int day;
  } cases[] = {
    {-2208988800, 1900, 1, 1},   // Start of 20th century
    {946684800, 2000, 1, 1},      // Start of 21st century
    {4102444800, 2100, 1, 1},     // Start of 22nd century
  };
  
  for (const auto& tc : cases) {
    DateResult result = unix_to_date_fast(tc.timestamp);
    ASSERT_TRUE(result.valid);
    ASSERT_EQ(result.year, tc.year);
    ASSERT_EQ(result.month, tc.month);
    ASSERT_EQ(result.day, tc.day);
  }
}

TEST(sequential_days) {
  // Test 1000 consecutive days starting from epoch
  int64_t timestamp = 0;
  int prev_yday = -1;
  int prev_year = 0;
  
  for (int i = 0; i < 1000; i++) {
    DateResult result = unix_to_date_fast(timestamp);
    ASSERT_TRUE(result.valid);
    
    // yday should increment (or reset to 0 on new year)
    if (result.year == prev_year) {
      ASSERT_EQ(result.yday, prev_yday + 1);
    } else if (result.year == prev_year + 1) {
      ASSERT_EQ(result.yday, 0);
    }
    
    prev_yday = result.yday;
    prev_year = result.year;
    timestamp += 86400; // Next day
  }
}

TEST(invalid_dates) {
  // Test that inverse function handles invalid inputs
  int64_t ts;
  
  ts = date_to_unix_fast(2000, 13, 1, 0, 0, 0); // Invalid month
  ASSERT_EQ(ts, -1);
  
  ts = date_to_unix_fast(2000, 0, 1, 0, 0, 0); // Invalid month
  ASSERT_EQ(ts, -1);
  
  ts = date_to_unix_fast(2000, 1, 32, 0, 0, 0); // Invalid day
  ASSERT_EQ(ts, -1);
  
  ts = date_to_unix_fast(2000, 1, 1, 24, 0, 0); // Invalid hour
  ASSERT_EQ(ts, -1);
  
  ts = date_to_unix_fast(2000, 1, 1, 0, 60, 0); // Invalid minute
  ASSERT_EQ(ts, -1);
  
  ts = date_to_unix_fast(2000, 1, 1, 0, 0, 60); // Invalid second
  ASSERT_EQ(ts, -1);
}

int main() {
  printf("========================================\n");
  printf("Fast Date Algorithm Unit Tests\n");
  printf("========================================\n\n");
  
  RUN_TEST(unix_epoch);
  RUN_TEST(y2k);
  RUN_TEST(leap_day_2000);
  RUN_TEST(leap_day_2004);
  RUN_TEST(non_leap_year_2001);
  RUN_TEST(year_1900_not_leap);
  RUN_TEST(year_2100_not_leap);
  RUN_TEST(year_2400_is_leap);
  RUN_TEST(32bit_limit);
  RUN_TEST(negative_timestamp);
  RUN_TEST(far_past);
  RUN_TEST(far_future);
  RUN_TEST(time_components);
  RUN_TEST(all_months);
  RUN_TEST(day_of_week);
  RUN_TEST(inverse_function_basic);
  RUN_TEST(round_trip);
  RUN_TEST(compare_with_system_gmtime);
  RUN_TEST(edge_cases_end_of_month);
  RUN_TEST(century_boundaries);
  RUN_TEST(sequential_days);
  RUN_TEST(invalid_dates);
  
  printf("\n========================================\n");
  printf("Test Results\n");
  printf("========================================\n");
  printf("Passed: %d\n", tests_passed);
  printf("Failed: %d\n", tests_failed);
  printf("Total:  %d\n", tests_passed + tests_failed);
  
  if (tests_failed == 0) {
    printf("\n✓ All tests PASSED!\n");
    return 0;
  } else {
    printf("\n✗ Some tests FAILED!\n");
    return 1;
  }
}
