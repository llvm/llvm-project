//===-- Unittests for update_from_seconds_fast ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "hdr/errno_macros.h"
#include "hdr/types/struct_tm.h"
#include "src/__support/CPP/limits.h"
#include "src/time/time_constants.h"
#include "src/time/time_utils.h"
#include "test/UnitTest/ErrnoCheckingTest.h"
#include "test/UnitTest/Test.h"
#include "test/src/time/TmMatcher.h"

using LIBC_NAMESPACE::time_utils::update_from_seconds;
using LIBC_NAMESPACE::time_utils::update_from_seconds_fast;

// Test that fast and old implementations produce identical results
class UpdateFromSecondsFastTest : public LIBC_NAMESPACE::testing::Test {
public:
  void compare_implementations(time_t seconds, const char *description) {
    struct tm result_old, result_fast;
    
    int64_t ret_old = update_from_seconds(seconds, &result_old);
    int64_t ret_fast = update_from_seconds_fast(seconds, &result_fast);
    
    EXPECT_EQ(ret_old, ret_fast) << description << ": return values differ";
    EXPECT_TM_EQ(result_old, result_fast) << description << ": struct tm differs";
  }
};

TEST_F(UpdateFromSecondsFastTest, UnixEpoch) {
  compare_implementations(0, "Unix epoch");
  
  // Also verify exact values
  struct tm result;
  update_from_seconds_fast(0, &result);
  EXPECT_TM_EQ(
      (tm{0, // sec
          0, // min
          0, // hr
          1, // day
          0, // tm_mon (Jan = 0)
          1970 - LIBC_NAMESPACE::time_constants::TIME_YEAR_BASE, // year
          4, // wday (Thursday)
          0, // yday
          0}),
      result);
}

TEST_F(UpdateFromSecondsFastTest, Y2K) {
  compare_implementations(946684800, "Y2K (2000-01-01)");
  
  struct tm result;
  update_from_seconds_fast(946684800, &result);
  EXPECT_TM_EQ(
      (tm{0, // sec
          0, // min
          0, // hr
          1, // day
          0, // tm_mon (Jan = 0)
          2000 - LIBC_NAMESPACE::time_constants::TIME_YEAR_BASE, // year
          6, // wday (Saturday)
          0, // yday
          0}),
      result);
}

TEST_F(UpdateFromSecondsFastTest, LeapYears) {
  // Leap year 2000 - Feb 29
  compare_implementations(951782400, "Leap day 2000 (2000-02-29)");
  
  struct tm result;
  update_from_seconds_fast(951782400, &result);
  EXPECT_EQ(result.tm_year, 2000 - LIBC_NAMESPACE::time_constants::TIME_YEAR_BASE);
  EXPECT_EQ(result.tm_mon, 1); // February
  EXPECT_EQ(result.tm_mday, 29);
  EXPECT_EQ(result.tm_yday, 59); // 31 (Jan) + 29 - 1 = 59
  
  // Leap year 2004 - Feb 29
  compare_implementations(1078012800, "Leap day 2004 (2004-02-29)");
  
  // Non-leap year 1900 (divisible by 100 but not 400)
  compare_implementations(-2203977600, "1900-03-01 (year 1900 is NOT leap)");
  
  // Leap year 2400 (divisible by 400)
  compare_implementations(13574563200, "Leap day 2400 (2400-02-29)");
}

TEST_F(UpdateFromSecondsFastTest, CenturyBoundaries) {
  // 1900-01-01
  compare_implementations(-2208988800, "Century boundary 1900");
  
  // 2000-01-01 (already tested but important)
  compare_implementations(946684800, "Century boundary 2000");
  
  // 2100-01-01
  compare_implementations(4102444800, "Century boundary 2100");
}

TEST_F(UpdateFromSecondsFastTest, AllMonthsOf2024) {
  // Test each month of 2024 (leap year)
  time_t timestamps[] = {
      1704067200,  // 2024-01-01
      1706745600,  // 2024-02-01
      1709251200,  // 2024-03-01
      1711929600,  // 2024-04-01
      1714521600,  // 2024-05-01
      1717200000,  // 2024-06-01
      1719792000,  // 2024-07-01
      1722470400,  // 2024-08-01
      1725148800,  // 2024-09-01
      1727740800,  // 2024-10-01
      1730419200,  // 2024-11-01
      1733011200   // 2024-12-01
  };
  
  for (int i = 0; i < 12; i++) {
    compare_implementations(timestamps[i], "2024 month test");
    
    struct tm result;
    update_from_seconds_fast(timestamps[i], &result);
    EXPECT_EQ(result.tm_year, 2024 - LIBC_NAMESPACE::time_constants::TIME_YEAR_BASE);
    EXPECT_EQ(result.tm_mon, i);
    EXPECT_EQ(result.tm_mday, 1);
  }
}

TEST_F(UpdateFromSecondsFastTest, NegativeTimestamps) {
  // -1 day (1969-12-31)
  compare_implementations(-86400, "1969-12-31");
  
  struct tm result;
  update_from_seconds_fast(-86400, &result);
  EXPECT_EQ(result.tm_year, 1969 - LIBC_NAMESPACE::time_constants::TIME_YEAR_BASE);
  EXPECT_EQ(result.tm_mon, 11); // December
  EXPECT_EQ(result.tm_mday, 31);
  EXPECT_EQ(result.tm_wday, 3); // Wednesday
  
  // -1 second (1969-12-31 23:59:59)
  compare_implementations(-1, "1969-12-31 23:59:59");
  
  update_from_seconds_fast(-1, &result);
  EXPECT_EQ(result.tm_hour, 23);
  EXPECT_EQ(result.tm_min, 59);
  EXPECT_EQ(result.tm_sec, 59);
}

TEST_F(UpdateFromSecondsFastTest, ThirtyTwoBitLimits) {
  // Maximum 32-bit signed integer: 2038-01-19 03:14:07
  compare_implementations(2147483647, "32-bit max");
  
  struct tm result;
  update_from_seconds_fast(2147483647, &result);
  EXPECT_EQ(result.tm_year, 2038 - LIBC_NAMESPACE::time_constants::TIME_YEAR_BASE);
  EXPECT_EQ(result.tm_mon, 0); // January
  EXPECT_EQ(result.tm_mday, 19);
  EXPECT_EQ(result.tm_hour, 3);
  EXPECT_EQ(result.tm_min, 14);
  EXPECT_EQ(result.tm_sec, 7);
  
  // Minimum 32-bit signed integer: 1901-12-13 20:45:52
  compare_implementations(-2147483648LL, "32-bit min");
}

TEST_F(UpdateFromSecondsFastTest, TimeComponents) {
  // Test time parsing: 2023-11-14 22:13:20
  compare_implementations(1700000000, "Time components test");
  
  struct tm result;
  update_from_seconds_fast(1700000000, &result);
  EXPECT_EQ(result.tm_hour, 22);
  EXPECT_EQ(result.tm_min, 13);
  EXPECT_EQ(result.tm_sec, 20);
}

TEST_F(UpdateFromSecondsFastTest, DayOfWeek) {
  // Test day-of-week calculation for known dates
  struct TestCase {
    time_t timestamp;
    int expected_wday;
    const char *description;
  } cases[] = {
      {0, 4, "1970-01-01 Thursday"},
      {946684800, 6, "2000-01-01 Saturday"},
      {1609459200, 5, "2021-01-01 Friday"},
      {1234567890, 5, "2009-02-13 Friday"},
  };
  
  for (const auto &tc : cases) {
    compare_implementations(tc.timestamp, tc.description);
    
    struct tm result;
    update_from_seconds_fast(tc.timestamp, &result);
    EXPECT_EQ(result.tm_wday, tc.expected_wday) << tc.description;
  }
}

TEST_F(UpdateFromSecondsFastTest, DayOfYear) {
  // Jan 1: yday = 0
  struct tm result;
  update_from_seconds_fast(946684800, &result); // 2000-01-01
  EXPECT_EQ(result.tm_yday, 0);
  
  // Feb 29 in leap year: yday = 59
  update_from_seconds_fast(951782400, &result); // 2000-02-29
  EXPECT_EQ(result.tm_yday, 59);
  
  // Dec 31 in leap year: yday = 365
  update_from_seconds_fast(978220800, &result); // 2000-12-31
  EXPECT_EQ(result.tm_yday, 365);
  
  // Dec 31 in non-leap year: yday = 364
  update_from_seconds_fast(1009756800, &result); // 2001-12-31
  EXPECT_EQ(result.tm_yday, 364);
}

TEST_F(UpdateFromSecondsFastTest, SequentialDays) {
  // Test 100 consecutive days to ensure continuity
  time_t base = 946684800; // 2000-01-01
  for (int i = 0; i < 100; i++) {
    time_t ts = base + i * 86400;
    compare_implementations(ts, "Sequential days test");
  }
}

TEST_F(UpdateFromSecondsFastTest, EndOfMonths) {
  // Test end-of-month dates for all months
  time_t timestamps[] = {
      949363200,  // 2000-01-31
      951782400,  // 2000-02-29 (leap year)
      954374400,  // 2000-03-31
      957052800,  // 2000-04-30
      959731200,  // 2000-05-31
      962409600,  // 2000-06-30
      965001600,  // 2000-07-31
      967680000,  // 2000-08-31
      970358400,  // 2000-09-30
      972950400,  // 2000-10-31
      975628800,  // 2000-11-30
      978220800   // 2000-12-31
  };
  
  for (size_t i = 0; i < sizeof(timestamps) / sizeof(timestamps[0]); i++) {
    compare_implementations(timestamps[i], "End of month test");
  }
}

TEST_F(UpdateFromSecondsFastTest, YearTransitions) {
  // Test year transitions
  struct TestCase {
    time_t timestamp;
    int expected_year;
    int expected_mon;
    int expected_mday;
  } cases[] = {
      {946684799, 1999, 11, 31},  // 1999-12-31 23:59:59
      {946684800, 2000, 0, 1},    // 2000-01-01 00:00:00
      {978220799, 2000, 11, 31},  // 2000-12-31 23:59:59
      {978220800, 2001, 0, 1},    // 2001-01-01 00:00:00
  };
  
  for (const auto &tc : cases) {
    compare_implementations(tc.timestamp, "Year transition test");
    
    struct tm result;
    update_from_seconds_fast(tc.timestamp, &result);
    EXPECT_EQ(result.tm_year, tc.expected_year - LIBC_NAMESPACE::time_constants::TIME_YEAR_BASE);
    EXPECT_EQ(result.tm_mon, tc.expected_mon);
    EXPECT_EQ(result.tm_mday, tc.expected_mday);
  }
}

TEST_F(UpdateFromSecondsFastTest, FarPastAndFuture) {
  // Far past: year 1000
  compare_implementations(-30578688000LL, "Year 1000");
  
  // Far future: year 3000
  compare_implementations(32503680000LL, "Year 3000");
  
  // Year 5000
  compare_implementations(95617584000LL, "Year 5000");
}

TEST_F(UpdateFromSecondsFastTest, AllYears1900To2100) {
  // Test Jan 1 of every year from 1900 to 2100
  for (int year = 1900; year <= 2100; year++) {
    // Calculate timestamp for Jan 1 of this year
    // This is approximate but sufficient for comparison testing
    int64_t days_from_1970 = 0;
    for (int y = 1970; y < year; y++) {
      bool is_leap = (y % 4 == 0) && ((y % 100 != 0) || (y % 400 == 0));
      days_from_1970 += is_leap ? 366 : 365;
    }
    for (int y = year; y < 1970; y++) {
      bool is_leap = (y % 4 == 0) && ((y % 100 != 0) || (y % 400 == 0));
      days_from_1970 -= is_leap ? 366 : 365;
    }
    
    time_t ts = days_from_1970 * 86400;
    compare_implementations(ts, "All years 1900-2100");
    
    struct tm result;
    update_from_seconds_fast(ts, &result);
    EXPECT_EQ(result.tm_year, year - LIBC_NAMESPACE::time_constants::TIME_YEAR_BASE);
    EXPECT_EQ(result.tm_mon, 0);
    EXPECT_EQ(result.tm_mday, 1);
  }
}

TEST_F(UpdateFromSecondsFastTest, OutOfRange) {
  if (sizeof(time_t) < sizeof(int64_t))
    return;
    
  struct tm result;
  
  time_t seconds =
      1 + INT_MAX * static_cast<int64_t>(
          LIBC_NAMESPACE::time_constants::NUMBER_OF_SECONDS_IN_LEAP_YEAR);
  int64_t ret = update_from_seconds_fast(seconds, &result);
  EXPECT_LT(ret, 0); // Should return error
  
  seconds = INT_MIN * static_cast<int64_t>(
          LIBC_NAMESPACE::time_constants::NUMBER_OF_SECONDS_IN_LEAP_YEAR) - 1;
  ret = update_from_seconds_fast(seconds, &result);
  EXPECT_LT(ret, 0); // Should return error
}

// Benchmark comparison test (not a unit test, but useful for validation)
TEST_F(UpdateFromSecondsFastTest, PerformanceComparison) {
  // This test validates that both implementations handle the same workload
  // Actual performance benchmarking should be done separately
  const int N = 10000;
  time_t base = 946684800; // 2000-01-01
  
  for (int i = 0; i < N; i++) {
    time_t ts = base + i * 1000; // Every 1000 seconds
    compare_implementations(ts, "Performance test");
  }
}
