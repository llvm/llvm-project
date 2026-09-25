//===-- Unittests for gmtime ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "hdr/errno_macros.h"

#include "hdr/types/struct_tm.h"
#include "src/__support/CPP/limits.h" // INT_MAX, INT_MIN
#include "src/time/gmtime.h"
#include "src/time/time_constants.h"
#include "src/time/time_utils.h"
#include "test/UnitTest/ErrnoCheckingTest.h"
#include "test/UnitTest/Test.h"
#include "test/src/time/TmMatcher.h"

using LlvmLibcGmTime = LIBC_NAMESPACE::testing::ErrnoCheckingTest;

static_assert(sizeof(time_t) == 8, "LLVM libc requires a 64-bit time_t.");

TEST_F(LlvmLibcGmTime, OutOfRange) {
  time_t seconds =
      1 +
      INT_MAX *
          static_cast<int64_t>(
              LIBC_NAMESPACE::time_constants::NUMBER_OF_SECONDS_IN_LEAP_YEAR);
  struct tm *tm_data = LIBC_NAMESPACE::gmtime(&seconds);
  EXPECT_TRUE(tm_data == nullptr);
  ASSERT_ERRNO_EQ(LIBC_NAMESPACE::time_utils::TIME_OVERFLOW);

  seconds =
      INT_MIN *
          static_cast<int64_t>(
              LIBC_NAMESPACE::time_constants::NUMBER_OF_SECONDS_IN_LEAP_YEAR) -
      1;
  tm_data = LIBC_NAMESPACE::gmtime(&seconds);
  EXPECT_TRUE(tm_data == nullptr);
  ASSERT_ERRNO_EQ(LIBC_NAMESPACE::time_utils::TIME_OVERFLOW);
}

TEST_F(LlvmLibcGmTime, NullPtr) {
  EXPECT_DEATH([] { LIBC_NAMESPACE::gmtime(nullptr); }, WITH_SIGNAL(-1));
}

TEST_F(LlvmLibcGmTime, InvalidSeconds) {
  time_t seconds = 0;
  struct tm *tm_data = nullptr;
  // -1 second from 1970-01-01 00:00:00 returns 1969-12-31 23:59:59.
  seconds = -1;
  tm_data = LIBC_NAMESPACE::gmtime(&seconds);
  EXPECT_TM_EQ(
      (tm{59,     // sec
          59,     // min
          23,     // hr
          31,     // day
          12 - 1, // tm_mon starts with 0 for Jan
          1969 - LIBC_NAMESPACE::time_constants::TIME_YEAR_BASE, // year
          3,                                                     // wday
          364,                                                   // yday
          0}),
      *tm_data);
  // 60 seconds from 1970-01-01 00:00:00 returns 1970-01-01 00:01:00.
  seconds = 60;
  tm_data = LIBC_NAMESPACE::gmtime(&seconds);
  EXPECT_TM_EQ(
      (tm{0, // sec
          1, // min
          0, // hr
          1, // day
          0, // tm_mon starts with 0 for Jan
          1970 - LIBC_NAMESPACE::time_constants::TIME_YEAR_BASE, // year
          4,                                                     // wday
          0,                                                     // yday
          0}),
      *tm_data);
}

TEST_F(LlvmLibcGmTime, InvalidMinutes) {
  time_t seconds = 0;
  struct tm *tm_data = nullptr;
  // -1 minute from 1970-01-01 00:00:00 returns 1969-12-31 23:59:00.
  seconds = -LIBC_NAMESPACE::time_constants::SECONDS_PER_MIN;
  tm_data = LIBC_NAMESPACE::gmtime(&seconds);
  EXPECT_TM_EQ(
      (tm{0,  // sec
          59, // min
          23, // hr
          31, // day
          11, // tm_mon starts with 0 for Jan
          1969 - LIBC_NAMESPACE::time_constants::TIME_YEAR_BASE, // year
          3,                                                     // wday
          364,                                                   // yday
          0}),
      *tm_data);
  // 60 minutes from 1970-01-01 00:00:00 returns 1970-01-01 01:00:00.
  seconds = 60 * LIBC_NAMESPACE::time_constants::SECONDS_PER_MIN;
  tm_data = LIBC_NAMESPACE::gmtime(&seconds);
  EXPECT_TM_EQ(
      (tm{0, // sec
          0, // min
          1, // hr
          1, // day
          0, // tm_mon starts with 0 for Jan
          1970 - LIBC_NAMESPACE::time_constants::TIME_YEAR_BASE, // year
          4,                                                     // wday
          0,                                                     // yday
          0}),
      *tm_data);
}

TEST_F(LlvmLibcGmTime, InvalidHours) {
  time_t seconds = 0;
  struct tm *tm_data = nullptr;
  // -1 hour from 1970-01-01 00:00:00 returns 1969-12-31 23:00:00.
  seconds = -LIBC_NAMESPACE::time_constants::SECONDS_PER_HOUR;
  tm_data = LIBC_NAMESPACE::gmtime(&seconds);
  EXPECT_TM_EQ(
      (tm{0,  // sec
          0,  // min
          23, // hr
          31, // day
          11, // tm_mon starts with 0 for Jan
          1969 - LIBC_NAMESPACE::time_constants::TIME_YEAR_BASE, // year
          3,                                                     // wday
          364,                                                   // yday
          0}),
      *tm_data);
  // 24 hours from 1970-01-01 00:00:00 returns 1970-01-02 00:00:00.
  seconds = 24 * LIBC_NAMESPACE::time_constants::SECONDS_PER_HOUR;
  tm_data = LIBC_NAMESPACE::gmtime(&seconds);
  EXPECT_TM_EQ(
      (tm{0, // sec
          0, // min
          0, // hr
          2, // day
          0, // tm_mon starts with 0 for Jan
          1970 - LIBC_NAMESPACE::time_constants::TIME_YEAR_BASE, // year
          5,                                                     // wday
          1,                                                     // yday
          0}),
      *tm_data);
}

TEST_F(LlvmLibcGmTime, InvalidYear) {
  // -1 year from 1970-01-01 00:00:00 returns 1969-01-01 00:00:00.
  time_t seconds = -LIBC_NAMESPACE::time_constants::DAYS_PER_NON_LEAP_YEAR *
                   LIBC_NAMESPACE::time_constants::SECONDS_PER_DAY;
  struct tm *tm_data = LIBC_NAMESPACE::gmtime(&seconds);
  EXPECT_TM_EQ(
      (tm{0, // sec
          0, // min
          0, // hr
          1, // day
          0, // tm_mon starts with 0 for Jan
          1969 - LIBC_NAMESPACE::time_constants::TIME_YEAR_BASE, // year
          3,                                                     // wday
          0,                                                     // yday
          0}),
      *tm_data);
}

TEST_F(LlvmLibcGmTime, InvalidMonths) {
  time_t seconds = 0;
  struct tm *tm_data = nullptr;
  // -1 month from 1970-01-01 00:00:00 returns 1969-12-01 00:00:00.
  seconds = -31 * LIBC_NAMESPACE::time_constants::SECONDS_PER_DAY;
  tm_data = LIBC_NAMESPACE::gmtime(&seconds);
  EXPECT_TM_EQ(
      (tm{0,      // sec
          0,      // min
          0,      // hr
          1,      // day
          12 - 1, // tm_mon starts with 0 for Jan
          1969 - LIBC_NAMESPACE::time_constants::TIME_YEAR_BASE, // year
          1,                                                     // wday
          334,                                                   // yday
          0}),
      *tm_data);
  // 1970-13-01 00:00:00 returns 1971-01-01 00:00:00.
  seconds = LIBC_NAMESPACE::time_constants::DAYS_PER_NON_LEAP_YEAR *
            LIBC_NAMESPACE::time_constants::SECONDS_PER_DAY;
  tm_data = LIBC_NAMESPACE::gmtime(&seconds);
  EXPECT_TM_EQ(
      (tm{0, // sec
          0, // min
          0, // hr
          1, // day
          0, // tm_mon starts with 0 for Jan
          1971 - LIBC_NAMESPACE::time_constants::TIME_YEAR_BASE, // year
          5,                                                     // wday
          0,                                                     // yday
          0}),
      *tm_data);
}

TEST_F(LlvmLibcGmTime, InvalidDays) {
  time_t seconds = 0;
  struct tm *tm_data = nullptr;
  // -1 day from 1970-01-01 00:00:00 returns 1969-12-31 00:00:00.
  seconds = -1 * LIBC_NAMESPACE::time_constants::SECONDS_PER_DAY;
  tm_data = LIBC_NAMESPACE::gmtime(&seconds);
  EXPECT_TM_EQ(
      (tm{0,  // sec
          0,  // min
          0,  // hr
          31, // day
          11, // tm_mon starts with 0 for Jan
          1969 - LIBC_NAMESPACE::time_constants::TIME_YEAR_BASE, // year
          3,                                                     // wday
          364,                                                   // yday
          0}),
      *tm_data);

  // 1970-01-32 00:00:00 returns 1970-02-01 00:00:00.
  seconds = 31 * LIBC_NAMESPACE::time_constants::SECONDS_PER_DAY;
  tm_data = LIBC_NAMESPACE::gmtime(&seconds);
  EXPECT_TM_EQ(
      (tm{0, // sec
          0, // min
          0, // hr
          1, // day
          1, // tm_mon starts with 0 for Jan
          1970 - LIBC_NAMESPACE::time_constants::TIME_YEAR_BASE, // year
          0,                                                     // wday
          31,                                                    // yday
          0}),
      *tm_data);

  // 1970-02-29 00:00:00 returns 1970-03-01 00:00:00.
  seconds = 59 * LIBC_NAMESPACE::time_constants::SECONDS_PER_DAY;
  tm_data = LIBC_NAMESPACE::gmtime(&seconds);
  EXPECT_TM_EQ(
      (tm{0, // sec
          0, // min
          0, // hr
          1, // day
          2, // tm_mon starts with 0 for Jan
          1970 - LIBC_NAMESPACE::time_constants::TIME_YEAR_BASE, // year
          0,                                                     // wday
          59,                                                    // yday
          0}),
      *tm_data);

  // 1972-02-30 00:00:00 returns 1972-03-01 00:00:00.
  seconds =
      ((2 * LIBC_NAMESPACE::time_constants::DAYS_PER_NON_LEAP_YEAR) + 60) *
      LIBC_NAMESPACE::time_constants::SECONDS_PER_DAY;
  tm_data = LIBC_NAMESPACE::gmtime(&seconds);
  EXPECT_TM_EQ(
      (tm{0, // sec
          0, // min
          0, // hr
          1, // day
          2, // tm_mon starts with 0 for Jan
          1972 - LIBC_NAMESPACE::time_constants::TIME_YEAR_BASE, // year
          3,                                                     // wday
          60,                                                    // yday
          0}),
      *tm_data);
}

TEST_F(LlvmLibcGmTime, EndOf32BitEpochYear) {
  // Test for maximum value of a signed 32-bit integer.
  // Test implementation can encode time for Tue 19 January 2038 03:14:07 UTC.
  time_t seconds = 0x7FFFFFFF;
  struct tm *tm_data = LIBC_NAMESPACE::gmtime(&seconds);
  EXPECT_TM_EQ(
      (tm{7,  // sec
          14, // min
          3,  // hr
          19, // day
          0,  // tm_mon starts with 0 for Jan
          2038 - LIBC_NAMESPACE::time_constants::TIME_YEAR_BASE, // year
          2,                                                     // wday
          18,                                                    // yday
          0}),
      *tm_data);
}

TEST_F(LlvmLibcGmTime, Max64BitYear) {
  // Mon Jan 1 12:50:50 2170 (200 years from 1970),
  time_t seconds = 6311479850;
  struct tm *tm_data = LIBC_NAMESPACE::gmtime(&seconds);
  EXPECT_TM_EQ(
      (tm{50, // sec
          50, // min
          12, // hr
          1,  // day
          0,  // tm_mon starts with 0 for Jan
          2170 - LIBC_NAMESPACE::time_constants::TIME_YEAR_BASE, // year
          1,                                                     // wday
          0,                                                     // yday
          0}),
      *tm_data);

  // Test for Tue Jan 1 12:50:50 in 2,147,483,647th year.
  seconds = 67767976202043050;
  tm_data = LIBC_NAMESPACE::gmtime(&seconds);
  EXPECT_TM_EQ(
      (tm{50, // sec
          50, // min
          12, // hr
          1,  // day
          0,  // tm_mon starts with 0 for Jan
          2147483647 - LIBC_NAMESPACE::time_constants::TIME_YEAR_BASE, // year
          2,                                                           // wday
          0,                                                           // yday
          0}),
      *tm_data);
}

TEST_F(LlvmLibcGmTime, LeapYearRules) {
  time_t seconds;
  struct tm *tm_data;

  // Non-leap year 1900 (divisible by 100 but not 400) - Test March 1
  // Feb 29, 1900 doesn't exist, so we test March 1
  seconds = -2203891200;
  tm_data = LIBC_NAMESPACE::gmtime(&seconds);
  EXPECT_TM_EQ((tm{0, // sec
                   0, // min
                   0, // hr
                   1, // day
                   2, // tm_mon (March)
                   1900 - LIBC_NAMESPACE::time_constants::TIME_YEAR_BASE,
                   4,  // wday (Thursday)
                   59, // yday (Jan 31 + Feb 28)
                   0}),
               *tm_data);

  // Leap year 2000 (divisible by 400) - Feb 29 exists
  seconds = 951782400;
  tm_data = LIBC_NAMESPACE::gmtime(&seconds);
  EXPECT_TM_EQ((tm{0,  // sec
                   0,  // min
                   0,  // hr
                   29, // day
                   1,  // tm_mon (February)
                   2000 - LIBC_NAMESPACE::time_constants::TIME_YEAR_BASE,
                   2,  // wday (Tuesday)
                   59, // yday (Jan 31 + Feb 29 - 1)
                   0}),
               *tm_data);

  // Leap year 2400 (divisible by 400) - Feb 29 exists
  seconds = 13574563200LL;
  tm_data = LIBC_NAMESPACE::gmtime(&seconds);
  EXPECT_TM_EQ((tm{0,  // sec
                   0,  // min
                   0,  // hr
                   29, // day
                   1,  // tm_mon (February)
                   2400 - LIBC_NAMESPACE::time_constants::TIME_YEAR_BASE,
                   2,  // wday (Tuesday)
                   59, // yday (Jan 31 + Feb 29 - 1)
                   0}),
               *tm_data);
}

TEST_F(LlvmLibcGmTime, CenturyBoundaries) {
  time_t seconds;
  struct tm *tm_data;

  // 1900-01-01 (Monday)
  seconds = -2208988800;
  tm_data = LIBC_NAMESPACE::gmtime(&seconds);
  EXPECT_TM_EQ((tm{0, // sec
                   0, // min
                   0, // hr
                   1, // day
                   0, // tm_mon (January)
                   1900 - LIBC_NAMESPACE::time_constants::TIME_YEAR_BASE,
                   1, // wday (Monday)
                   0, // yday
                   0}),
               *tm_data);

  // 2100-01-01 (Friday)
  seconds = 4102444800;
  tm_data = LIBC_NAMESPACE::gmtime(&seconds);
  EXPECT_TM_EQ((tm{0, // sec
                   0, // min
                   0, // hr
                   1, // day
                   0, // tm_mon (January)
                   2100 - LIBC_NAMESPACE::time_constants::TIME_YEAR_BASE,
                   5, // wday (Friday)
                   0, // yday
                   0}),
               *tm_data);
}

TEST_F(LlvmLibcGmTime, FarPastAndFuture) {
  time_t seconds;
  struct tm *tm_data;

  // Far past: year 1000 (Wednesday)
  seconds = -30610224000LL;
  tm_data = LIBC_NAMESPACE::gmtime(&seconds);
  EXPECT_TM_EQ((tm{0, // sec
                   0, // min
                   0, // hr
                   1, // day
                   0, // tm_mon (January)
                   1000 - LIBC_NAMESPACE::time_constants::TIME_YEAR_BASE,
                   3, // wday (Wednesday)
                   0, // yday
                   0}),
               *tm_data);

  // Far future: year 3000 (Wednesday)
  seconds = 32503680000LL;
  tm_data = LIBC_NAMESPACE::gmtime(&seconds);
  EXPECT_TM_EQ((tm{0, // sec
                   0, // min
                   0, // hr
                   1, // day
                   0, // tm_mon (January)
                   3000 - LIBC_NAMESPACE::time_constants::TIME_YEAR_BASE,
                   3, // wday (Wednesday)
                   0, // yday
                   0}),
               *tm_data);
}

TEST_F(LlvmLibcGmTime, KeyYears) {
  // Test Jan 1 of key years to ensure calendar correctness
  struct TestCase {
    time_t timestamp;
    int year;
    int wday;
    const char *description;
  };

  TestCase cases[] = {
      {-2208988800, 1900, 1, "1900-01-01 Monday (non-leap century)"},
      {0, 1970, 4, "1970-01-01 Thursday (epoch)"},
      {915148800, 1999, 5, "1999-01-01 Friday"},
      {946684800, 2000, 6, "2000-01-01 Saturday (leap century)"},
      {978307200, 2001, 1, "2001-01-01 Monday"},
      {4102444800, 2100, 5, "2100-01-01 Friday (non-leap century)"},
      {13569465600LL, 2400, 6, "2400-01-01 Saturday (leap century)"},
  };

  for (const auto &tc : cases) {
    time_t seconds = tc.timestamp;
    struct tm *tm_data = LIBC_NAMESPACE::gmtime(&seconds);
    ASSERT_NE(tm_data, nullptr) << tc.description;
    EXPECT_EQ(tm_data->tm_year,
              tc.year - LIBC_NAMESPACE::time_constants::TIME_YEAR_BASE)
        << tc.description;
    EXPECT_EQ(tm_data->tm_mon, 0) << tc.description;
    EXPECT_EQ(tm_data->tm_mday, 1) << tc.description;
    EXPECT_EQ(tm_data->tm_wday, tc.wday) << tc.description;
    EXPECT_EQ(tm_data->tm_yday, 0) << tc.description;
  }
}
