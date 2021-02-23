//===-- Unittests for mktime ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/time/mktime.h"
#include "src/time/time_utils.h"
#include "test/ErrnoSetterMatcher.h"
#include "test/src/time/TmMatcher.h"
#include "utils/UnitTest/Test.h"

#include <errno.h>
#include <limits.h>
#include <string.h>

using __llvm_libc::testing::ErrnoSetterMatcher::Fails;
using __llvm_libc::testing::ErrnoSetterMatcher::Succeeds;
using __llvm_libc::time_utils::TimeConstants;

// A helper function to initialize tm data structure.
static inline void initialize_tm_data(struct tm *tm_data, int year, int month,
                                      int mday, int hour, int min, int sec,
                                      int wday, int yday) {
  struct tm temp = {.tm_sec = sec,
                    .tm_min = min,
                    .tm_hour = hour,
                    .tm_mday = mday,
                    .tm_mon = month - 1, // tm_mon starts with 0 for Jan
                    // years since 1900
                    .tm_year = year - TimeConstants::TimeYearBase,
                    .tm_wday = wday,
                    .tm_yday = yday};
  *tm_data = temp;
}

static inline time_t call_mktime(struct tm *tm_data, int year, int month,
                                 int mday, int hour, int min, int sec, int wday,
                                 int yday) {
  initialize_tm_data(tm_data, year, month, mday, hour, min, sec, wday, yday);
  return __llvm_libc::mktime(tm_data);
}

TEST(LlvmLibcMkTime, FailureSetsErrno) {
  struct tm tm_data;
  initialize_tm_data(&tm_data, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX, -1,
                     0, 0);
  EXPECT_THAT(__llvm_libc::mktime(&tm_data), Fails(EOVERFLOW));
}

TEST(LlvmLibcMkTime, MkTimesInvalidSeconds) {
  struct tm tm_data;
  // -1 second from 1970-01-01 00:00:00 returns 1969-12-31 23:59:59.
  EXPECT_THAT(call_mktime(&tm_data,
                          1970, // year
                          1,    // month
                          1,    // day
                          0,    // hr
                          0,    // min
                          -1,   // sec
                          0,    // wday
                          0),   // yday
              Succeeds(-1));
  EXPECT_TM_EQ((tm{59,     // sec
                   59,     // min
                   23,     // hr
                   31,     // day
                   12 - 1, // tm_mon starts with 0 for Jan
                   1969 - TimeConstants::TimeYearBase, // year
                   3,                                  // wday
                   364,                                // yday
                   0}),
               tm_data);
  // 60 seconds from 1970-01-01 00:00:00 returns 1970-01-01 00:01:00.
  EXPECT_THAT(call_mktime(&tm_data,
                          1970, // year
                          1,    // month
                          1,    // day
                          0,    // hr
                          0,    // min
                          60,   // sec
                          0,    // wday
                          0),   // yday
              Succeeds(60));
  EXPECT_TM_EQ((tm{0, // sec
                   1, // min
                   0, // hr
                   1, // day
                   0, // tm_mon starts with 0 for Jan
                   1970 - TimeConstants::TimeYearBase, // year
                   4,                                  // wday
                   0,                                  // yday
                   0}),
               tm_data);
}

TEST(LlvmLibcMkTime, MktimeTestsInvalidMinutes) {
  struct tm tm_data;
  // -1 minute from 1970-01-01 00:00:00 returns 1969-12-31 23:59:00.
  EXPECT_THAT(call_mktime(&tm_data,
                          1970, // year
                          1,    // month
                          1,    // day
                          0,    // hr
                          -1,   // min
                          0,    // sec
                          0,    // wday
                          0),   // yday
              Succeeds(-TimeConstants::SecondsPerMin));
  EXPECT_TM_EQ((tm{0,  // sec
                   59, // min
                   23, // hr
                   31, // day
                   11, // tm_mon starts with 0 for Jan
                   1969 - TimeConstants::TimeYearBase, // year
                   3,                                  // wday
                   0,                                  // yday
                   0}),
               tm_data);
  // 60 minutes from 1970-01-01 00:00:00 returns 1970-01-01 01:00:00.
  EXPECT_THAT(call_mktime(&tm_data,
                          1970, // year
                          1,    // month
                          1,    // day
                          0,    // hr
                          60,   // min
                          0,    // sec
                          0,    // wday
                          0),   // yday
              Succeeds(60 * TimeConstants::SecondsPerMin));
  EXPECT_TM_EQ((tm{0, // sec
                   0, // min
                   1, // hr
                   1, // day
                   0, // tm_mon starts with 0 for Jan
                   1970 - TimeConstants::TimeYearBase, // year
                   4,                                  // wday
                   0,                                  // yday
                   0}),
               tm_data);
}

TEST(LlvmLibcMkTime, MktimeTestsInvalidHours) {
  struct tm tm_data;
  // -1 hour from 1970-01-01 00:00:00 returns 1969-12-31 23:00:00.
  EXPECT_THAT(call_mktime(&tm_data,
                          1970, // year
                          1,    // month
                          1,    // day
                          -1,   // hr
                          0,    // min
                          0,    // sec
                          0,    // wday
                          0),   // yday
              Succeeds(-TimeConstants::SecondsPerHour));
  EXPECT_TM_EQ((tm{0,  // sec
                   0,  // min
                   23, // hr
                   31, // day
                   11, // tm_mon starts with 0 for Jan
                   1969 - TimeConstants::TimeYearBase, // year
                   3,                                  // wday
                   0,                                  // yday
                   0}),
               tm_data);
  // 24 hours from 1970-01-01 00:00:00 returns 1970-01-02 00:00:00.
  EXPECT_THAT(call_mktime(&tm_data,
                          1970, // year
                          1,    // month
                          1,    // day
                          24,   // hr
                          0,    // min
                          0,    // sec
                          0,    // wday
                          0),   // yday
              Succeeds(24 * TimeConstants::SecondsPerHour));
  EXPECT_TM_EQ((tm{0, // sec
                   0, // min
                   0, // hr
                   2, // day
                   0, // tm_mon starts with 0 for Jan
                   1970 - TimeConstants::TimeYearBase, // year
                   5,                                  // wday
                   0,                                  // yday
                   0}),
               tm_data);
}

TEST(LlvmLibcMkTime, MktimeTestsInvalidYear) {
  struct tm tm_data;
  // -1 year from 1970-01-01 00:00:00 returns 1969-01-01 00:00:00.
  EXPECT_THAT(call_mktime(&tm_data,
                          1969, // year
                          1,    // month
                          1,    // day
                          0,    // hr
                          0,    // min
                          0,    // sec
                          0,    // wday
                          0),   // yday
              Succeeds(-TimeConstants::DaysPerNonLeapYear *
                       TimeConstants::SecondsPerDay));
  EXPECT_TM_EQ((tm{0, // sec
                   0, // min
                   0, // hr
                   1, // day
                   0, // tm_mon starts with 0 for Jan
                   1969 - TimeConstants::TimeYearBase, // year
                   3,                                  // wday
                   0,                                  // yday
                   0}),
               tm_data);
}

TEST(LlvmLibcMkTime, MktimeTestsInvalidEndOf32BitEpochYear) {
  if (sizeof(size_t) != 4)
    return;
  struct tm tm_data;
  // 2038-01-19 03:14:08 tests overflow of the second in 2038.
  EXPECT_THAT(call_mktime(&tm_data, 2038, 1, 19, 3, 14, 8, 0, 0),
              Succeeds(TimeConstants::OutOfRangeReturnValue));
  // 2038-01-19 03:15:07 tests overflow of the minute in 2038.
  EXPECT_THAT(call_mktime(&tm_data, 2038, 1, 19, 3, 15, 7, 0, 0),
              Succeeds(TimeConstants::OutOfRangeReturnValue));
  // 2038-01-19 04:14:07 tests overflow of the hour in 2038.
  EXPECT_THAT(call_mktime(&tm_data, 2038, 1, 19, 4, 14, 7, 0, 0),
              Succeeds(TimeConstants::OutOfRangeReturnValue));
  // 2038-01-20 03:14:07 tests overflow of the day in 2038.
  EXPECT_THAT(call_mktime(&tm_data, 2038, 1, 20, 3, 14, 7, 0, 0),
              Succeeds(TimeConstants::OutOfRangeReturnValue));
  // 2038-02-19 03:14:07 tests overflow of the month in 2038.
  EXPECT_THAT(call_mktime(&tm_data, 2038, 2, 19, 3, 14, 7, 0, 0),
              Succeeds(TimeConstants::OutOfRangeReturnValue));
  // 2039-01-19 03:14:07 tests overflow of the year.
  EXPECT_THAT(call_mktime(&tm_data, 2039, 1, 19, 3, 14, 7, 0, 0),
              Succeeds(TimeConstants::OutOfRangeReturnValue));
}

TEST(LlvmLibcMkTime, MktimeTestsInvalidMonths) {
  struct tm tm_data;
  // -1 month from 1970-01-01 00:00:00 returns 1969-12-01 00:00:00.
  EXPECT_THAT(call_mktime(&tm_data,
                          1970, // year
                          0,    // month
                          1,    // day
                          0,    // hr
                          0,    // min
                          0,    // sec
                          0,    // wday
                          0),   // yday
              Succeeds(-31 * TimeConstants::SecondsPerDay));
  EXPECT_TM_EQ((tm{0,      // sec
                   0,      // min
                   0,      // hr
                   1,      // day
                   12 - 1, // tm_mon starts with 0 for Jan
                   1969 - TimeConstants::TimeYearBase, // year
                   1,                                  // wday
                   0,                                  // yday
                   0}),
               tm_data);
  // 1970-13-01 00:00:00 returns 1971-01-01 00:00:00.
  EXPECT_THAT(call_mktime(&tm_data,
                          1970, // year
                          13,   // month
                          1,    // day
                          0,    // hr
                          0,    // min
                          0,    // sec
                          0,    // wday
                          0),   // yday
              Succeeds(TimeConstants::DaysPerNonLeapYear *
                       TimeConstants::SecondsPerDay));
  EXPECT_TM_EQ((tm{0, // sec
                   0, // min
                   0, // hr
                   1, // day
                   0, // tm_mon starts with 0 for Jan
                   1971 - TimeConstants::TimeYearBase, // year
                   5,                                  // wday
                   0,                                  // yday
                   0}),
               tm_data);
}

TEST(LlvmLibcMkTime, MktimeTestsInvalidDays) {
  struct tm tm_data;
  // -1 day from 1970-01-01 00:00:00 returns 1969-12-31 00:00:00.
  EXPECT_THAT(call_mktime(&tm_data,
                          1970, // year
                          1,    // month
                          0,    // day
                          0,    // hr
                          0,    // min
                          0,    // sec
                          0,    // wday
                          0),   // yday
              Succeeds(-1 * TimeConstants::SecondsPerDay));
  EXPECT_TM_EQ((tm{0,  // sec
                   0,  // min
                   0,  // hr
                   31, // day
                   11, // tm_mon starts with 0 for Jan
                   1969 - TimeConstants::TimeYearBase, // year
                   3,                                  // wday
                   0,                                  // yday
                   0}),
               tm_data);

  // 1970-01-32 00:00:00 returns 1970-02-01 00:00:00.
  EXPECT_THAT(call_mktime(&tm_data,
                          1970, // year
                          1,    // month
                          32,   // day
                          0,    // hr
                          0,    // min
                          0,    // sec
                          0,    // wday
                          0),   // yday
              Succeeds(31 * TimeConstants::SecondsPerDay));
  EXPECT_TM_EQ((tm{0, // sec
                   0, // min
                   0, // hr
                   1, // day
                   0, // tm_mon starts with 0 for Jan
                   1970 - TimeConstants::TimeYearBase, // year
                   0,                                  // wday
                   0,                                  // yday
                   0}),
               tm_data);

  // 1970-02-29 00:00:00 returns 1970-03-01 00:00:00.
  EXPECT_THAT(call_mktime(&tm_data,
                          1970, // year
                          2,    // month
                          29,   // day
                          0,    // hr
                          0,    // min
                          0,    // sec
                          0,    // wday
                          0),   // yday
              Succeeds(59 * TimeConstants::SecondsPerDay));
  EXPECT_TM_EQ((tm{0, // sec
                   0, // min
                   0, // hr
                   1, // day
                   2, // tm_mon starts with 0 for Jan
                   1970 - TimeConstants::TimeYearBase, // year
                   0,                                  // wday
                   0,                                  // yday
                   0}),
               tm_data);

  // 1972-02-30 00:00:00 returns 1972-03-01 00:00:00.
  EXPECT_THAT(call_mktime(&tm_data,
                          1972, // year
                          2,    // month
                          30,   // day
                          0,    // hr
                          0,    // min
                          0,    // sec
                          0,    // wday
                          0),   // yday
              Succeeds(((2 * TimeConstants::DaysPerNonLeapYear) + 60) *
                       TimeConstants::SecondsPerDay));
  EXPECT_TM_EQ((tm{0, // sec
                   0, // min
                   0, // hr
                   1, // day
                   2, // tm_mon starts with 0 for Jan
                   1972 - TimeConstants::TimeYearBase, // year
                   3,                                  // wday
                   0,                                  // yday
                   0}),
               tm_data);
}

TEST(LlvmLibcMkTime, MktimeTestsEndOf32BitEpochYear) {
  struct tm tm_data;
  // Test for maximum value of a signed 32-bit integer.
  // Test implementation can encode time for Tue 19 January 2038 03:14:07 UTC.
  EXPECT_THAT(call_mktime(&tm_data,
                          2038, // year
                          1,    // month
                          19,   // day
                          3,    // hr
                          14,   // min
                          7,    // sec
                          0,    // wday
                          0),   // yday
              Succeeds(0x7FFFFFFF));
  EXPECT_TM_EQ((tm{7,  // sec
                   14, // min
                   3,  // hr
                   19, // day
                   0,  // tm_mon starts with 0 for Jan
                   2038 - TimeConstants::TimeYearBase, // year
                   2,                                  // wday
                   7,                                  // yday
                   0}),
               tm_data);
}

TEST(LlvmLibcMkTime, MktimeTests64BitYear) {
  if (sizeof(time_t) == 4)
    return;
  // Mon Jan 1 12:50:50 2170 (200 years from 1970),
  struct tm tm_data;
  EXPECT_THAT(call_mktime(&tm_data,
                          2170, // year
                          1,    // month
                          1,    // day
                          12,   // hr
                          50,   // min
                          50,   // sec
                          0,    // wday
                          0),   // yday
              Succeeds(6311479850));
  EXPECT_TM_EQ((tm{50, // sec
                   50, // min
                   12, // hr
                   1,  // day
                   0,  // tm_mon starts with 0 for Jan
                   2170 - TimeConstants::TimeYearBase, // year
                   1,                                  // wday
                   50,                                 // yday
                   0}),
               tm_data);

  // Test for Tue Jan 1 12:50:50 in 2,147,483,647th year.
  EXPECT_THAT(call_mktime(&tm_data, 2147483647, 1, 1, 12, 50, 50, 0, 0),
              Succeeds(67767976202043050));
  EXPECT_TM_EQ((tm{50, // sec
                   50, // min
                   12, // hr
                   1,  // day
                   0,  // tm_mon starts with 0 for Jan
                   2147483647 - TimeConstants::TimeYearBase, // year
                   2,                                        // wday
                   50,                                       // yday
                   0}),
               tm_data);
}
