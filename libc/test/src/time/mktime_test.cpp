//===-- Unittests for mktime ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/time/mktime.h"
#include "test/ErrnoSetterMatcher.h"
#include "utils/UnitTest/Test.h"

#include <errno.h>
#include <string.h>

using __llvm_libc::testing::ErrnoSetterMatcher::Fails;

static constexpr time_t OutOfRangeReturnValue = -1;

// A helper function to initialize tm data structure.
static inline void initialize_tm_data(struct tm *tm_data, int year, int month,
                                      int mday, int hour, int min, int sec) {
  struct tm temp = {.tm_sec = sec,
                    .tm_min = min,
                    .tm_hour = hour,
                    .tm_mday = mday,
                    .tm_mon = month,
                    .tm_year = year - 1900};
  *tm_data = temp;
}

static inline time_t call_mktime(struct tm *tm_data, int year, int month,
                                 int mday, int hour, int min, int sec) {
  initialize_tm_data(tm_data, year, month, mday, hour, min, sec);
  return __llvm_libc::mktime(tm_data);
}

TEST(MkTime, FailureSetsErrno) {
  struct tm tm_data;
  initialize_tm_data(&tm_data, 0, 0, 0, 0, 0, -1);
  EXPECT_THAT(__llvm_libc::mktime(&tm_data), Fails(EOVERFLOW));
}

TEST(MkTime, MktimeTestsInvalidSeconds) {
  struct tm tm_data;
  EXPECT_EQ(call_mktime(&tm_data, 0, 0, 0, 0, 0, -1), OutOfRangeReturnValue);
  EXPECT_EQ(call_mktime(&tm_data, 0, 0, 0, 0, 0, 60), OutOfRangeReturnValue);
}

TEST(MkTime, MktimeTestsInvalidMinutes) {
  struct tm tm_data;
  EXPECT_EQ(call_mktime(&tm_data, 0, 0, 0, 0, -1, 0), OutOfRangeReturnValue);
  EXPECT_EQ(call_mktime(&tm_data, 0, 0, 1, 0, 60, 0), OutOfRangeReturnValue);
}

TEST(MkTime, MktimeTestsInvalidHours) {
  struct tm tm_data;
  EXPECT_EQ(call_mktime(&tm_data, 0, 0, 0, -1, 0, 0), OutOfRangeReturnValue);
  EXPECT_EQ(call_mktime(&tm_data, 0, 0, 0, 24, 0, 0), OutOfRangeReturnValue);
}

TEST(MkTime, MktimeTestsInvalidYear) {
  struct tm tm_data;
  EXPECT_EQ(call_mktime(&tm_data, 1969, 0, 0, 0, 0, 0), OutOfRangeReturnValue);
}

TEST(MkTime, MktimeTestsInvalidEndOf32BitEpochYear) {
  if (sizeof(time_t) != 4)
    return;
  struct tm tm_data;
  // 2038-01-19 03:14:08 tests overflow of the second in 2038.
  EXPECT_EQ(call_mktime(&tm_data, 2038, 0, 19, 3, 14, 8),
            OutOfRangeReturnValue);
  // 2038-01-19 03:15:07 tests overflow of the minute in 2038.
  EXPECT_EQ(call_mktime(&tm_data, 2038, 0, 19, 3, 15, 7),
            OutOfRangeReturnValue);
  // 2038-01-19 04:14:07 tests overflow of the hour in 2038.
  EXPECT_EQ(call_mktime(&tm_data, 2038, 0, 19, 4, 14, 7),
            OutOfRangeReturnValue);
  // 2038-01-20 03:14:07 tests overflow of the day in 2038.
  EXPECT_EQ(call_mktime(&tm_data, 2038, 0, 20, 3, 14, 7),
            OutOfRangeReturnValue);
  // 2038-02-19 03:14:07 tests overflow of the month in 2038.
  EXPECT_EQ(call_mktime(&tm_data, 2038, 1, 19, 3, 14, 7),
            OutOfRangeReturnValue);
  // 2039-01-19 03:14:07 tests overflow of the year.
  EXPECT_EQ(call_mktime(&tm_data, 2039, 0, 19, 3, 14, 7),
            OutOfRangeReturnValue);
}

TEST(MkTime, MktimeTestsInvalidMonths) {
  struct tm tm_data;
  // Before Jan of 1970
  EXPECT_EQ(call_mktime(&tm_data, 1970, -1, 15, 0, 0, 0),
            OutOfRangeReturnValue);
  // After Dec of 1970
  EXPECT_EQ(call_mktime(&tm_data, 1970, 12, 15, 0, 0, 0),
            OutOfRangeReturnValue);
}

TEST(MkTime, MktimeTestsInvalidDays) {
  struct tm tm_data;
  // -1 day of Jan, 1970
  EXPECT_EQ(call_mktime(&tm_data, 1970, 0, -1, 0, 0, 0), OutOfRangeReturnValue);
  // 32 day of Jan, 1970
  EXPECT_EQ(call_mktime(&tm_data, 1970, 0, 32, 0, 0, 0), OutOfRangeReturnValue);
  // 29 day of Feb, 1970
  EXPECT_EQ(call_mktime(&tm_data, 1970, 1, 29, 0, 0, 0), OutOfRangeReturnValue);
  // 30 day of Feb, 1972
  EXPECT_EQ(call_mktime(&tm_data, 1972, 1, 30, 0, 0, 0), OutOfRangeReturnValue);
  // 31 day of Apr, 1970
  EXPECT_EQ(call_mktime(&tm_data, 1970, 3, 31, 0, 0, 0), OutOfRangeReturnValue);
}

TEST(MkTime, MktimeTestsStartEpochYear) {
  // Thu Jan 1 00:00:00 1970
  struct tm tm_data;
  EXPECT_EQ(call_mktime(&tm_data, 1970, 0, 1, 0, 0, 0), static_cast<time_t>(0));
  EXPECT_EQ(4, tm_data.tm_wday);
  EXPECT_EQ(0, tm_data.tm_yday);
}

TEST(MkTime, MktimeTestsEpochYearRandomTime) {
  // Thu Jan 1 12:50:50 1970
  struct tm tm_data;
  EXPECT_EQ(call_mktime(&tm_data, 1970, 0, 1, 12, 50, 50),
            static_cast<time_t>(46250));
  EXPECT_EQ(4, tm_data.tm_wday);
  EXPECT_EQ(0, tm_data.tm_yday);
}

TEST(MkTime, MktimeTestsEndOf32BitEpochYear) {
  struct tm tm_data;
  // Test for maximum value of a signed 32-bit integer.
  // Test implementation can encode time for Tue 19 January 2038 03:14:07 UTC.
  EXPECT_EQ(call_mktime(&tm_data, 2038, 0, 19, 3, 14, 7),
            static_cast<time_t>(0x7FFFFFFF));
  EXPECT_EQ(2, tm_data.tm_wday);
  EXPECT_EQ(18, tm_data.tm_yday);
}

TEST(MkTime, MktimeTests64BitYear) {
  if (sizeof(time_t) == 4)
    return;
  // Mon Jan 1 12:50:50 2170
  struct tm tm_data;
  EXPECT_EQ(call_mktime(&tm_data, 2170, 0, 1, 12, 50, 50),
            static_cast<time_t>(6311479850));
  EXPECT_EQ(1, tm_data.tm_wday);
  EXPECT_EQ(0, tm_data.tm_yday);

  // Test for Tue Jan 1 12:50:50 in 2,147,483,647th year.
  EXPECT_EQ(call_mktime(&tm_data, 2147483647, 0, 1, 12, 50, 50),
            static_cast<time_t>(67767976202043050));
  EXPECT_EQ(2, tm_data.tm_wday);
  EXPECT_EQ(0, tm_data.tm_yday);
}
