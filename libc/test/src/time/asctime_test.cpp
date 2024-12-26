//===-- Unittests for asctime ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/errno/libc_errno.h"
#include "src/time/asctime.h"
#include "src/time/time_utils.h"
#include "test/UnitTest/Test.h"
#include "test/src/time/TmHelper.h"

static inline char *call_asctime(struct tm *tm_data, int year, int month,
                                 int mday, int hour, int min, int sec, int wday,
                                 int yday) {
  LIBC_NAMESPACE::tmhelper::testing::initialize_tm_data(
      tm_data, year, month, mday, hour, min, sec, wday, yday);
  return LIBC_NAMESPACE::asctime(tm_data);
}

TEST(LlvmLibcAsctime, Nullptr) {
  char *result;
  result = LIBC_NAMESPACE::asctime(nullptr);
  ASSERT_ERRNO_EQ(EINVAL);
  ASSERT_STREQ(nullptr, result);
}

// Weekdays are in the range 0 to 6. Test passing invalid value in wday.
TEST(LlvmLibcAsctime, InvalidWday) {
  struct tm tm_data;

  // Test with wday = -1.
  call_asctime(&tm_data,
               1970, // year
               1,    // month
               1,    // day
               0,    // hr
               0,    // min
               0,    // sec
               -1,   // wday
               0);   // yday
  ASSERT_ERRNO_EQ(EINVAL);

  // Test with wday = 7.
  call_asctime(&tm_data,
               1970, // year
               1,    // month
               1,    // day
               0,    // hr
               0,    // min
               0,    // sec
               7,    // wday
               0);   // yday
  ASSERT_ERRNO_EQ(EINVAL);
}

// Months are from January to December. Test passing invalid value in month.
TEST(LlvmLibcAsctime, InvalidMonth) {
  struct tm tm_data;

  // Test with month = 0.
  call_asctime(&tm_data,
               1970, // year
               0,    // month
               1,    // day
               0,    // hr
               0,    // min
               0,    // sec
               4,    // wday
               0);   // yday
  ASSERT_ERRNO_EQ(EINVAL);

  // Test with month = 13.
  call_asctime(&tm_data,
               1970, // year
               13,   // month
               1,    // day
               0,    // hr
               0,    // min
               0,    // sec
               4,    // wday
               0);   // yday
  ASSERT_ERRNO_EQ(EINVAL);
}

TEST(LlvmLibcAsctime, ValidWeekdays) {
  struct tm tm_data;
  char *result;
  // 1970-01-01 00:00:00.
  result = call_asctime(&tm_data,
                        1970, // year
                        1,    // month
                        1,    // day
                        0,    // hr
                        0,    // min
                        0,    // sec
                        4,    // wday
                        0);   // yday
  ASSERT_STREQ("Thu Jan  1 00:00:00 1970\n", result);

  // 1970-01-03 00:00:00.
  result = call_asctime(&tm_data,
                        1970, // year
                        1,    // month
                        3,    // day
                        0,    // hr
                        0,    // min
                        0,    // sec
                        6,    // wday
                        0);   // yday
  ASSERT_STREQ("Sat Jan  3 00:00:00 1970\n", result);

  // 1970-01-04 00:00:00.
  result = call_asctime(&tm_data,
                        1970, // year
                        1,    // month
                        4,    // day
                        0,    // hr
                        0,    // min
                        0,    // sec
                        0,    // wday
                        0);   // yday
  ASSERT_STREQ("Sun Jan  4 00:00:00 1970\n", result);
}

TEST(LlvmLibcAsctime, ValidMonths) {
  struct tm tm_data;
  char *result;
  // 1970-01-01 00:00:00.
  result = call_asctime(&tm_data,
                        1970, // year
                        1,    // month
                        1,    // day
                        0,    // hr
                        0,    // min
                        0,    // sec
                        4,    // wday
                        0);   // yday
  ASSERT_STREQ("Thu Jan  1 00:00:00 1970\n", result);

  // 1970-02-01 00:00:00.
  result = call_asctime(&tm_data,
                        1970, // year
                        2,    // month
                        1,    // day
                        0,    // hr
                        0,    // min
                        0,    // sec
                        0,    // wday
                        0);   // yday
  ASSERT_STREQ("Sun Feb  1 00:00:00 1970\n", result);

  // 1970-12-31 23:59:59.
  result = call_asctime(&tm_data,
                        1970, // year
                        12,   // month
                        31,   // day
                        23,   // hr
                        59,   // min
                        59,   // sec
                        4,    // wday
                        0);   // yday
  ASSERT_STREQ("Thu Dec 31 23:59:59 1970\n", result);
}

TEST(LlvmLibcAsctime, EndOf32BitEpochYear) {
  struct tm tm_data;
  char *result;
  // Test for maximum value of a signed 32-bit integer.
  // Test implementation can encode time for Tue 19 January 2038 03:14:07 UTC.
  result = call_asctime(&tm_data,
                        2038, // year
                        1,    // month
                        19,   // day
                        3,    // hr
                        14,   // min
                        7,    // sec
                        2,    // wday
                        7);   // yday
  ASSERT_STREQ("Tue Jan 19 03:14:07 2038\n", result);
}

TEST(LlvmLibcAsctime, Max64BitYear) {
  if (sizeof(time_t) == 4)
    return;
  // Mon Jan 1 12:50:50 2170 (200 years from 1970),
  struct tm tm_data;
  char *result;
  result = call_asctime(&tm_data,
                        2170, // year
                        1,    // month
                        1,    // day
                        12,   // hr
                        50,   // min
                        50,   // sec
                        1,    // wday
                        50);  // yday
  ASSERT_STREQ("Mon Jan  1 12:50:50 2170\n", result);

  // Test for Tue Jan 1 12:50:50 in 2,147,483,647th year.
  // This test would cause buffer overflow and thus asctime returns nullptr.
  result = call_asctime(&tm_data,
                        2147483647, // year
                        1,          // month
                        1,          // day
                        12,         // hr
                        50,         // min
                        50,         // sec
                        2,          // wday
                        50);        // yday
  ASSERT_ERRNO_EQ(EOVERFLOW);
  ASSERT_STREQ(nullptr, result);
}
