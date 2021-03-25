//===-- Unittests for gmtime ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/time/gmtime.h"
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

TEST(LlvmLibcGmTime, OutOfRange) {
  time_t seconds = 1 + INT_MAX * static_cast<int64_t>(
                                     TimeConstants::NumberOfSecondsInLeapYear);
  struct tm *tm_data = __llvm_libc::gmtime(&seconds);
  EXPECT_TRUE(tm_data == nullptr);
  EXPECT_EQ(llvmlibc_errno, EOVERFLOW);

  llvmlibc_errno = 0;
  seconds =
      INT_MIN * static_cast<int64_t>(TimeConstants::NumberOfSecondsInLeapYear) -
      1;
  tm_data = __llvm_libc::gmtime(&seconds);
  EXPECT_TRUE(tm_data == nullptr);
  EXPECT_EQ(llvmlibc_errno, EOVERFLOW);
}

TEST(LlvmLibcGmTime, InvalidSeconds) {
  time_t seconds = 0;
  struct tm *tm_data = nullptr;
  // -1 second from 1970-01-01 00:00:00 returns 1969-12-31 23:59:59.
  seconds = -1;
  tm_data = __llvm_libc::gmtime(&seconds);
  EXPECT_TM_EQ((tm{59,     // sec
                   59,     // min
                   23,     // hr
                   31,     // day
                   12 - 1, // tm_mon starts with 0 for Jan
                   1969 - TimeConstants::TimeYearBase, // year
                   3,                                  // wday
                   364,                                // yday
                   0}),
               *tm_data);
  // 60 seconds from 1970-01-01 00:00:00 returns 1970-01-01 00:01:00.
  seconds = 60;
  tm_data = __llvm_libc::gmtime(&seconds);
  EXPECT_TM_EQ((tm{0, // sec
                   1, // min
                   0, // hr
                   1, // day
                   0, // tm_mon starts with 0 for Jan
                   1970 - TimeConstants::TimeYearBase, // year
                   4,                                  // wday
                   0,                                  // yday
                   0}),
               *tm_data);
}

TEST(LlvmLibcGmTime, InvalidMinutes) {
  time_t seconds = 0;
  struct tm *tm_data = nullptr;
  // -1 minute from 1970-01-01 00:00:00 returns 1969-12-31 23:59:00.
  seconds = -TimeConstants::SecondsPerMin;
  tm_data = __llvm_libc::gmtime(&seconds);
  EXPECT_TM_EQ((tm{0,  // sec
                   59, // min
                   23, // hr
                   31, // day
                   11, // tm_mon starts with 0 for Jan
                   1969 - TimeConstants::TimeYearBase, // year
                   3,                                  // wday
                   0,                                  // yday
                   0}),
               *tm_data);
  // 60 minutes from 1970-01-01 00:00:00 returns 1970-01-01 01:00:00.
  seconds = 60 * TimeConstants::SecondsPerMin;
  tm_data = __llvm_libc::gmtime(&seconds);
  EXPECT_TM_EQ((tm{0, // sec
                   0, // min
                   1, // hr
                   1, // day
                   0, // tm_mon starts with 0 for Jan
                   1970 - TimeConstants::TimeYearBase, // year
                   4,                                  // wday
                   0,                                  // yday
                   0}),
               *tm_data);
}

TEST(LlvmLibcGmTime, InvalidHours) {
  time_t seconds = 0;
  struct tm *tm_data = nullptr;
  // -1 hour from 1970-01-01 00:00:00 returns 1969-12-31 23:00:00.
  seconds = -TimeConstants::SecondsPerHour;
  tm_data = __llvm_libc::gmtime(&seconds);
  EXPECT_TM_EQ((tm{0,  // sec
                   0,  // min
                   23, // hr
                   31, // day
                   11, // tm_mon starts with 0 for Jan
                   1969 - TimeConstants::TimeYearBase, // year
                   3,                                  // wday
                   0,                                  // yday
                   0}),
               *tm_data);
  // 24 hours from 1970-01-01 00:00:00 returns 1970-01-02 00:00:00.
  seconds = 24 * TimeConstants::SecondsPerHour;
  tm_data = __llvm_libc::gmtime(&seconds);
  EXPECT_TM_EQ((tm{0, // sec
                   0, // min
                   0, // hr
                   2, // day
                   0, // tm_mon starts with 0 for Jan
                   1970 - TimeConstants::TimeYearBase, // year
                   5,                                  // wday
                   0,                                  // yday
                   0}),
               *tm_data);
}

TEST(LlvmLibcGmTime, InvalidYear) {
  // -1 year from 1970-01-01 00:00:00 returns 1969-01-01 00:00:00.
  time_t seconds =
      -TimeConstants::DaysPerNonLeapYear * TimeConstants::SecondsPerDay;
  struct tm *tm_data = __llvm_libc::gmtime(&seconds);
  EXPECT_TM_EQ((tm{0, // sec
                   0, // min
                   0, // hr
                   1, // day
                   0, // tm_mon starts with 0 for Jan
                   1969 - TimeConstants::TimeYearBase, // year
                   3,                                  // wday
                   0,                                  // yday
                   0}),
               *tm_data);
}

TEST(LlvmLibcGmTime, InvalidMonths) {
  time_t seconds = 0;
  struct tm *tm_data = nullptr;
  // -1 month from 1970-01-01 00:00:00 returns 1969-12-01 00:00:00.
  seconds = -31 * TimeConstants::SecondsPerDay;
  tm_data = __llvm_libc::gmtime(&seconds);
  EXPECT_TM_EQ((tm{0,      // sec
                   0,      // min
                   0,      // hr
                   1,      // day
                   12 - 1, // tm_mon starts with 0 for Jan
                   1969 - TimeConstants::TimeYearBase, // year
                   1,                                  // wday
                   0,                                  // yday
                   0}),
               *tm_data);
  // 1970-13-01 00:00:00 returns 1971-01-01 00:00:00.
  seconds = TimeConstants::DaysPerNonLeapYear * TimeConstants::SecondsPerDay;
  tm_data = __llvm_libc::gmtime(&seconds);
  EXPECT_TM_EQ((tm{0, // sec
                   0, // min
                   0, // hr
                   1, // day
                   0, // tm_mon starts with 0 for Jan
                   1971 - TimeConstants::TimeYearBase, // year
                   5,                                  // wday
                   0,                                  // yday
                   0}),
               *tm_data);
}

TEST(LlvmLibcGmTime, InvalidDays) {
  time_t seconds = 0;
  struct tm *tm_data = nullptr;
  // -1 day from 1970-01-01 00:00:00 returns 1969-12-31 00:00:00.
  seconds = -1 * TimeConstants::SecondsPerDay;
  tm_data = __llvm_libc::gmtime(&seconds);
  EXPECT_TM_EQ((tm{0,  // sec
                   0,  // min
                   0,  // hr
                   31, // day
                   11, // tm_mon starts with 0 for Jan
                   1969 - TimeConstants::TimeYearBase, // year
                   3,                                  // wday
                   0,                                  // yday
                   0}),
               *tm_data);

  // 1970-01-32 00:00:00 returns 1970-02-01 00:00:00.
  seconds = 31 * TimeConstants::SecondsPerDay;
  tm_data = __llvm_libc::gmtime(&seconds);
  EXPECT_TM_EQ((tm{0, // sec
                   0, // min
                   0, // hr
                   1, // day
                   0, // tm_mon starts with 0 for Jan
                   1970 - TimeConstants::TimeYearBase, // year
                   0,                                  // wday
                   0,                                  // yday
                   0}),
               *tm_data);

  // 1970-02-29 00:00:00 returns 1970-03-01 00:00:00.
  seconds = 59 * TimeConstants::SecondsPerDay;
  tm_data = __llvm_libc::gmtime(&seconds);
  EXPECT_TM_EQ((tm{0, // sec
                   0, // min
                   0, // hr
                   1, // day
                   2, // tm_mon starts with 0 for Jan
                   1970 - TimeConstants::TimeYearBase, // year
                   0,                                  // wday
                   0,                                  // yday
                   0}),
               *tm_data);

  // 1972-02-30 00:00:00 returns 1972-03-01 00:00:00.
  seconds = ((2 * TimeConstants::DaysPerNonLeapYear) + 60) *
            TimeConstants::SecondsPerDay;
  tm_data = __llvm_libc::gmtime(&seconds);
  EXPECT_TM_EQ((tm{0, // sec
                   0, // min
                   0, // hr
                   1, // day
                   2, // tm_mon starts with 0 for Jan
                   1972 - TimeConstants::TimeYearBase, // year
                   3,                                  // wday
                   0,                                  // yday
                   0}),
               *tm_data);
}

TEST(LlvmLibcGmTime, EndOf32BitEpochYear) {
  // Test for maximum value of a signed 32-bit integer.
  // Test implementation can encode time for Tue 19 January 2038 03:14:07 UTC.
  time_t seconds = 0x7FFFFFFF;
  struct tm *tm_data = __llvm_libc::gmtime(&seconds);
  EXPECT_TM_EQ((tm{7,  // sec
                   14, // min
                   3,  // hr
                   19, // day
                   0,  // tm_mon starts with 0 for Jan
                   2038 - TimeConstants::TimeYearBase, // year
                   2,                                  // wday
                   7,                                  // yday
                   0}),
               *tm_data);
}

TEST(LlvmLibcGmTime, Max64BitYear) {
  if (sizeof(time_t) == 4)
    return;
  // Mon Jan 1 12:50:50 2170 (200 years from 1970),
  time_t seconds = 6311479850;
  struct tm *tm_data = __llvm_libc::gmtime(&seconds);
  EXPECT_TM_EQ((tm{50, // sec
                   50, // min
                   12, // hr
                   1,  // day
                   0,  // tm_mon starts with 0 for Jan
                   2170 - TimeConstants::TimeYearBase, // year
                   1,                                  // wday
                   50,                                 // yday
                   0}),
               *tm_data);

  // Test for Tue Jan 1 12:50:50 in 2,147,483,647th year.
  seconds = 67767976202043050;
  tm_data = __llvm_libc::gmtime(&seconds);
  EXPECT_TM_EQ((tm{50, // sec
                   50, // min
                   12, // hr
                   1,  // day
                   0,  // tm_mon starts with 0 for Jan
                   2147483647 - TimeConstants::TimeYearBase, // year
                   2,                                        // wday
                   50,                                       // yday
                   0}),
               *tm_data);
}
