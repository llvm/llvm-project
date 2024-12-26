//===-- Unittests for gmtime_r --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/time/gmtime_r.h"
#include "src/time/time_constants.h"
#include "src/time/time_utils.h"
#include "src/time/localtime.h"
#include "test/UnitTest/Test.h"
#include "test/src/time/TmMatcher.h"

// gmtime and gmtime_r share the same code and thus didn't repeat all the tests
// from gmtime. Added couple of validation tests.
TEST(LlvmLibcGmTimeR, EndOf32BitEpochYear) {
  // Test for maximum value of a signed 32-bit integer.
  // Test implementation can encode time for Tue 19 January 2038 03:14:07 UTC.
  time_t seconds = 0x7FFFFFFF;
  struct tm tm_data;
  struct tm *tm_data_ptr;
  tm_data_ptr = LIBC_NAMESPACE::gmtime_r(&seconds, &tm_data);
  EXPECT_TM_EQ(
      (tm{7,  // sec
          14, // min
          3,  // hr
          19, // day
          0,  // tm_mon starts with 0 for Jan
          2038 - LIBC_NAMESPACE::time_constants::TIME_YEAR_BASE, // year
          2,                                                     // wday
          7,                                                     // yday
          0}),
      *tm_data_ptr);
  EXPECT_TM_EQ(*tm_data_ptr, tm_data);
}

TEST(LlvmLibcGmTimeR, Max64BitYear) {
  if (sizeof(time_t) == 4)
    return;
  // Test for Tue Jan 1 12:50:50 in 2,147,483,647th year.
  time_t seconds = 67767976202043050;
  struct tm tm_data;
  struct tm *tm_data_ptr;
  tm_data_ptr = LIBC_NAMESPACE::gmtime_r(&seconds, &tm_data);
  EXPECT_TM_EQ(
      (tm{50, // sec
          50, // min
          12, // hr
          1,  // day
          0,  // tm_mon starts with 0 for Jan
          2147483647 - LIBC_NAMESPACE::time_constants::TIME_YEAR_BASE, // year
          2,                                                           // wday
          50,                                                          // yday
          0}),
      *tm_data_ptr);
  EXPECT_TM_EQ(*tm_data_ptr, tm_data);
}
