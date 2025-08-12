//===-- Unittests for asctime_r -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/libc_errno.h"
#include "src/time/asctime_r.h"
#include "src/time/time_constants.h"
#include "test/UnitTest/Test.h"
#include "test/src/time/TmHelper.h"

static inline char *call_asctime_r(struct tm *tm_data, int year, int month,
                                   int mday, int hour, int min, int sec,
                                   int wday, int yday, char *buffer) {
  LIBC_NAMESPACE::tmhelper::testing::initialize_tm_data(
      tm_data, year, month, mday, hour, min, sec, wday, yday);
  return LIBC_NAMESPACE::asctime_r(tm_data, buffer);
}

// asctime and asctime_r share the same code and thus didn't repeat all the
// tests from asctime. Added couple of validation tests.
TEST(LlvmLibcAsctimeR, Nullptr) {
  char *result;
  result = LIBC_NAMESPACE::asctime_r(nullptr, nullptr);
  ASSERT_ERRNO_EQ(EINVAL);
  ASSERT_STREQ(nullptr, result);

  char buffer[LIBC_NAMESPACE::time_constants::ASCTIME_BUFFER_SIZE];
  result = LIBC_NAMESPACE::asctime_r(nullptr, buffer);
  ASSERT_ERRNO_EQ(EINVAL);
  ASSERT_STREQ(nullptr, result);

  struct tm tm_data;
  result = LIBC_NAMESPACE::asctime_r(&tm_data, nullptr);
  ASSERT_ERRNO_EQ(EINVAL);
  ASSERT_STREQ(nullptr, result);
}

TEST(LlvmLibcAsctimeR, ValidDate) {
  char buffer[LIBC_NAMESPACE::time_constants::ASCTIME_BUFFER_SIZE];
  struct tm tm_data;
  char *result;
  // 1970-01-01 00:00:00. Test with a valid buffer size.
  result = call_asctime_r(&tm_data,
                          1970, // year
                          1,    // month
                          1,    // day
                          0,    // hr
                          0,    // min
                          0,    // sec
                          4,    // wday
                          0,    // yday
                          buffer);
  ASSERT_STREQ("Thu Jan  1 00:00:00 1970\n", result);
}
