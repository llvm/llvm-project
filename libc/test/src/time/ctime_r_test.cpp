//===-- Unittests for ctime_r ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/errno/libc_errno.h"
#include "src/time/ctime_r.h"
#include "src/time/time_utils.h"
#include "test/UnitTest/Test.h"
#include "test/src/time/TmHelper.h"

using LIBC_NAMESPACE::time_utils::TimeConstants;

static inline char *call_ctime_r(time_t *t, int year, int month, int mday,
                                 int hour, int min, int sec, int wday, int yday,
                                 char *buffer) {
  LIBC_NAMESPACE::tmhelper::testing::initialize_tm_data(
      localtime(t), year, month, mday, hour, min, sec, wday, yday);
  return LIBC_NAMESPACE::ctime_r(t, buffer);
}

// ctime and ctime_r share the same code and thus didn't repeat all the
// tests from ctime. Added couple of validation tests.
TEST(LlvmLibcCtimeR, Nullptr) {
  char *result;
  result = LIBC_NAMESPACE::ctime_r(nullptr, nullptr);
  ASSERT_ERRNO_EQ(EINVAL);
  ASSERT_STREQ(nullptr, result);

  char buffer[TimeConstants::CTIME_BUFFER_SIZE];
  result = LIBC_NAMESPACE::ctime_r(nullptr, buffer);
  ASSERT_ERRNO_EQ(EINVAL);
  ASSERT_STREQ(nullptr, result);

  time_t t;
  result = LIBC_NAMESPACE::ctime_r(&tm_data, nullptr);
  ASSERT_ERRNO_EQ(EINVAL);
  ASSERT_STREQ(nullptr, result);
}

TEST(LlvmLibcCtimeR, ValidDate) {
  char buffer[TimeConstants::CTIME_BUFFER_SIZE];
  struct time_t t;
  char *result;
  // 1970-01-01 00:00:00. Test with a valid buffer size.
  result = call_ctime_r(&t,
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
