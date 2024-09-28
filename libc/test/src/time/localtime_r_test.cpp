//===-- Unittests for localtime_r -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/errno/libc_errno.h"
#include "src/time/localtime_r.h"
#include "test/UnitTest/Test.h"
#include "test/src/time/TmHelper.h"

TEST(LlvmLibcLocaltimeR, ValidUnixTimestamp0) {
  const time_t t_ptr = 1;
  static struct tm input = (struct tm) {
      .tm_sec = 0,
      .tm_min = 0,
      .tm_hour = 0,
      .tm_mday = 0,
      .tm_mon = 0,
      .tm_year = 0,
      .tm_wday = 0,
      .tm_yday = 0,
      .tm_isdst = 0
  };
  struct tm *result = LIBC_NAMESPACE::localtime_r(&t_ptr, &input);
  ASSERT_EQ(70, result->tm_year);
  ASSERT_EQ(0, result->tm_mon);
  ASSERT_EQ(1, result->tm_mday);
  ASSERT_EQ(2, result->tm_hour);
  ASSERT_EQ(0, result->tm_min);
  ASSERT_EQ(1, result->tm_sec);
  ASSERT_EQ(4, result->tm_wday);
  ASSERT_EQ(0, result->tm_yday);
  ASSERT_EQ(0, result->tm_isdst);
}

TEST(LlvmLibcLocaltimeR, ValidUnixTimestamp32Int) {
  time_t t_ptr = 2147483647;
  static struct tm input = (struct tm) {
      .tm_sec = 0,
      .tm_min = 0,
      .tm_hour = 0,
      .tm_mday = 0,
      .tm_mon = 0,
      .tm_year = 0,
      .tm_wday = 0,
      .tm_yday = 0,
      .tm_isdst = 0
  };
  struct tm *result = LIBC_NAMESPACE::localtime_r(&t_ptr, &input);
  ASSERT_EQ(138, result->tm_year);
  ASSERT_EQ(0, result->tm_mon);
  ASSERT_EQ(19, result->tm_mday);
  ASSERT_EQ(5, result->tm_hour);
  ASSERT_EQ(14, result->tm_min);
  ASSERT_EQ(7, result->tm_sec);
  ASSERT_EQ(2, result->tm_wday);
  ASSERT_EQ(18, result->tm_yday);
  ASSERT_EQ(0, result->tm_isdst);
}

TEST(LlvmLibcLocaltimeR, ValidUnixTimestamp32IntDst) {
  time_t t_ptr = 1627225465;
  static struct tm input = (struct tm) {
      .tm_sec = 0,
      .tm_min = 0,
      .tm_hour = 0,
      .tm_mday = 0,
      .tm_mon = 0,
      .tm_year = 0,
      .tm_wday = 0,
      .tm_yday = 0,
      .tm_isdst = 0
  };
  struct tm *result = LIBC_NAMESPACE::localtime_r(&t_ptr, &input);
  ASSERT_EQ(121, result->tm_year);
  ASSERT_EQ(6, result->tm_mon);
  ASSERT_EQ(25, result->tm_mday);
  ASSERT_EQ(18, result->tm_hour);
  ASSERT_EQ(4, result->tm_min);
  ASSERT_EQ(25, result->tm_sec);
  ASSERT_EQ(0, result->tm_wday);
  ASSERT_EQ(205, result->tm_yday);
  ASSERT_EQ(1, result->tm_isdst);
}
