//===-- Unittests for localtime_s -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/time/localtime_s.h"
#include "src/time/time_utils.h"
#include "test/UnitTest/Test.h"
#include "test/src/time/TmHelper.h"

using LIBC_NAMESPACE::time_utils::TimeConstants;

/*TEST(LlvmLibcLocaltimeS, ValidUnixTimestamp0) {
  struct tm input = (struct tm){.tm_sec = 0,
                                .tm_min = 0,
                                .tm_hour = 0,
                                .tm_mday = 0,
                                .tm_mon = 0,
                                .tm_year = 0,
                                .tm_wday = 0,
                                .tm_yday = 0,
                                .tm_isdst = 0};
  time_t t_ptr = 0;
  int result = LIBC_NAMESPACE::localtime_s(&t_ptr, &input);
  ASSERT_EQ(0, result);

  ASSERT_EQ(70, input.tm_year);
  ASSERT_EQ(0, input.tm_mon);
  ASSERT_EQ(1, input.tm_mday);
  ASSERT_EQ(1, input.tm_hour);
  ASSERT_EQ(0, input.tm_min);
  ASSERT_EQ(0, input.tm_sec);
  ASSERT_EQ(4, input.tm_wday);
  ASSERT_EQ(0, input.tm_yday);
  ASSERT_EQ(0, input.tm_isdst);
}*/

/*TEST(LlvmLibcLocaltimeS, ValidUnixTimestamp32Int) {
  time_t t_ptr = 2147483647;
  struct tm input = (struct tm){.tm_sec = 0,
                                .tm_min = 0,
                                .tm_hour = 0,
                                .tm_mday = 0,
                                .tm_mon = 0,
                                .tm_year = 0,
                                .tm_wday = 0,
                                .tm_yday = 0,
                                .tm_isdst = 0};
  int result = LIBC_NAMESPACE::localtime_s(&t_ptr, &input);
  ASSERT_EQ(0, result);

  ASSERT_EQ(138, input.tm_year);
  ASSERT_EQ(0, input.tm_mon);
  ASSERT_EQ(19, input.tm_mday);
  ASSERT_EQ(4, input.tm_hour);
  ASSERT_EQ(14, input.tm_min);
  ASSERT_EQ(7, input.tm_sec);
  ASSERT_EQ(2, input.tm_wday);
  ASSERT_EQ(18, input.tm_yday);
  ASSERT_EQ(0, input.tm_isdst);
}*/

/*TEST(LlvmLibcLocaltimeS, ValidUnixTimestamp32IntDst) {
  time_t t_ptr = 1627225465;
  struct tm input = (struct tm){.tm_sec = 0,
                                .tm_min = 0,
                                .tm_hour = 0,
                                .tm_mday = 0,
                                .tm_mon = 0,
                                .tm_year = 0,
                                .tm_wday = 0,
                                .tm_yday = 0,
                                .tm_isdst = 0};
  int result = LIBC_NAMESPACE::localtime_s(&t_ptr, &input);
  ASSERT_EQ(0, result);

  ASSERT_EQ(121, input.tm_year);
  ASSERT_EQ(6, input.tm_mon);
  ASSERT_EQ(25, input.tm_mday);
  ASSERT_EQ(17, input.tm_hour);
  ASSERT_EQ(4, input.tm_min);
  ASSERT_EQ(25, input.tm_sec);
  ASSERT_EQ(0, input.tm_wday);
  ASSERT_EQ(205, input.tm_yday);
  ASSERT_EQ(1, input.tm_isdst);
}*/

TEST(LlvmLibcLocaltimeS, InvalidUnixTimestamp) {
  time_t t_ptr = -1;
  struct tm input = (struct tm){.tm_sec = 0,
                                .tm_min = 0,
                                .tm_hour = 0,
                                .tm_mday = 0,
                                .tm_mon = 0,
                                .tm_year = 0,
                                .tm_wday = 0,
                                .tm_yday = 0,
                                .tm_isdst = 0};
  int result = LIBC_NAMESPACE::localtime_s(&t_ptr, &input);
  ASSERT_EQ(-1, result);

  ASSERT_EQ(-1, input.tm_year);
  ASSERT_EQ(-1, input.tm_mon);
  ASSERT_EQ(-1, input.tm_mday);
  ASSERT_EQ(-1, input.tm_hour);
  ASSERT_EQ(-1, input.tm_min);
  ASSERT_EQ(-1, input.tm_sec);
  ASSERT_EQ(-1, input.tm_wday);
  ASSERT_EQ(-1, input.tm_yday);
  ASSERT_EQ(-1, input.tm_isdst);
}
