//===-- Unittests for localtime_s -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/time/localtime_s.h"
#include "src/time/mktime.h"
#include "src/time/time_utils.h"
#include "test/UnitTest/Test.h"
#include "test/src/time/TmHelper.h"

using LIBC_NAMESPACE::time_utils::TimeConstants;

TEST(LlvmLibcLocaltimeS, ValidUnixTimestamp0) {
  struct tm input;
  const time_t t_ptr = 0;
  int result = LIBC_NAMESPACE::localtime_s(&t_ptr, &input);
  ASSERT_EQ(-1, result);
}

TEST(LlvmLibcLocaltimeS, ValidUnixTimestamp32Int) {
  time_t t_ptr = 2147483647;
  static struct tm input = (struct tm){.tm_sec = 0,
                                       .tm_min = 0,
                                       .tm_hour = 0,
                                       .tm_mday = 0,
                                       .tm_mon = 0,
                                       .tm_year = 0,
                                       .tm_wday = 0,
                                       .tm_yday = 0,
                                       .tm_isdst = 0};
  int result = LIBC_NAMESPACE::localtime_s(&t_ptr, &input);
  ASSERT_EQ(1, result);
}

TEST(LlvmLibcLocaltimeS, ValidUnixTimestamp32IntDst) {
  time_t t_ptr = 1627225465;
  static struct tm input = (struct tm){.tm_sec = 0,
                                       .tm_min = 0,
                                       .tm_hour = 0,
                                       .tm_mday = 0,
                                       .tm_mon = 0,
                                       .tm_year = 0,
                                       .tm_wday = 0,
                                       .tm_yday = 0,
                                       .tm_isdst = 0};
  int result = LIBC_NAMESPACE::localtime_s(&t_ptr, &input);
  ASSERT_EQ(1, result);
}
