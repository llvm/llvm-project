//===-- Unittests for localtime_s -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/time/localtime_s.h"
#include "test/UnitTest/Test.h"

extern char **environ;

void set_env_var(char *env) {
  environ[0] = env;
  environ[1] = "\0";
}

TEST(LlvmLibcLocaltimeS, ValidUnixTimestamp0) {
  set_env_var("TZ=Europe/Paris");

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
}
