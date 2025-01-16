//===-- Unittests for localtime_r -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/time/localtime_r.h"
#include "test/UnitTest/Test.h"

extern char **environ;

void set_env_var(char *env) {
  environ[0] = env;
  environ[1] = "\0";
}

TEST(LlvmLibcLocaltimeR, ValidUnixTimestamp0) {
  set_env_var("TZ=Europe/Berlin");

  struct tm input;
  time_t t_ptr = 0;
  struct tm *result = LIBC_NAMESPACE::localtime_r(&t_ptr, &input);

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
