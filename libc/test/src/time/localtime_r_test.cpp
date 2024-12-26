//===-- Unittests for localtime_r -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/time/localtime_r.h"
#include "src/time/time_utils.h"
#include "test/UnitTest/Test.h"

// TODO: remove this header file
#include <string.h>

extern char **environ;

// TODO: rewrite this function and remove malloc
void set_env_var(const char *env) {
  int i = 0;
  if (environ[i] != NULL) {
    i++;
  }

  environ[i] = (char *)malloc(strlen(env) + 1);
  if (environ[i] != NULL) {
    memcpy(environ[i], env, strlen(env) + 1);
    environ[i + 1] = NULL;
  }
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

  ASSERT_EQ(70, result->tm_year);
  ASSERT_EQ(0, result->tm_mon);
  ASSERT_EQ(1, result->tm_mday);
  ASSERT_EQ(1, result->tm_hour);
  ASSERT_EQ(0, result->tm_min);
  ASSERT_EQ(0, result->tm_sec);
  ASSERT_EQ(4, result->tm_wday);
  ASSERT_EQ(0, result->tm_yday);
  ASSERT_EQ(0, result->tm_isdst);
}

TEST(LlvmLibcLocaltimeR, ValidUnixTimestamp32Int) {
  set_env_var("TZ=Europe/Berlin");

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
  struct tm *result = LIBC_NAMESPACE::localtime_r(&t_ptr, &input);

  ASSERT_EQ(138, input.tm_year);
  ASSERT_EQ(0, input.tm_mon);
  ASSERT_EQ(19, input.tm_mday);
  ASSERT_EQ(4, input.tm_hour);
  ASSERT_EQ(14, input.tm_min);
  ASSERT_EQ(7, input.tm_sec);
  ASSERT_EQ(2, input.tm_wday);
  ASSERT_EQ(18, input.tm_yday);
  ASSERT_EQ(0, input.tm_isdst);

  ASSERT_EQ(138, result->tm_year);
  ASSERT_EQ(0, result->tm_mon);
  ASSERT_EQ(19, result->tm_mday);
  ASSERT_EQ(4, result->tm_hour);
  ASSERT_EQ(14, result->tm_min);
  ASSERT_EQ(7, result->tm_sec);
  ASSERT_EQ(2, result->tm_wday);
  ASSERT_EQ(18, result->tm_yday);
  ASSERT_EQ(0, result->tm_isdst);
}

TEST(LlvmLibcLocaltimeR, ValidUnixTimestamp32IntDst) {
  set_env_var("TZ=Europe/Berlin");

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
  struct tm *result = LIBC_NAMESPACE::localtime_r(&t_ptr, &input);

  ASSERT_EQ(121, input.tm_year);
  ASSERT_EQ(6, input.tm_mon);
  ASSERT_EQ(25, input.tm_mday);
  // ASSERT_EQ(17, input.tm_hour);
  ASSERT_EQ(4, input.tm_min);
  ASSERT_EQ(25, input.tm_sec);
  ASSERT_EQ(0, input.tm_wday);
  ASSERT_EQ(205, input.tm_yday);
  // ASSERT_EQ(1, input.tm_isdst);

  ASSERT_EQ(121, result->tm_year);
  ASSERT_EQ(6, result->tm_mon);
  ASSERT_EQ(25, result->tm_mday);
  // ASSERT_EQ(17, result->tm_hour);
  ASSERT_EQ(4, result->tm_min);
  ASSERT_EQ(25, result->tm_sec);
  ASSERT_EQ(0, result->tm_wday);
  ASSERT_EQ(205, result->tm_yday);
  // ASSERT_EQ(1, result->tm_isdst);
}
