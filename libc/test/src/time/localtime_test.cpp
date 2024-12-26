//===-- Unittests for localtime -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <stddef.h>
#include "src/time/localtime.h"
#include "src/time/timezone.h"
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

TEST(LlvmLibcLocaltime, ValidUnixTimestamp0) {
  set_env_var("TZ=Europe/Stockholm");

  const time_t t_ptr = 0;
  struct tm *result = LIBC_NAMESPACE::localtime(&t_ptr);
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

TEST(LlvmLibcLocaltime, ValidUnixTimestamp32Int) {
  set_env_var("TZ=Europe/Berlin");

  time_t t_ptr = 2147483647;
  struct tm *result = LIBC_NAMESPACE::localtime(&t_ptr);
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

/*TEST(LlvmLibcLocaltime, ValidUnixTimestamp32IntDst) {
  set_env_var("TZ=Europe/Berlin");

  time_t t_ptr = 1627225465;
  struct tm *result = LIBC_NAMESPACE::localtime(&t_ptr);
  ASSERT_EQ(121, result->tm_year);
  ASSERT_EQ(6, result->tm_mon);
  ASSERT_EQ(25, result->tm_mday);
  ASSERT_EQ(17, result->tm_hour);
  ASSERT_EQ(4, result->tm_min);
  ASSERT_EQ(25, result->tm_sec);
  ASSERT_EQ(0, result->tm_wday);
  ASSERT_EQ(205, result->tm_yday);
  ASSERT_EQ(1, result->tm_isdst);
}*/

/*TEST(LlvmLibcLocaltime, ValidUnixTimestampTzEnvironmentVariableUsaPst) {
  set_env_var("TZ=America/Los_Angeles");

  time_t t_ptr = 1627225465;
  struct tm *result = LIBC_NAMESPACE::localtime(&t_ptr);
  ASSERT_EQ(121, result->tm_year);
  ASSERT_EQ(6, result->tm_mon);
  ASSERT_EQ(25, result->tm_mday);
  ASSERT_EQ(8, result->tm_hour);
  ASSERT_EQ(4, result->tm_min);
  ASSERT_EQ(25, result->tm_sec);
  ASSERT_EQ(0, result->tm_wday);
  ASSERT_EQ(205, result->tm_yday);
  ASSERT_EQ(1, result->tm_isdst);
}*/

/*TEST(LlvmLibcLocaltime, ValidUnixTimestampTzEnvironmentVariableUsaEst) {
  set_env_var("TZ=America/New_York");

  time_t t_ptr = 1627225465;
  struct tm *result = LIBC_NAMESPACE::localtime(&t_ptr);
  ASSERT_EQ(121, result->tm_year);
  ASSERT_EQ(6, result->tm_mon);
  ASSERT_EQ(25, result->tm_mday);
  ASSERT_EQ(11, result->tm_hour);
  ASSERT_EQ(4, result->tm_min);
  ASSERT_EQ(25, result->tm_sec);
  ASSERT_EQ(0, result->tm_wday);
  ASSERT_EQ(205, result->tm_yday);
  ASSERT_EQ(1, result->tm_isdst);
}*/

TEST(LlvmLibcLocaltime, ValidUnixTimestampTzEnvironmentVariableUTC) {
  set_env_var("TZ=UTC");

  time_t t_ptr = 1627225465;
  struct tm *result = LIBC_NAMESPACE::localtime(&t_ptr);
  ASSERT_EQ(121, result->tm_year);
  ASSERT_EQ(6, result->tm_mon);
  ASSERT_EQ(25, result->tm_mday);
  ASSERT_EQ(15, result->tm_hour);
  ASSERT_EQ(4, result->tm_min);
  ASSERT_EQ(25, result->tm_sec);
  ASSERT_EQ(0, result->tm_wday);
  ASSERT_EQ(205, result->tm_yday);
  ASSERT_EQ(0, result->tm_isdst);
}

TEST(LlvmLibcLocaltime, ValidUnixTimestampTzEnvironmentVariableGMT) {
  set_env_var("TZ=GMT");

  time_t t_ptr = 1627225465;
  struct tm *result = LIBC_NAMESPACE::localtime(&t_ptr);
  ASSERT_EQ(121, result->tm_year);
  ASSERT_EQ(6, result->tm_mon);
  ASSERT_EQ(25, result->tm_mday);
  ASSERT_EQ(15, result->tm_hour);
  ASSERT_EQ(4, result->tm_min);
  ASSERT_EQ(25, result->tm_sec);
  ASSERT_EQ(0, result->tm_wday);
  ASSERT_EQ(205, result->tm_yday);
  ASSERT_EQ(1, result->tm_isdst);
}

TEST(LlvmLibcLocaltime, ValidUnixTimestampTzEnvironmentVariableEuropeBerlin) {
  set_env_var("TZ=Europe/Berlin");

  time_t t_ptr = 1627225465;
  struct tm *result = LIBC_NAMESPACE::localtime(&t_ptr);
  ASSERT_EQ(121, result->tm_year);
  ASSERT_EQ(6, result->tm_mon);
  ASSERT_EQ(25, result->tm_mday);
  ASSERT_EQ(17, result->tm_hour);
  ASSERT_EQ(4, result->tm_min);
  ASSERT_EQ(25, result->tm_sec);
  ASSERT_EQ(0, result->tm_wday);
  ASSERT_EQ(205, result->tm_yday);
  ASSERT_EQ(1, result->tm_isdst);
}

/*TEST(LlvmLibcLocaltime, ValidUnixTimestampTzEnvironmentVariableEuropeMoscow) {
  set_env_var("TZ=Europe/Moscow");

  time_t t_ptr = 1627225465;
  struct tm *result = LIBC_NAMESPACE::localtime(&t_ptr);
  ASSERT_EQ(121, result->tm_year);
  ASSERT_EQ(6, result->tm_mon);
  ASSERT_EQ(25, result->tm_mday);
  ASSERT_EQ(18, result->tm_hour);
  ASSERT_EQ(4, result->tm_min);
  ASSERT_EQ(25, result->tm_sec);
  ASSERT_EQ(0, result->tm_wday);
  ASSERT_EQ(205, result->tm_yday);
  ASSERT_EQ(1, result->tm_isdst);
}*/
