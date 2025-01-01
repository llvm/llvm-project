//===-- Unittests for ctime -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/time/ctime.h"
#include "test/UnitTest/Test.h"

extern char **environ;

void set_env_var(char *env) {
    environ[0] = env;
    environ[1] = "\0";
}

TEST(LlvmLibcCtime, NULL) {
  char *result;
  result = LIBC_NAMESPACE::ctime(NULL);
  ASSERT_STREQ(NULL, result);
}

TEST(LlvmLibcCtime, ValidUnixTimestamp0) {
  set_env_var("TZ=Europe/Paris");

  time_t t;
  char *result;
  t = 0;
  result = LIBC_NAMESPACE::ctime(&t);
  ASSERT_STREQ("Thu Jan  1 01:00:00 1970\n", result);
}

TEST(LlvmLibcCtime, ValidUnixTimestamp32Int) {
  set_env_var("TZ=Europe/Berlin");

  time_t t;
  char *result;
  t = 2147483647;
  result = LIBC_NAMESPACE::ctime(&t);
  ASSERT_STREQ("Tue Jan 19 04:14:07 2038\n", result);
}

TEST(LlvmLibcCtime, InvalidArgument) {
  time_t t;
  char *result;
  t = 2147483648;
  result = LIBC_NAMESPACE::ctime(&t);
  ASSERT_STREQ(NULL, result);
}
