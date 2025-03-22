//===-- Unittests for localtime -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "hdr/types/struct_tm.h"
#include "src/time/localtime.h"
#include "test/UnitTest/Test.h"

extern char **environ;

void set_env_var(char *env) {
  environ[0] = env;
  environ[1] = "\0";
}

TEST(LlvmLibcLocaltime, ValidUnixTimestamp0) {
  set_env_var("TZ=Europe/Paris");

#ifdef LIBC_TARGET_OS_IS_LINUX
  const time_t t_ptr = 0;
  struct tm *result = LIBC_NAMESPACE::time_utils::linux::localtime(&t_ptr);
  ASSERT_EQ(70, result->tm_year);
  ASSERT_EQ(0, result->tm_mon);
  ASSERT_EQ(1, result->tm_mday);
  ASSERT_EQ(1, result->tm_hour);
  ASSERT_EQ(0, result->tm_min);
  ASSERT_EQ(0, result->tm_sec);
  ASSERT_EQ(4, result->tm_wday);
  ASSERT_EQ(0, result->tm_yday);
  ASSERT_EQ(0, result->tm_isdst);
#endif
}
