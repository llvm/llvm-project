//===-- Unittests for ctime -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/errno/libc_errno.h"
#include "src/time/ctime.h"
#include "test/UnitTest/Test.h"
#include "test/src/time/TmHelper.h"

TEST(LlvmLibcCtime, Nullptr) {
  char *result;
  result = LIBC_NAMESPACE::ctime(nullptr);
  ASSERT_ERRNO_EQ(EINVAL);
  ASSERT_STREQ(nullptr, result);
}

TEST(LlvmLibcCtime, ValidUnixTimestamp0) {
  time_t t = 0;
  char *result = LIBC_NAMESPACE::ctime(&t);
  ASSERT_STREQ("Thu Jan  1 00:00:00 1970\n", result);
}

TEST(LlvmLibcCtime, ValidUnixTimestamp32Int) {
  time_t t = 2147483647;
  char *result = LIBC_NAMESPACE::ctime(&t);
  ASSERT_STREQ("Tue Jan  19 03:14:07 2038\n", result);
}

TEST(LlvmLibcCtime, InvalidArgument) {
  time_t t = 2147483648;
  char *result = LIBC_NAMESPACE::ctime(&t);
  ASSERT_ERRNO_EQ(EINVAL);
  ASSERT_STREQ(nullptr, result);
}
