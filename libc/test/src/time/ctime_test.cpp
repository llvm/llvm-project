//===-- Unittests for ctime -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/time/ctime.h"
#include "test/UnitTest/ErrnoCheckingTest.h"
#include "test/UnitTest/Test.h"
#include "test/src/time/TmHelper.h"

using LlvmLibcCtime = LIBC_NAMESPACE::testing::ErrnoCheckingTest;

TEST_F(LlvmLibcCtime, nullptr) {
  char *result;
  result = LIBC_NAMESPACE::ctime(nullptr);
  ASSERT_STREQ(nullptr, result);
}

TEST_F(LlvmLibcCtime, ValidUnixTimestamp0) {
  time_t t;
  char *result;
  t = 0;
  result = LIBC_NAMESPACE::ctime(&t);
  ASSERT_STREQ("Thu Jan  1 00:00:00 1970\n", result);
}

TEST_F(LlvmLibcCtime, ValidUnixTimestamp32Int) {
  time_t t;
  char *result;
  t = 2147483647;
  result = LIBC_NAMESPACE::ctime(&t);
  ASSERT_STREQ("Tue Jan 19 03:14:07 2038\n", result);
}

TEST_F(LlvmLibcCtime, InvalidArgument) {
  time_t t;
  char *result;
  t = 2147483648;
  result = LIBC_NAMESPACE::ctime(&t);
  ASSERT_STREQ(nullptr, result);
}
