//===-- Unittests for localtime -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/time/localtime.h"
#include "test/UnitTest/Test.h"

TEST(LlvmLibcLocaltime, ValidUnixTimestamp0) {
  const time_t timer = 0;
  struct tm *result = LIBC_NAMESPACE::localtime(&timer);

  ASSERT_EQ(70, result->tm_year);
  ASSERT_EQ(0, result->tm_mon);
  ASSERT_EQ(1, result->tm_mday);
  ASSERT_EQ(0, result->tm_hour);
  ASSERT_EQ(0, result->tm_min);
  ASSERT_EQ(0, result->tm_sec);
  ASSERT_EQ(4, result->tm_wday);
  ASSERT_EQ(0, result->tm_yday);
  ASSERT_EQ(0, result->tm_isdst);
}

TEST(LlvmLibcLocaltime, NullPtr) {
  EXPECT_DEATH([] { LIBC_NAMESPACE::localtime(nullptr); }, WITH_SIGNAL(4));
}

// TODO(zimirza): These tests does not expect the correct output of localtime as
// per specification. This is due to timezone functions removed from
// https://github.com/llvm/llvm-project/pull/110363.
// This will be resolved a new pull request.

TEST(LlvmLibcLocaltime, ValidUnixTimestamp) {
  const time_t timer = 1756595338;
  struct tm *result = LIBC_NAMESPACE::localtime(&timer);

  ASSERT_EQ(125, result->tm_year);
  ASSERT_EQ(7, result->tm_mon);
  ASSERT_EQ(30, result->tm_mday);
  ASSERT_EQ(23, result->tm_hour);
  ASSERT_EQ(8, result->tm_min);
  ASSERT_EQ(58, result->tm_sec);
  ASSERT_EQ(6, result->tm_wday);
  ASSERT_EQ(241, result->tm_yday);
  ASSERT_EQ(0, result->tm_isdst);
}

TEST(LlvmLibcLocaltime, ValidUnixTimestampNegative) {
  const time_t timer = -1756595338;
  struct tm *result = LIBC_NAMESPACE::localtime(&timer);

  ASSERT_EQ(14, result->tm_year);
  ASSERT_EQ(4, result->tm_mon);
  ASSERT_EQ(4, result->tm_mday);
  ASSERT_EQ(0, result->tm_hour);
  ASSERT_EQ(51, result->tm_min);
  ASSERT_EQ(2, result->tm_sec);
  ASSERT_EQ(1, result->tm_wday);
  ASSERT_EQ(123, result->tm_yday);
  ASSERT_EQ(0, result->tm_isdst);
}
