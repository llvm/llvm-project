//===-- Unittests for ctime_r ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/errno/libc_errno.h"
#include "src/time/ctime_r.h"
#include "src/time/time_utils.h"
#include "test/UnitTest/Test.h"
#include "test/src/time/TmHelper.h"

using LIBC_NAMESPACE::time_utils::TimeConstants;

TEST(LlvmLibcCtimeR, Nullptr) {
  char *result;
  result = LIBC_NAMESPACE::ctime_r(nullptr, nullptr);
  ASSERT_STREQ(nullptr, result);

  char buffer[TimeConstants::ASCTIME_BUFFER_SIZE];
  result = LIBC_NAMESPACE::ctime_r(nullptr, buffer);
  ASSERT_STREQ(nullptr, result);

  time_t t;
  result = LIBC_NAMESPACE::ctime_r(&t, nullptr);
  ASSERT_STREQ(nullptr, result);
}

TEST(LlvmLibcCtimeR, ValidUnixTimestamp0) {
  char buffer[TimeConstants::ASCTIME_BUFFER_SIZE];
  time_t t;
  char *result;
  // 1970-01-01 00:00:00. Test with a valid buffer size.
  t = 0;
  result = LIBC_NAMESPACE::ctime_r(&t, buffer);
  ASSERT_STREQ("Thu Jan  1 00:00:00 1970\n", result);
}

TEST(LlvmLibcCtime, ValidUnixTimestamp32Int) {
  char buffer[TimeConstants::ASCTIME_BUFFER_SIZE];
  time_t t;
  char *result;
  // 2038-01-19 03:14:07. Test with a valid buffer size.
  t = 2147483647;
  result = LIBC_NAMESPACE::ctime_r(&t, buffer);
  ASSERT_STREQ("Tue Jan 19 03:14:07 2038\n", result);
}

TEST(LlvmLibcCtimeR, InvalidArgument) {
  char buffer[TimeConstants::ASCTIME_BUFFER_SIZE];
  time_t t;
  char *result;
  t = 2147483648;
  result = LIBC_NAMESPACE::ctime_r(&t, buffer);
  ASSERT_STREQ(nullptr, result);
}
