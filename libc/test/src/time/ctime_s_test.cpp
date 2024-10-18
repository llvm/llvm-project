//===-- Unittests for ctime_s ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "hdr/errno_macros.h"
#include "src/errno/libc_errno.h"
#include "src/time/ctime_s.h"
#include "src/time/time_utils.h"
#include "test/UnitTest/Test.h"
#include "test/src/time/TmHelper.h"

using LIBC_NAMESPACE::time_utils::TimeConstants;

TEST(LlvmLibcCtimeS, Nullptr) {
  int result;
  result = LIBC_NAMESPACE::ctime_s(nullptr, 0, nullptr);
  ASSERT_EQ(EINVAL, result);

  char buffer[TimeConstants::ASCTIME_BUFFER_SIZE];
  result = LIBC_NAMESPACE::ctime_s(buffer, sizeof(buffer), nullptr);
  ASSERT_EQ(EINVAL, result);

  time_t t;
  result = LIBC_NAMESPACE::ctime_s(nullptr, 0, &t);
  ASSERT_EQ(EINVAL, result);
}

TEST(LlvmLibcCtimeS, ValidUnixTimestamp0) {
  char buffer[TimeConstants::ASCTIME_BUFFER_SIZE];
  time_t t;
  int result;
  // 1970-01-01 00:00:00. Test with a valid buffer size.
  t = 0;
  result = LIBC_NAMESPACE::ctime_s(buffer, sizeof(buffer), &t);
  ASSERT_STREQ("Thu Jan  1 00:00:00 1970\n", buffer);
  ASSERT_EQ(0, result);
}

TEST(LlvmLibcCtimeS, ValidUnixTimestamp32Int) {
  char buffer[TimeConstants::ASCTIME_BUFFER_SIZE];
  time_t t;
  int result;
  // 2038-01-19 03:14:07. Test with a valid buffer size.
  t = 2147483647;
  result = LIBC_NAMESPACE::ctime_s(buffer, sizeof(buffer), &t);
  ASSERT_STREQ("Tue Jan 19 03:14:07 2038\n", buffer);
  ASSERT_EQ(0, result);
}

TEST(LlvmLibcCtimeS, InvalidArgument) {
  char buffer[TimeConstants::ASCTIME_BUFFER_SIZE];
  time_t t;
  int result;
  t = 2147483648;
  result = LIBC_NAMESPACE::ctime_s(buffer, sizeof(buffer), &t);
  ASSERT_EQ(EINVAL, result);
}
