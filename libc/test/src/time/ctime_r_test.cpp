//===-- Unittests for ctime_r ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "hdr/signal_macros.h"
#include "src/time/ctime_r.h"
#include "src/time/time_constants.h"
#include "src/time/time_utils.h"
#include "test/UnitTest/ErrnoCheckingTest.h"
#include "test/UnitTest/Test.h"
#include "test/src/time/TmHelper.h"

using LlvmLibcCtimeR = LIBC_NAMESPACE::testing::ErrnoCheckingTest;

TEST_F(LlvmLibcCtimeR, Nullptr) {
  char buffer[LIBC_NAMESPACE::time_constants::ASCTIME_BUFFER_SIZE];
  time_t t;
  EXPECT_DEATH([] { LIBC_NAMESPACE::ctime_r(nullptr, nullptr); },
               WITH_SIGNAL(-1));
  EXPECT_DEATH([&] { LIBC_NAMESPACE::ctime_r(nullptr, buffer); },
               WITH_SIGNAL(-1));
  EXPECT_DEATH([&] { LIBC_NAMESPACE::ctime_r(&t, nullptr); }, WITH_SIGNAL(-1));
}

TEST_F(LlvmLibcCtimeR, ValidUnixTimestamp0) {
  char buffer[LIBC_NAMESPACE::time_constants::ASCTIME_BUFFER_SIZE];
  time_t t;
  char *result;
  // 1970-01-01 00:00:00. Test with a valid buffer size.
  t = 0;
  result = LIBC_NAMESPACE::ctime_r(&t, buffer);
  ASSERT_STREQ("Thu Jan  1 00:00:00 1970\n", result);
}

TEST_F(LlvmLibcCtimeR, ValidUnixTimestamp32Int) {
  char buffer[LIBC_NAMESPACE::time_constants::ASCTIME_BUFFER_SIZE];
  time_t t;
  char *result;
  // 2038-01-19 03:14:07. Test with a valid buffer size.
  t = 2147483647;
  result = LIBC_NAMESPACE::ctime_r(&t, buffer);
  ASSERT_STREQ("Tue Jan 19 03:14:07 2038\n", result);
}

TEST_F(LlvmLibcCtimeR, ValidUnixTimestamp2039) {
  char buffer[LIBC_NAMESPACE::time_constants::ASCTIME_BUFFER_SIZE];
  time_t t;
  char *result;
  // 2039-01-01 00:00:00 UTC. Test with a valid buffer size. This is after the
  // 32-bit time_t max.
  t = 2177452800;
  result = LIBC_NAMESPACE::ctime_r(&t, buffer);
  ASSERT_STREQ("Sat Jan  1 00:00:00 2039\n", result);
}

TEST_F(LlvmLibcCtimeR, InvalidArgument) {
  char buffer[LIBC_NAMESPACE::time_constants::ASCTIME_BUFFER_SIZE];
  time_t t;
  char *result;
  t = 253402300800; // 10000-01-01 00:00:00 UTC (overflows 26-byte buffer)
  result = LIBC_NAMESPACE::ctime_r(&t, buffer);
  ASSERT_ERRNO_EQ(LIBC_NAMESPACE::time_utils::TIME_OVERFLOW);
  ASSERT_STREQ(nullptr, result);
}
