//===-- Unittests for ctime -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "hdr/signal_macros.h"
#include "src/time/ctime.h"
#include "src/time/time_utils.h"
#include "test/UnitTest/ErrnoCheckingTest.h"
#include "test/UnitTest/Test.h"
#include "test/src/time/TmHelper.h"

using LlvmLibcCtime = LIBC_NAMESPACE::testing::ErrnoCheckingTest;

TEST_F(LlvmLibcCtime, nullptr) {
  EXPECT_DEATH([] { LIBC_NAMESPACE::ctime(nullptr); }, WITH_SIGNAL(-1));
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

TEST_F(LlvmLibcCtime, ValidUnixTimestamp2039) {
  time_t t;
  char *result;
  // 2039-01-01 00:00:00 UTC. This is after the 32-bit time_t max.
  t = 2177452800;
  result = LIBC_NAMESPACE::ctime(&t);
  ASSERT_STREQ("Sat Jan  1 00:00:00 2039\n", result);
}

TEST_F(LlvmLibcCtime, InvalidArgument) {
  time_t t;
  char *result;
  t = 253402300800; // 10000-01-01 00:00:00 UTC (overflows 26-byte buffer)
  result = LIBC_NAMESPACE::ctime(&t);
  ASSERT_ERRNO_EQ(LIBC_NAMESPACE::time_utils::TIME_OVERFLOW);
  ASSERT_STREQ(nullptr, result);
}
