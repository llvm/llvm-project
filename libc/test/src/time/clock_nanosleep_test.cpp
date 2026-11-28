//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unittests for clock_nanosleep.
///
//===----------------------------------------------------------------------===//

#include "hdr/errno_macros.h"
#include "hdr/time_macros.h"
#include "hdr/types/clockid_t.h"
#include "hdr/types/struct_timespec.h"
#include "src/time/clock_nanosleep.h"
#include "test/UnitTest/ErrnoCheckingTest.h"
#include "test/UnitTest/Test.h"

using LlvmLibcClockNanosleep = LIBC_NAMESPACE::testing::ErrnoCheckingTest;

TEST_F(LlvmLibcClockNanosleep, InvalidClock) {
  struct timespec tim = {0, 500};
  // Invalid clock id should return EINVAL directly without setting errno.
  int ret = LIBC_NAMESPACE::clock_nanosleep(clockid_t(-1), 0, &tim, nullptr);
  EXPECT_EQ(ret, EINVAL);
}

TEST_F(LlvmLibcClockNanosleep, InvalidNsec) {
  struct timespec tim = {0, 1000000000L}; // tv_nsec >= 10^9
  int ret = LIBC_NAMESPACE::clock_nanosleep(CLOCK_REALTIME, 0, &tim, nullptr);
  EXPECT_EQ(ret, EINVAL);

  struct timespec tim_neg = {0, -1};
  ret = LIBC_NAMESPACE::clock_nanosleep(CLOCK_REALTIME, 0, &tim_neg, nullptr);
  EXPECT_EQ(ret, EINVAL);
}

TEST_F(LlvmLibcClockNanosleep, InvalidFlag) {
  struct timespec tim = {0, 500};
  // Invalid flags (bits other than TIMER_ABSTIME) should fail with EINVAL.
  int ret = LIBC_NAMESPACE::clock_nanosleep(CLOCK_REALTIME, ~0, &tim, nullptr);
  EXPECT_EQ(ret, EINVAL);
}

TEST_F(LlvmLibcClockNanosleep, SleepRealtime) {
  struct timespec tim = {0, 500};
  int ret = LIBC_NAMESPACE::clock_nanosleep(CLOCK_REALTIME, 0, &tim, nullptr);
  EXPECT_EQ(ret, 0);
}

TEST_F(LlvmLibcClockNanosleep, SleepMonotonic) {
  struct timespec tim = {0, 500};
  int ret = LIBC_NAMESPACE::clock_nanosleep(CLOCK_MONOTONIC, 0, &tim, nullptr);
  EXPECT_EQ(ret, 0);
}

TEST_F(LlvmLibcClockNanosleep, SleepAbsTimePast) {
  // If TIMER_ABSTIME is specified and the absolute time has already passed,
  // clock_nanosleep should return immediately with 0.
  struct timespec tim = {0, 0};
  int ret = LIBC_NAMESPACE::clock_nanosleep(CLOCK_REALTIME, TIMER_ABSTIME, &tim,
                                            nullptr);
  EXPECT_EQ(ret, 0);
}
