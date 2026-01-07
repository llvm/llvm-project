//===-- Unittests for clock_settime ---------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "hdr/time_macros.h"
#include "hdr/types/struct_timespec.h"
#include "src/time/clock_settime.h"
#include "test/UnitTest/ErrnoCheckingTest.h"
#include "test/UnitTest/Test.h"

using LlvmLibcClockSetTime = LIBC_NAMESPACE::testing::ErrnoCheckingTest;

#ifdef CLOCK_MONOTONIC
TEST_F(LlvmLibcClockSetTime, MonotonicIsNotSettable) {
  timespec ts = {0, 0};
  int result = LIBC_NAMESPACE::clock_settime(CLOCK_MONOTONIC, &ts);
  ASSERT_EQ(result, -1);
  ASSERT_ERRNO_EQ(EINVAL);
}
#endif // CLOCK_MONOTONIC

TEST_F(LlvmLibcClockSetTime, InvalidClockId) {
  timespec ts = {0, 0};
  int result = LIBC_NAMESPACE::clock_settime(static_cast<clockid_t>(-1), &ts);
  ASSERT_EQ(result, -1);
  ASSERT_ERRNO_EQ(EINVAL);
}

TEST_F(LlvmLibcClockSetTime, InvalidTimespecNsec) {
  timespec ts = {0, 1000000000L};
  int result = LIBC_NAMESPACE::clock_settime(CLOCK_REALTIME, &ts);
  ASSERT_EQ(result, -1);
  ASSERT_ERRNO_EQ(EINVAL);
}

TEST_F(LlvmLibcClockSetTime, NullPointerIsEFAULT) {
  int result = LIBC_NAMESPACE::clock_settime(CLOCK_REALTIME, nullptr);
  ASSERT_EQ(result, -1);
  ASSERT_ERRNO_EQ(EFAULT);
}

TEST_F(LlvmLibcClockSetTime, ClockIsSet) {
  timespec ts = {0, 0};
  int result = LIBC_NAMESPACE::clock_settime(CLOCK_REALTIME, &ts);
  if (result == 0) {
    ASSERT_ERRNO_SUCCESS();
  } else {
    ASSERT_ERRNO_EQ(EPERM);
  }
}
