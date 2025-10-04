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
