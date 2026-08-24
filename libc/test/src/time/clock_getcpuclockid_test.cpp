//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unittests for clock_getcpuclockid.
///
//===----------------------------------------------------------------------===//

#include "hdr/errno_macros.h"
#include "hdr/types/clockid_t.h"
#include "hdr/types/pid_t.h"
#include "hdr/types/struct_timespec.h"
#include "src/time/clock_getcpuclockid.h"
#include "src/time/clock_gettime.h"
#include "src/unistd/getpid.h"
#include "test/UnitTest/ErrnoCheckingTest.h"
#include "test/UnitTest/Test.h"

using LlvmLibcClockGetCpuClockIdTest =
    LIBC_NAMESPACE::testing::ErrnoCheckingTest;

TEST_F(LlvmLibcClockGetCpuClockIdTest, CurrentProcessZeroPid) {
  clockid_t clock_id = 0;
  ASSERT_EQ(LIBC_NAMESPACE::clock_getcpuclockid(0, &clock_id), 0);
  // On Linux, pid 0 translates to (~0 << 3) | 2 == -6.
  EXPECT_EQ(clock_id, static_cast<clockid_t>(-6));

  struct timespec ts;
  EXPECT_EQ(LIBC_NAMESPACE::clock_gettime(clock_id, &ts), 0);
  EXPECT_GE(ts.tv_sec, static_cast<time_t>(0));
}

TEST_F(LlvmLibcClockGetCpuClockIdTest, CurrentProcessExplicitPid) {
  pid_t pid = LIBC_NAMESPACE::getpid();
  clockid_t clock_id = 0;
  ASSERT_EQ(LIBC_NAMESPACE::clock_getcpuclockid(pid, &clock_id), 0);
  EXPECT_EQ(clock_id,
            static_cast<clockid_t>((static_cast<unsigned int>(~pid) << 3) | 2));

  struct timespec ts;
  EXPECT_EQ(LIBC_NAMESPACE::clock_gettime(clock_id, &ts), 0);
  EXPECT_GE(ts.tv_sec, static_cast<time_t>(0));
}

TEST_F(LlvmLibcClockGetCpuClockIdTest, InvalidPid) {
  clockid_t clock_id = 0;
  // Invalid PIDs should fail with ESRCH.
  EXPECT_EQ(LIBC_NAMESPACE::clock_getcpuclockid(-1, &clock_id), ESRCH);
  EXPECT_EQ(LIBC_NAMESPACE::clock_getcpuclockid(1000000000, &clock_id), ESRCH);
}
