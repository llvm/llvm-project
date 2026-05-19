//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unittests for alarm.
///
//===----------------------------------------------------------------------===//

#include "hdr/signal_macros.h"
#include "src/signal/signal.h"
#include "src/unistd/alarm.h"
#include "test/UnitTest/Test.h"

static volatile int alarm_fired = 0;
extern "C" void alarm_handler(int) { alarm_fired = 1; }

TEST(LlvmLibcAlarmTest, Basic) {
  LIBC_NAMESPACE::signal(SIGALRM, alarm_handler);
  alarm_fired = 0;

  // Set alarm for 10 seconds.
  unsigned int prev = LIBC_NAMESPACE::alarm(10);
  // The first call to alarm should return 0 as no alarm has been set before.
  EXPECT_EQ(prev, 0U);

  // Set alarm for 5 seconds. It should return the remaining time of the
  // previous 10-second alarm. Since very little time has passed, it should
  // be 10 (or 9 if we are extremely slow).
  prev = LIBC_NAMESPACE::alarm(5);
  EXPECT_GE(prev, 9U);
  EXPECT_LE(prev, 10U);

  // Cancel the alarm by setting it to 0. It should return the remaining
  // time of the 5-second alarm.
  prev = LIBC_NAMESPACE::alarm(0);
  EXPECT_GE(prev, 4U);
  EXPECT_LE(prev, 5U);

  // Ensure it didn't fire since we canceled it immediately.
  EXPECT_FALSE(alarm_fired);
}

// This test actually waits for the alarm to fire, which takes at least 1s.
TEST(LlvmLibcAlarmTest, FiringTest) {
  LIBC_NAMESPACE::signal(SIGALRM, alarm_handler);
  alarm_fired = 0;

  // Set alarm for 1 second.
  LIBC_NAMESPACE::alarm(1);

  // Busy wait until it fires or we timeout (e.g. after ~2 seconds).
  // 1 billion iterations in the inner loop is roughly 1-2 seconds on modern
  // CPUs.
  for (int i = 0; i < 20 && !alarm_fired; ++i) {
    for (volatile int j = 0; j < 100000000; ++j)
      ;
  }

  EXPECT_TRUE(alarm_fired);
}
