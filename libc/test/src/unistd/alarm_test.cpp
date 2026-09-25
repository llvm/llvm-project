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
#include "hdr/types/struct_timespec.h"
#include "src/signal/signal.h"
#include "src/time/nanosleep.h"
#include "src/unistd/alarm.h"
#include "test/UnitTest/Test.h"

static volatile bool alarm_fired = false;
extern "C" void alarm_handler(int) { alarm_fired = true; }

TEST(LlvmLibcAlarmTest, Basic) {
  LIBC_NAMESPACE::signal(SIGALRM, alarm_handler);
  alarm_fired = false;

  // Set alarm for 10 seconds.
  unsigned int prev = LIBC_NAMESPACE::alarm(10);
  // The first call to alarm should return 0 as no alarm has been set before.
  EXPECT_EQ(prev, 0U);

  // Set alarm for 5 seconds. It should return the remaining time of the
  // previous 10-second alarm. Since very little time has passed, it should
  // be close to 10.
  prev = LIBC_NAMESPACE::alarm(5);
  EXPECT_GT(prev, 0U);
  EXPECT_LE(prev, 10U);

  // Cancel the alarm by setting it to 0. It should return the remaining
  // time of the 5-second alarm.
  prev = LIBC_NAMESPACE::alarm(0);
  EXPECT_GT(prev, 0U);
  EXPECT_LE(prev, 5U);

  // Ensure it didn't fire since we canceled it immediately.
  EXPECT_FALSE(alarm_fired);
}

// This test actually waits for the alarm to fire, which takes at least 1s.
TEST(LlvmLibcAlarmTest, FiringTest) {
  LIBC_NAMESPACE::signal(SIGALRM, alarm_handler);
  alarm_fired = false;

  // Set alarm for 1 second.
  LIBC_NAMESPACE::alarm(1);

  // Wait for the alarm to fire. nanosleep will return early when the
  // SIGALRM is delivered.
  struct timespec tv = {5, 0};
  LIBC_NAMESPACE::nanosleep(&tv, nullptr);

  EXPECT_TRUE(alarm_fired);
}
