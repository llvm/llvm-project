//===-- Unittests for clock_gettime ---------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/time/clock_gettime.h"
#include "test/UnitTest/Test.h"

#include <time.h>

TEST(LlvmLibcClockGetTime, RealTime) {
  struct timespec tp;
  int result;
  result = clock_gettime(CLOCK_REALTIME, &tp);
  ASSERT_EQ(result, 0);
  ASSERT_GT(tp.tv_sec, time_t(0));
}

#ifdef CLOCK_MONOTONIC
TEST(LlvmLibcClockGetTime, MonotonicTime) {
  struct timespec tp1, tp2;
  int result;
  result = clock_gettime(CLOCK_MONOTONIC, &tp1);
  ASSERT_EQ(result, 0);
  ASSERT_GT(tp1.tv_sec, time_t(0));
  result = clock_gettime(CLOCK_MONOTONIC, &tp2);
  ASSERT_EQ(result, 0);
  ASSERT_GE(tp2.tv_sec, tp1.tv_sec); // The monotonic clock should increase.
  if (tp2.tv_sec == tp1.tv_sec) {
    ASSERT_GE(tp2.tv_nsec, tp1.tv_nsec);
  }
}
#endif // CLOCK_MONOTONIC
