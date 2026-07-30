//===-- Unittests for clock_gettime ---------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "hdr/time_macros.h"
#include "hdr/types/struct_timespec.h"
#include "hdr/types/time_t.h"
#include "src/__support/macros/properties/architectures.h"
#include "src/time/clock_gettime.h"
#include "test/UnitTest/Test.h"

TEST(LlvmLibcClockGetTime, RealTime) {
  timespec tp;
  int result;
  result = LIBC_NAMESPACE::clock_gettime(CLOCK_REALTIME, &tp);
  // The GPU does not implement CLOCK_REALTIME but provides it so programs will
  // compile.
#ifdef LIBC_TARGET_ARCH_IS_GPU
  ASSERT_EQ(result, -1);
#else
  ASSERT_EQ(result, 0);
  ASSERT_GT(tp.tv_sec, time_t(0));
#endif
}

#ifdef CLOCK_MONOTONIC
TEST(LlvmLibcClockGetTime, MonotonicTime) {
  timespec tp1, tp2;
  int result;
  result = LIBC_NAMESPACE::clock_gettime(CLOCK_MONOTONIC, &tp1);
  ASSERT_EQ(result, 0);
  ASSERT_GT(tp1.tv_sec, time_t(0));
  result = LIBC_NAMESPACE::clock_gettime(CLOCK_MONOTONIC, &tp2);
  ASSERT_EQ(result, 0);
  ASSERT_GE(tp2.tv_sec, tp1.tv_sec); // The monotonic clock should increase.
  if (tp2.tv_sec == tp1.tv_sec) {
    ASSERT_GE(tp2.tv_nsec, tp1.tv_nsec);
  }
}
#endif // CLOCK_MONOTONIC
