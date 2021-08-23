//===-- flang/unittests/RuntimeGTest/Time.cpp -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "gtest/gtest.h"
#include "../../runtime/time-intrinsic.h"

using namespace Fortran::runtime;

TEST(TimeIntrinsics, CpuTime) {
  // We can't really test that we get the "right" result for CPU_TIME, but we
  // can have a smoke test to see that we get something reasonable on the
  // platforms where we expect to support it.
  double start{RTNAME(CpuTime)()};
  ASSERT_GE(start, 0.0);

  // Loop until we get a different value from CpuTime. If we don't get one
  // before we time out, then we should probably look into an implementation
  // for CpuTime with a better timer resolution.
  for (double end = start; end == start; end = RTNAME(CpuTime)()) {
    ASSERT_GE(end, 0.0);
    ASSERT_GE(end, start);
  }
}

using count_t = CppTypeFor<TypeCategory::Integer, 8>;

TEST(TimeIntrinsics, SystemClock) {
  // We can't really test that we get the "right" result for SYSTEM_CLOCK, but
  // we can have a smoke test to see that we get something reasonable on the
  // platforms where we expect to support it.

  // The value of the count rate and max will vary by platform, but they should
  // always be strictly positive if we have a working implementation of
  // SYSTEM_CLOCK.
  EXPECT_GT(RTNAME(SystemClockCountRate)(), 0);

  count_t max{RTNAME(SystemClockCountMax)()};
  EXPECT_GT(max, 0);

  count_t start{RTNAME(SystemClockCount)()};
  EXPECT_GE(start, 0);
  EXPECT_LE(start, max);

  // Loop until we get a different value from SystemClockCount. If we don't get
  // one before we time out, then we should probably look into an implementation
  // for SystemClokcCount with a better timer resolution on this platform.
  for (count_t end = start; end == start; end = RTNAME(SystemClockCount)()) {
    EXPECT_GE(end, 0);
    EXPECT_LE(end, max);

    EXPECT_GE(end, start);
  }
}
