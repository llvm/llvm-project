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
  double start = RTNAME(CpuTime)();
  ASSERT_GE(start, 0.0);

  // Loop until we get a different value from CpuTime. If we don't get one
  // before we time out, then we should probably look into an implementation
  // for CpuTime with a better timer resolution.
  for (double end = start; end == start; end = RTNAME(CpuTime)()) {
    ASSERT_GT(end, 0.0);
    ASSERT_GE(end, start);
  }
}
