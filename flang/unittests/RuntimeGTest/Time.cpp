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

volatile int x = 0;

void LookBusy() {
  // We're trying to track actual processor time, so sleeping is not an option.
  // Doing some writes to a volatile variable should do the trick.
  for (int i = 0; i < (1 << 8); ++i) {
    x = i;
  }
}

TEST(TimeIntrinsics, CpuTime) {
  // We can't really test that we get the "right" result for CPU_TIME, but we
  // can have a smoke test to see that we get something reasonable on the
  // platforms where we expect to support it.
  double start = RTNAME(CpuTime)();
  LookBusy();
  double end = RTNAME(CpuTime)();

  ASSERT_GE(start, 0.0);
  ASSERT_GT(end, 0.0);
  ASSERT_GT(end, start);
}
