//===-- Unittests for clock -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/time/clock.h"
#include "test/UnitTest/Test.h"

#include <limits.h>
#include <time.h>

TEST(LlvmLibcClockTest, SmokeTest) {
  clock_t c1 = __llvm_libc::clock();
  ASSERT_GT(c1, clock_t(0));

  clock_t c2 = __llvm_libc::clock();
  ASSERT_GE(c2, c1);
}
