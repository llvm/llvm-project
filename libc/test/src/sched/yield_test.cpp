//===-- Unittests for sched_yield -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/errno/libc_errno.h"
#include "src/sched/sched_yield.h"
#include "test/UnitTest/Test.h"

TEST(LlvmLibcSchedYieldTest, SmokeTest) {
  libc_errno = 0;
  // sched_yield() always succeeds, just do a basic test that errno/ret are
  // properly 0.
  ASSERT_EQ(LIBC_NAMESPACE::sched_yield(), 0);
  ASSERT_EQ(libc_errno, 0);
}
