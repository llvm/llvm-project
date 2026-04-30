//===-- Unittests for getcpu ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/OSUtil/syscall.h"
#include "src/sched/sched_getcpu.h"
#include "test/UnitTest/ErrnoCheckingTest.h"
#include "test/UnitTest/ErrnoSetterMatcher.h"

using LlvmLibcSchedSchedGetCpuTest = LIBC_NAMESPACE::testing::ErrnoCheckingTest;

TEST_F(LlvmLibcSchedSchedGetCpuTest, SmokeTest) {
  using LIBC_NAMESPACE::testing::ErrnoSetterMatcher::Succeeds;
  ASSERT_THAT(LIBC_NAMESPACE::sched_getcpu(), Succeeds());
}
