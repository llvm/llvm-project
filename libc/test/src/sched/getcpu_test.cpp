//===-- Unittests for getcpu ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/OSUtil/syscall.h"
#include "src/sched/getcpu.h"
#include "test/UnitTest/ErrnoCheckingTest.h"
#include "test/UnitTest/ErrnoSetterMatcher.h"

using LlvmLibcSchedGetCpuTest = LIBC_NAMESPACE::testing::ErrnoCheckingTest;

TEST_F(LlvmLibcSchedGetCpuTest, SmokeTest) {
  unsigned int current_cpu;
  unsigned int current_node;
  using LIBC_NAMESPACE::testing::ErrnoSetterMatcher::Succeeds;
  ASSERT_THAT(LIBC_NAMESPACE::getcpu(&current_cpu, &current_node), Succeeds(0));
}

TEST_F(LlvmLibcSchedGetCpuTest, BadPointer) {
  unsigned int current_cpu;
  unsigned int *current_node = reinterpret_cast<unsigned int *>(-1);
  using LIBC_NAMESPACE::testing::ErrnoSetterMatcher::Fails;
  ASSERT_THAT(LIBC_NAMESPACE::getcpu(&current_cpu, current_node),
              Fails(EFAULT));
}
