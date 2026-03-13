//===-- Unittests for __sched_cpu_count -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/OSUtil/syscall.h"
#include "src/sched/sched_getaffinity.h"
#include "src/sched/sched_getcpucount.h"
#include "test/UnitTest/ErrnoCheckingTest.h"
#include "test/UnitTest/ErrnoSetterMatcher.h"

#include "hdr/sched_macros.h"
#include "hdr/types/cpu_set_t.h"
#include "hdr/types/pid_t.h"

using LlvmLibcSchedCpuCountTest = LIBC_NAMESPACE::testing::ErrnoCheckingTest;

TEST_F(LlvmLibcSchedCpuCountTest, SmokeTest) {
  cpu_set_t mask;
  using LIBC_NAMESPACE::testing::ErrnoSetterMatcher::Succeeds;
  pid_t tid = LIBC_NAMESPACE::syscall_impl<pid_t>(SYS_gettid);
  ASSERT_GT(tid, pid_t(0));
  ASSERT_THAT(LIBC_NAMESPACE::sched_getaffinity(tid, sizeof(cpu_set_t), &mask),
              Succeeds(0));

  // CPU_COUNT is a macro, but it expands to an LLVM-libc internal function that
  // needs to be in the appropriate namespace for the test.
  int num_cpus = LIBC_NAMESPACE::CPU_COUNT(&mask);
  ASSERT_GT(num_cpus, 0);
  ASSERT_LE(num_cpus, int(sizeof(cpu_set_t) * sizeof(unsigned long)));
}
