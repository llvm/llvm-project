//===-- Unittests for __sched_cpu_count -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/OSUtil/syscall.h"
#include "src/errno/libc_errno.h"
#include "src/sched/sched_getaffinity.h"
#include "src/sched/sched_getcpucount.h"
#include "test/UnitTest/ErrnoSetterMatcher.h"

#include <sched.h>
#include <sys/syscall.h>

TEST(LlvmLibcSchedCpuCountTest, SmokeTest) {
  cpu_set_t mask;
  libc_errno = 0;
  using __llvm_libc::testing::ErrnoSetterMatcher::Succeeds;
  pid_t tid = __llvm_libc::syscall_impl(SYS_gettid);
  ASSERT_GT(tid, pid_t(0));
  ASSERT_THAT(__llvm_libc::sched_getaffinity(tid, sizeof(cpu_set_t), &mask),
              Succeeds(0));

  // CPU_COUNT is a macro, but it expands to an LLVM-libc internal function that
  // needs to be in the appropriate namespace for the test.
  int num_cpus = __llvm_libc::CPU_COUNT(&mask);
  ASSERT_GT(num_cpus, 0);
  ASSERT_LE(num_cpus, int(sizeof(cpu_set_t) * sizeof(unsigned long)));
}
