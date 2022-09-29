//===-- Unittests for sched_getaffinity and sched_setaffinity -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/OSUtil/syscall.h"
#include "src/sched/sched_getaffinity.h"
#include "src/sched/sched_setaffinity.h"
#include "test/ErrnoSetterMatcher.h"

#include <errno.h>
#include <sched.h>
#include <sys/syscall.h>

TEST(LlvmLibcSchedAffinityTest, SmokeTest) {
  cpu_set_t mask;
  errno = 0;
  using __llvm_libc::testing::ErrnoSetterMatcher::Succeeds;
  pid_t tid = __llvm_libc::syscall(SYS_gettid);
  ASSERT_GT(tid, pid_t(0));
  // We just get and set the same mask.
  ASSERT_THAT(__llvm_libc::sched_getaffinity(tid, sizeof(cpu_set_t), &mask),
              Succeeds(0));
  ASSERT_THAT(__llvm_libc::sched_setaffinity(tid, sizeof(cpu_set_t), &mask),
              Succeeds(0));
}

TEST(LlvmLibcSchedAffinityTest, BadMask) {
  using __llvm_libc::testing::ErrnoSetterMatcher::Fails;
  pid_t tid = __llvm_libc::syscall(SYS_gettid);

  errno = 0;
  ASSERT_THAT(__llvm_libc::sched_getaffinity(tid, sizeof(cpu_set_t), nullptr),
              Fails(EFAULT));

  errno = 0;
  ASSERT_THAT(__llvm_libc::sched_setaffinity(tid, sizeof(cpu_set_t), nullptr),
              Fails(EFAULT));

  errno = 0;
}
