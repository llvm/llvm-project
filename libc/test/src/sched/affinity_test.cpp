//===-- Unittests for sched_getaffinity and sched_setaffinity -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/OSUtil/syscall.h"
#include "src/errno/libc_errno.h"
#include "src/sched/sched_getaffinity.h"
#include "src/sched/sched_setaffinity.h"
#include "test/UnitTest/ErrnoSetterMatcher.h"

#include <sched.h>
#include <sys/syscall.h>

TEST(LlvmLibcSchedAffinityTest, SmokeTest) {
  cpu_set_t mask;
  LIBC_NAMESPACE::libc_errno = 0;
  using LIBC_NAMESPACE::testing::ErrnoSetterMatcher::Succeeds;
  pid_t tid = LIBC_NAMESPACE::syscall_impl<pid_t>(SYS_gettid);
  ASSERT_GT(tid, pid_t(0));
  // We just get and set the same mask.
  ASSERT_THAT(LIBC_NAMESPACE::sched_getaffinity(tid, sizeof(cpu_set_t), &mask),
              Succeeds(0));
  ASSERT_THAT(LIBC_NAMESPACE::sched_setaffinity(tid, sizeof(cpu_set_t), &mask),
              Succeeds(0));
}

TEST(LlvmLibcSchedAffinityTest, BadMask) {
  using LIBC_NAMESPACE::testing::ErrnoSetterMatcher::Fails;
  pid_t tid = LIBC_NAMESPACE::syscall_impl<pid_t>(SYS_gettid);

  LIBC_NAMESPACE::libc_errno = 0;
  ASSERT_THAT(
      LIBC_NAMESPACE::sched_getaffinity(tid, sizeof(cpu_set_t), nullptr),
      Fails(EFAULT));

  LIBC_NAMESPACE::libc_errno = 0;
  ASSERT_THAT(
      LIBC_NAMESPACE::sched_setaffinity(tid, sizeof(cpu_set_t), nullptr),
      Fails(EFAULT));

  LIBC_NAMESPACE::libc_errno = 0;
}
