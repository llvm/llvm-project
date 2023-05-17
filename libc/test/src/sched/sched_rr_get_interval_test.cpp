//===-- Unittests for sched_rr_get_interval -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/errno/libc_errno.h"
#include "src/sched/sched_get_priority_min.h"
#include "src/sched/sched_getscheduler.h"
#include "src/sched/sched_rr_get_interval.h"
#include "src/sched/sched_setscheduler.h"
#include "src/unistd/getuid.h"
#include "test/UnitTest/Test.h"

#include <sched.h>

TEST(LlvmLibcSchedRRGetIntervalTest, SmokeTest) {
  libc_errno = 0;
  auto SetSched = [&](int policy) {
    int min_priority = __llvm_libc::sched_get_priority_min(policy);
    ASSERT_GE(min_priority, 0);
    ASSERT_EQ(libc_errno, 0);
    struct sched_param param;
    param.sched_priority = min_priority;
    ASSERT_EQ(__llvm_libc::sched_setscheduler(0, policy, &param), 0);
    ASSERT_EQ(libc_errno, 0);
  };

  auto TimespecToNs = [](struct timespec t) {
    return t.tv_sec * 1000UL * 1000UL * 1000UL + t.tv_nsec;
  };

  struct timespec ts;

  // We can only set SCHED_RR with CAP_SYS_ADMIN
  if (__llvm_libc::getuid() == 0)
    SetSched(SCHED_RR);

  int cur_policy = __llvm_libc::sched_getscheduler(0);
  ASSERT_GE(cur_policy, 0);
  ASSERT_EQ(libc_errno, 0);

  // We can actually run meaningful tests.
  if (cur_policy == SCHED_RR) {
    // Success
    ASSERT_EQ(__llvm_libc::sched_rr_get_interval(0, &ts), 0);
    ASSERT_EQ(libc_errno, 0);

    // Check that numbers make sense (liberal bound of 10ns - 30sec)
    ASSERT_GT(TimespecToNs(ts), 10UL);
    ASSERT_LT(TimespecToNs(ts), 30UL * 1000UL * 1000UL * 1000UL);

    // Null timespec
    ASSERT_EQ(__llvm_libc::sched_rr_get_interval(0, nullptr), -1);
    ASSERT_EQ(libc_errno, EFAULT);
    libc_errno = 0;

    // Negative pid
    ASSERT_EQ(__llvm_libc::sched_rr_get_interval(-1, &ts), -1);
    ASSERT_EQ(libc_errno, EINVAL);
    libc_errno = 0;
  }

  // Negative tests don't have SCHED_RR set
  SetSched(SCHED_OTHER);
  ASSERT_EQ(__llvm_libc::sched_rr_get_interval(0, &ts), 0);
  ASSERT_EQ(libc_errno, 0);
  libc_errno = 0;

  // TODO: Missing unkown pid -> ESRCH. This is read only so safe to try a few
  //       unlikely values.
}
