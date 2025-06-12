//===-- Unittests for sched_get_priority_{min,max} ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/libc_errno.h"
#include "src/sched/sched_get_priority_max.h"
#include "src/sched/sched_get_priority_min.h"
#include "test/UnitTest/Test.h"

#include <sched.h>

TEST(LlvmLibcSchedGetPriorityTest, HandleBadPolicyTest) {

  // Test arbitrary values for which there is no policy.
  {
    int policy = -1;
    int max_priority = LIBC_NAMESPACE::sched_get_priority_max(policy);
    ASSERT_EQ(max_priority, -1);
    ASSERT_ERRNO_EQ(EINVAL);
    int min_priority = LIBC_NAMESPACE::sched_get_priority_min(policy);
    ASSERT_EQ(min_priority, -1);
    ASSERT_ERRNO_EQ(EINVAL);
  }

  {
    int policy = 30;
    int max_priority = LIBC_NAMESPACE::sched_get_priority_max(policy);
    ASSERT_EQ(max_priority, -1);
    ASSERT_ERRNO_EQ(EINVAL);
    int min_priority = LIBC_NAMESPACE::sched_get_priority_min(policy);
    ASSERT_EQ(min_priority, -1);
    ASSERT_ERRNO_EQ(EINVAL);
  }

  {
    int policy = 80;
    int max_priority = LIBC_NAMESPACE::sched_get_priority_max(policy);
    ASSERT_EQ(max_priority, -1);
    ASSERT_ERRNO_EQ(EINVAL);
    int min_priority = LIBC_NAMESPACE::sched_get_priority_min(policy);
    ASSERT_EQ(min_priority, -1);
    ASSERT_ERRNO_EQ(EINVAL);
  }

  {
    int policy = 110;
    int max_priority = LIBC_NAMESPACE::sched_get_priority_max(policy);
    ASSERT_EQ(max_priority, -1);
    ASSERT_ERRNO_EQ(EINVAL);
    int min_priority = LIBC_NAMESPACE::sched_get_priority_min(policy);
    ASSERT_EQ(min_priority, -1);
    ASSERT_ERRNO_EQ(EINVAL);
  }
}

TEST(LlvmLibcSchedGetPriorityTest, SmokeTest) {
  libc_errno = 0;

  // We Test:
  // SCHED_OTHER, SCHED_FIFO, SCHED_RR
  // Linux specific test could also include:
  // SCHED_BATCH, SCHED_ISO, SCHED_IDLE, SCHED_DEADLINE
  {
    int policy = SCHED_OTHER;
    int max_priority = LIBC_NAMESPACE::sched_get_priority_max(policy);
    ASSERT_GE(max_priority, 0);
    ASSERT_ERRNO_SUCCESS();
    int min_priority = LIBC_NAMESPACE::sched_get_priority_min(policy);
    ASSERT_GE(min_priority, 0);
    ASSERT_ERRNO_SUCCESS();

    ASSERT_LE(max_priority, 99);
    ASSERT_GE(min_priority, 0);
    ASSERT_GE(max_priority, min_priority);
  }

  {
    int policy = SCHED_FIFO;
    int max_priority = LIBC_NAMESPACE::sched_get_priority_max(policy);
    ASSERT_GE(max_priority, 0);
    ASSERT_ERRNO_SUCCESS();
    int min_priority = LIBC_NAMESPACE::sched_get_priority_min(policy);
    ASSERT_GE(min_priority, 0);
    ASSERT_ERRNO_SUCCESS();

    ASSERT_LE(max_priority, 99);
    ASSERT_GE(min_priority, 0);
    ASSERT_GT(max_priority, min_priority);
  }

  {
    int policy = SCHED_RR;
    int max_priority = LIBC_NAMESPACE::sched_get_priority_max(policy);
    ASSERT_GE(max_priority, 0);
    ASSERT_ERRNO_SUCCESS();
    int min_priority = LIBC_NAMESPACE::sched_get_priority_min(policy);
    ASSERT_GE(min_priority, 0);
    ASSERT_ERRNO_SUCCESS();

    ASSERT_LE(max_priority, 99);
    ASSERT_GE(min_priority, 0);
    ASSERT_GT(max_priority, min_priority);
  }
}
