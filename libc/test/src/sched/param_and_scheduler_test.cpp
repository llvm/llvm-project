//===-- Unittests for sched_{set,get}{scheduler,param} --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/libc_errno.h"
#include "src/sched/sched_get_priority_max.h"
#include "src/sched/sched_get_priority_min.h"
#include "src/sched/sched_getparam.h"
#include "src/sched/sched_getscheduler.h"
#include "src/sched/sched_setparam.h"
#include "src/sched/sched_setscheduler.h"
#include "src/unistd/getuid.h"
#include "test/UnitTest/Test.h"

#include <sched.h>

// We Test:
// SCHED_OTHER, SCHED_FIFO, SCHED_RR
//
// TODO: Missing two tests.
//       1) Missing permissions -> EPERM. Maybe doable by finding
//          another pid that exists and changing its policy, but that
//          seems risky. Maybe something with fork/clone would work.
//
//       2) Unkown pid -> ESRCH. Probably safe to choose a large range
//          number or scanning current pids and getting one that doesn't
//          exist, but again seems like it may risk actually changing
//          sched policy on a running task.
//
//       Linux specific test could also include:
//          SCHED_ISO, SCHED_DEADLINE

class SchedTest : public LIBC_NAMESPACE::testing::Test {
public:
  void testSched(int policy, bool is_mandatory) {
    libc_errno = 0;

    int init_policy = LIBC_NAMESPACE::sched_getscheduler(0);
    ASSERT_GE(init_policy, 0);
    ASSERT_ERRNO_SUCCESS();

    int max_priority = LIBC_NAMESPACE::sched_get_priority_max(policy);
    ASSERT_GE(max_priority, 0);
    ASSERT_ERRNO_SUCCESS();
    int min_priority = LIBC_NAMESPACE::sched_get_priority_min(policy);
    ASSERT_GE(min_priority, 0);
    ASSERT_ERRNO_SUCCESS();

    struct sched_param param = {min_priority};

    // Negative pid
    ASSERT_EQ(LIBC_NAMESPACE::sched_setscheduler(-1, policy, &param), -1);
    ASSERT_ERRNO_EQ(EINVAL);
    libc_errno = 0;

    ASSERT_EQ(LIBC_NAMESPACE::sched_getscheduler(-1), -1);
    ASSERT_ERRNO_EQ(EINVAL);
    libc_errno = 0;

    // Invalid Policy
    ASSERT_EQ(LIBC_NAMESPACE::sched_setscheduler(0, policy | 128, &param), -1);
    ASSERT_ERRNO_EQ(EINVAL);
    libc_errno = 0;

    // Out of bounds priority
    param.sched_priority = min_priority - 1;
    ASSERT_EQ(LIBC_NAMESPACE::sched_setscheduler(0, policy, &param), -1);
    ASSERT_ERRNO_EQ(EINVAL);
    libc_errno = 0;

    param.sched_priority = max_priority + 1;
    ASSERT_EQ(LIBC_NAMESPACE::sched_setscheduler(0, policy, &param), -1);
    // A bit hard to test as depending on user privileges we can run into
    // different issues.
    ASSERT_TRUE(libc_errno == EINVAL || libc_errno == EPERM);
    libc_errno = 0;

    param.sched_priority = min_priority;
    // Success/unsupported policy/missing permissions.
    int setscheduler_result =
        LIBC_NAMESPACE::sched_setscheduler(0, policy, &param);
    ASSERT_TRUE(setscheduler_result == 0 || setscheduler_result == -1);
    ASSERT_TRUE(
        setscheduler_result != -1
            ? (libc_errno == 0)
            : ((!is_mandatory && libc_errno == EINVAL) || libc_errno == EPERM));
    libc_errno = 0;

    ASSERT_EQ(LIBC_NAMESPACE::sched_getscheduler(0),
              setscheduler_result != -1 ? policy : init_policy);
    ASSERT_ERRNO_SUCCESS();

    // Out of bounds priority
    param.sched_priority = -1;
    ASSERT_EQ(LIBC_NAMESPACE::sched_setparam(0, &param), -1);
    ASSERT_ERRNO_EQ(EINVAL);
    libc_errno = 0;

    param.sched_priority = max_priority + 1;
    ASSERT_EQ(LIBC_NAMESPACE::sched_setparam(0, &param), -1);
    ASSERT_ERRNO_EQ(EINVAL);
    libc_errno = 0;

    for (int priority = min_priority; priority <= max_priority; ++priority) {
      ASSERT_EQ(LIBC_NAMESPACE::sched_getparam(0, &param), 0);
      ASSERT_ERRNO_SUCCESS();
      int init_priority = param.sched_priority;

      param.sched_priority = priority;

      // Negative pid
      ASSERT_EQ(LIBC_NAMESPACE::sched_setparam(-1, &param), -1);
      ASSERT_ERRNO_EQ(EINVAL);
      libc_errno = 0;

      ASSERT_EQ(LIBC_NAMESPACE::sched_getparam(-1, &param), -1);
      ASSERT_ERRNO_EQ(EINVAL);
      libc_errno = 0;

      // Success/unsupported policy/missing permissions
      int setparam_result = LIBC_NAMESPACE::sched_setparam(0, &param);
      ASSERT_TRUE(setparam_result == 0 || setparam_result == -1);
      ASSERT_TRUE(setparam_result != -1
                      ? (libc_errno == 0)
                      : ((setscheduler_result == -1 && libc_errno == EINVAL) ||
                         libc_errno == EPERM));
      libc_errno = 0;

      ASSERT_EQ(LIBC_NAMESPACE::sched_getparam(0, &param), 0);
      ASSERT_ERRNO_SUCCESS();

      ASSERT_EQ(param.sched_priority,
                setparam_result != -1 ? priority : init_priority);
    }

    // Null test
    ASSERT_EQ(LIBC_NAMESPACE::sched_setscheduler(0, policy, nullptr), -1);
    ASSERT_ERRNO_EQ(EINVAL);
    libc_errno = 0;
  }
};

#define LIST_SCHED_TESTS(policy, can_set)                                      \
  using LlvmLibcSchedTest = SchedTest;                                         \
  TEST_F(LlvmLibcSchedTest, Sched_##policy) { testSched(policy, can_set); }

// Mandated by POSIX.
LIST_SCHED_TESTS(SCHED_OTHER, true)
LIST_SCHED_TESTS(SCHED_FIFO, true)
LIST_SCHED_TESTS(SCHED_RR, true)

// Linux extensions.
LIST_SCHED_TESTS(SCHED_BATCH, true)
LIST_SCHED_TESTS(SCHED_IDLE, true)

TEST(LlvmLibcSchedParamAndSchedulerTest, NullParamTest) {
  libc_errno = 0;

  ASSERT_EQ(LIBC_NAMESPACE::sched_setparam(0, nullptr), -1);
  ASSERT_ERRNO_EQ(EINVAL);
  libc_errno = 0;

  ASSERT_EQ(LIBC_NAMESPACE::sched_getparam(0, nullptr), -1);
  ASSERT_ERRNO_EQ(EINVAL);
  libc_errno = 0;
}
