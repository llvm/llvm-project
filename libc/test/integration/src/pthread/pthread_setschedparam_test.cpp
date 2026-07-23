//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Integration tests for pthread_setschedparam and pthread_getschedparam.
///
//===----------------------------------------------------------------------===//

#include "hdr/errno_macros.h"
#include "hdr/sched_macros.h"
#include "hdr/types/pthread_t.h"
#include "hdr/types/struct_sched_param.h"
#include "src/pthread/pthread_create.h"
#include "src/pthread/pthread_getschedparam.h"
#include "src/pthread/pthread_join.h"
#include "src/pthread/pthread_self.h"
#include "src/pthread/pthread_setschedparam.h"
#include "test/IntegrationTest/test.h"

static void *child_func(void *) { return nullptr; }

TEST_MAIN() {
  auto main_thread = LIBC_NAMESPACE::pthread_self();
  struct sched_param param;
  int policy;

  // 1. Test getschedparam on self
  ASSERT_EQ(LIBC_NAMESPACE::pthread_getschedparam(main_thread, &policy, &param),
            0);

  // 2. Test setschedparam on self (Success)
  param.sched_priority = 0;
  ASSERT_EQ(
      LIBC_NAMESPACE::pthread_setschedparam(main_thread, SCHED_OTHER, &param),
      0);

  // Verify it was set
  int new_policy;
  struct sched_param new_param;
  ASSERT_EQ(LIBC_NAMESPACE::pthread_getschedparam(main_thread, &new_policy,
                                                  &new_param),
            0);
  ASSERT_EQ(new_policy, SCHED_OTHER);
  ASSERT_EQ(new_param.sched_priority, 0);

  // 3. Test setschedparam on self (Failure - Invalid Policy)
  ASSERT_EQ(LIBC_NAMESPACE::pthread_setschedparam(main_thread, -1, &param),
            EINVAL);

  // 4. Test setschedparam on self (Failure - Invalid Priority for SCHED_OTHER)
  param.sched_priority = 1; // Invalid for SCHED_OTHER
  ASSERT_EQ(
      LIBC_NAMESPACE::pthread_setschedparam(main_thread, SCHED_OTHER, &param),
      EINVAL);
  param.sched_priority = 0; // Reset

  // 5. Test on Child Thread
  pthread_t th;
  ASSERT_EQ(LIBC_NAMESPACE::pthread_create(&th, nullptr, child_func, nullptr),
            0);

  // Get child's default sched param
  ASSERT_EQ(LIBC_NAMESPACE::pthread_getschedparam(th, &policy, &param), 0);

  // Set child's sched param (Success)
  param.sched_priority = 0;
  ASSERT_EQ(LIBC_NAMESPACE::pthread_setschedparam(th, SCHED_OTHER, &param), 0);

  // Verify child's sched param
  ASSERT_EQ(LIBC_NAMESPACE::pthread_getschedparam(th, &new_policy, &new_param),
            0);
  ASSERT_EQ(new_policy, SCHED_OTHER);
  ASSERT_EQ(new_param.sched_priority, 0);

  // Set child's sched param (Failure - Invalid Policy)
  ASSERT_EQ(LIBC_NAMESPACE::pthread_setschedparam(th, -1, &param), EINVAL);

  // Set child's sched param (Failure - Invalid Priority)
  param.sched_priority = 1;
  ASSERT_EQ(LIBC_NAMESPACE::pthread_setschedparam(th, SCHED_OTHER, &param),
            EINVAL);

  void *retval;
  ASSERT_EQ(LIBC_NAMESPACE::pthread_join(th, &retval), 0);

  return 0;
}
