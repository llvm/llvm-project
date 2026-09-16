//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unittests for pthread_attr_t.
///
//===----------------------------------------------------------------------===//

#include "hdr/errno_macros.h"
#include "hdr/pthread_macros.h"
#include "hdr/sched_macros.h"
#include "hdr/types/struct_sched_param.h"
#include "src/pthread/pthread_attr_destroy.h"
#include "src/pthread/pthread_attr_getdetachstate.h"
#include "src/pthread/pthread_attr_getguardsize.h"
#include "src/pthread/pthread_attr_getschedparam.h"
#include "src/pthread/pthread_attr_getschedpolicy.h"
#include "src/pthread/pthread_attr_getscope.h"
#include "src/pthread/pthread_attr_getstack.h"
#include "src/pthread/pthread_attr_getstacksize.h"
#include "src/pthread/pthread_attr_init.h"
#include "src/pthread/pthread_attr_setdetachstate.h"
#include "src/pthread/pthread_attr_setguardsize.h"
#include "src/pthread/pthread_attr_setschedparam.h"
#include "src/pthread/pthread_attr_setschedpolicy.h"
#include "src/pthread/pthread_attr_setscope.h"
#include "src/pthread/pthread_attr_setstack.h"
#include "src/pthread/pthread_attr_setstacksize.h"

#include "test/UnitTest/Test.h"

#include <linux/param.h> // For EXEC_PAGESIZE.
#include <pthread.h>

TEST(LlvmLibcPThreadAttrTest, InitAndDestroy) {
  pthread_attr_t attr;
  ASSERT_EQ(LIBC_NAMESPACE::pthread_attr_init(&attr), 0);

  int detachstate;
  ASSERT_EQ(LIBC_NAMESPACE::pthread_attr_getdetachstate(&attr, &detachstate),
            0);
  ASSERT_EQ(detachstate, int(PTHREAD_CREATE_JOINABLE));

  size_t guardsize;
  ASSERT_EQ(LIBC_NAMESPACE::pthread_attr_getguardsize(&attr, &guardsize), 0);
  ASSERT_EQ(guardsize, size_t(EXEC_PAGESIZE));

  ASSERT_EQ(LIBC_NAMESPACE::pthread_attr_destroy(&attr), 0);
}

TEST(LlvmLibcPThreadattrTest, SetAndGetDetachState) {
  pthread_attr_t attr;
  ASSERT_EQ(LIBC_NAMESPACE::pthread_attr_init(&attr), 0);

  int detachstate;
  LIBC_NAMESPACE::pthread_attr_getdetachstate(&attr, &detachstate);
  ASSERT_EQ(detachstate, int(PTHREAD_CREATE_JOINABLE));
  ASSERT_EQ(LIBC_NAMESPACE::pthread_attr_setdetachstate(
                &attr, PTHREAD_CREATE_DETACHED),
            0);
  ASSERT_EQ(LIBC_NAMESPACE::pthread_attr_getdetachstate(&attr, &detachstate),
            0);
  ASSERT_EQ(detachstate, int(PTHREAD_CREATE_DETACHED));

  ASSERT_EQ(LIBC_NAMESPACE::pthread_attr_setdetachstate(&attr, 0xBAD), EINVAL);

  ASSERT_EQ(LIBC_NAMESPACE::pthread_attr_destroy(&attr), 0);
}

TEST(LlvmLibcPThreadattrTest, SetAndGetGuardSize) {
  pthread_attr_t attr;
  ASSERT_EQ(LIBC_NAMESPACE::pthread_attr_init(&attr), 0);

  size_t guardsize;
  ASSERT_EQ(LIBC_NAMESPACE::pthread_attr_getguardsize(&attr, &guardsize), 0);
  ASSERT_EQ(guardsize, size_t(EXEC_PAGESIZE));
  ASSERT_EQ(LIBC_NAMESPACE::pthread_attr_setguardsize(&attr, 2 * EXEC_PAGESIZE),
            0);
  ASSERT_EQ(LIBC_NAMESPACE::pthread_attr_getguardsize(&attr, &guardsize), 0);
  ASSERT_EQ(guardsize, size_t(2 * EXEC_PAGESIZE));

  ASSERT_EQ(LIBC_NAMESPACE::pthread_attr_setguardsize(&attr, EXEC_PAGESIZE / 2),
            EINVAL);

  ASSERT_EQ(LIBC_NAMESPACE::pthread_attr_destroy(&attr), 0);
}

TEST(LlvmLibcPThreadattrTest, SetAndGetStackSize) {
  pthread_attr_t attr;
  ASSERT_EQ(LIBC_NAMESPACE::pthread_attr_init(&attr), 0);

  size_t stacksize;
  ASSERT_EQ(
      LIBC_NAMESPACE::pthread_attr_setstacksize(&attr, PTHREAD_STACK_MIN << 2),
      0);
  ASSERT_EQ(LIBC_NAMESPACE::pthread_attr_getstacksize(&attr, &stacksize), 0);
  ASSERT_EQ(stacksize, size_t(PTHREAD_STACK_MIN << 2));

  ASSERT_EQ(
      LIBC_NAMESPACE::pthread_attr_setstacksize(&attr, PTHREAD_STACK_MIN / 2),
      EINVAL);

  ASSERT_EQ(LIBC_NAMESPACE::pthread_attr_destroy(&attr), 0);
}

TEST(LlvmLibcPThreadattrTest, SetAndGetStack) {
  pthread_attr_t attr;
  ASSERT_EQ(LIBC_NAMESPACE::pthread_attr_init(&attr), 0);

  void *stack;
  size_t stacksize;
  ASSERT_EQ(
      LIBC_NAMESPACE::pthread_attr_setstack(&attr, 0, PTHREAD_STACK_MIN << 2),
      0);
  ASSERT_EQ(LIBC_NAMESPACE::pthread_attr_getstack(&attr, &stack, &stacksize),
            0);
  ASSERT_EQ(stacksize, size_t(PTHREAD_STACK_MIN << 2));
  ASSERT_EQ(reinterpret_cast<uintptr_t>(stack), uintptr_t(0));

  ASSERT_EQ(LIBC_NAMESPACE::pthread_attr_setstack(
                &attr, reinterpret_cast<void *>(1), PTHREAD_STACK_MIN << 2),
            EINVAL);
  ASSERT_EQ(
      LIBC_NAMESPACE::pthread_attr_setstack(&attr, 0, PTHREAD_STACK_MIN / 2),
      EINVAL);

  ASSERT_EQ(LIBC_NAMESPACE::pthread_attr_destroy(&attr), 0);
}

TEST(LlvmLibcPThreadattrTest, SetAndGetSchedParam) {
  pthread_attr_t attr;
  ASSERT_EQ(LIBC_NAMESPACE::pthread_attr_init(&attr), 0);

  struct sched_param param;
  ASSERT_EQ(LIBC_NAMESPACE::pthread_attr_getschedparam(&attr, &param), 0);
  ASSERT_EQ(param.sched_priority, 0);

  param.sched_priority = 42;
  ASSERT_EQ(LIBC_NAMESPACE::pthread_attr_setschedparam(&attr, &param), 0);
  param.sched_priority = 0;
  ASSERT_EQ(LIBC_NAMESPACE::pthread_attr_getschedparam(&attr, &param), 0);
  ASSERT_EQ(param.sched_priority, 42);

  // We do not attempt to validate scheduling parameters here. The OS will do
  // that when starting a thread.
  param.sched_priority = -1;
  ASSERT_EQ(LIBC_NAMESPACE::pthread_attr_setschedparam(&attr, &param), 0);
  param.sched_priority = 0;
  ASSERT_EQ(LIBC_NAMESPACE::pthread_attr_getschedparam(&attr, &param), 0);
  ASSERT_EQ(param.sched_priority, -1);

  ASSERT_EQ(LIBC_NAMESPACE::pthread_attr_destroy(&attr), 0);
}

TEST(LlvmLibcPThreadattrTest, SetAndGetSchedPolicy) {
  pthread_attr_t attr;
  ASSERT_EQ(LIBC_NAMESPACE::pthread_attr_init(&attr), 0);

  int policy;
  ASSERT_EQ(LIBC_NAMESPACE::pthread_attr_getschedpolicy(&attr, &policy), 0);
  ASSERT_EQ(policy, SCHED_OTHER);

  ASSERT_EQ(LIBC_NAMESPACE::pthread_attr_setschedpolicy(&attr, SCHED_FIFO), 0);
  ASSERT_EQ(LIBC_NAMESPACE::pthread_attr_getschedpolicy(&attr, &policy), 0);
  ASSERT_EQ(policy, SCHED_FIFO);

  ASSERT_EQ(LIBC_NAMESPACE::pthread_attr_setschedpolicy(&attr, SCHED_RR), 0);
  ASSERT_EQ(LIBC_NAMESPACE::pthread_attr_getschedpolicy(&attr, &policy), 0);
  ASSERT_EQ(policy, SCHED_RR);

  ASSERT_EQ(LIBC_NAMESPACE::pthread_attr_setschedpolicy(&attr, SCHED_OTHER), 0);
  ASSERT_EQ(LIBC_NAMESPACE::pthread_attr_getschedpolicy(&attr, &policy), 0);
  ASSERT_EQ(policy, SCHED_OTHER);

  // We do not attempt to validate scheduling policies here. The OS will do that
  // when starting a thread.
  ASSERT_EQ(LIBC_NAMESPACE::pthread_attr_setschedpolicy(&attr, 0xBAD), 0);
  ASSERT_EQ(LIBC_NAMESPACE::pthread_attr_getschedpolicy(&attr, &policy), 0);
  ASSERT_EQ(policy, 0xBAD);

  ASSERT_EQ(LIBC_NAMESPACE::pthread_attr_setschedpolicy(&attr, -1), 0);
  ASSERT_EQ(LIBC_NAMESPACE::pthread_attr_getschedpolicy(&attr, &policy), 0);
  ASSERT_EQ(policy, -1);

  ASSERT_EQ(LIBC_NAMESPACE::pthread_attr_destroy(&attr), 0);
}

TEST(LlvmLibcPThreadAttrTest, SetAndGetScope) {
  pthread_attr_t attr;
  ASSERT_EQ(LIBC_NAMESPACE::pthread_attr_init(&attr), 0);

  int scope;
  ASSERT_EQ(LIBC_NAMESPACE::pthread_attr_getscope(&attr, &scope), 0);
  ASSERT_EQ(scope, PTHREAD_SCOPE_SYSTEM);

  ASSERT_EQ(LIBC_NAMESPACE::pthread_attr_setscope(&attr, PTHREAD_SCOPE_SYSTEM),
            0);
  ASSERT_EQ(LIBC_NAMESPACE::pthread_attr_getscope(&attr, &scope), 0);
  ASSERT_EQ(scope, PTHREAD_SCOPE_SYSTEM);

  ASSERT_EQ(LIBC_NAMESPACE::pthread_attr_setscope(&attr, PTHREAD_SCOPE_PROCESS),
            ENOTSUP);
  ASSERT_EQ(LIBC_NAMESPACE::pthread_attr_getscope(&attr, &scope), 0);
  ASSERT_EQ(scope, PTHREAD_SCOPE_SYSTEM);

  ASSERT_EQ(LIBC_NAMESPACE::pthread_attr_setscope(&attr, 0xBAD), EINVAL);
  ASSERT_EQ(LIBC_NAMESPACE::pthread_attr_getscope(&attr, &scope), 0);
  ASSERT_EQ(scope, PTHREAD_SCOPE_SYSTEM);

  ASSERT_EQ(LIBC_NAMESPACE::pthread_attr_setscope(&attr, -1), EINVAL);
  ASSERT_EQ(LIBC_NAMESPACE::pthread_attr_getscope(&attr, &scope), 0);
  ASSERT_EQ(scope, PTHREAD_SCOPE_SYSTEM);

  ASSERT_EQ(LIBC_NAMESPACE::pthread_attr_destroy(&attr), 0);
}
