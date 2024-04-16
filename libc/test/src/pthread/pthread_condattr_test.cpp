//===-- Unittests for pthread_condattr_t ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "test/UnitTest/Test.h"

#include <errno.h>
#include <pthread.h>
#include <time.h>

TEST(LlvmLibcPThreadCondAttrTest, InitAndDestroy) {
  pthread_condattr_t cond;
  ASSERT_EQ(pthread_condattr_init(&cond), 0);
  ASSERT_EQ(pthread_condattr_destroy(&cond), 0);
}

TEST(LlvmLibcPThreadCondAttrTest, GetDefaultValues) {
  pthread_condattr_t cond;

  // Invalid clock id.
  clockid_t clock = 7;
  // Invalid value.
  int pshared = 42;

  ASSERT_EQ(pthread_condattr_init(&cond), 0);
  ASSERT_EQ(pthread_condattr_getclock(&cond, &clock), 0);
  ASSERT_EQ(clock, CLOCK_REALTIME);
  ASSERT_EQ(pthread_condattr_getpshared(&cond, &pshared), 0);
  ASSERT_EQ(pshared, PTHREAD_PROCESS_PRIVATE);
  ASSERT_EQ(pthread_condattr_destroy(&cond), 0);
}

TEST(LlvmLibcPThreadCondAttrTest, SetGoodValues) {
  pthread_condattr_t cond;

  // Invalid clock id.
  clockid_t clock = 7;
  // Invalid value.
  int pshared = 42;

  ASSERT_EQ(pthread_condattr_init(&cond), 0);
  ASSERT_EQ(pthread_condattr_setclock(&cond, CLOCK_MONOTONIC), 0);
  ASSERT_EQ(pthread_condattr_getclock(&cond, &clock), 0);
  ASSERT_EQ(clock, CLOCK_MONOTONIC);
  ASSERT_EQ(pthread_condattr_setpshared(&cond, PTHREAD_PROCESS_SHARED), 0);
  ASSERT_EQ(pthread_condattr_getpshared(&cond, &pshared), 0);
  ASSERT_EQ(pshared, PTHREAD_PROCESS_SHARED);
  ASSERT_EQ(pthread_condattr_destroy(&cond), 0);
}

TEST(LlvmLibcPThreadCondAttrTest, SetBadValues) {
  pthread_condattr_t cond;

  // Invalid clock id.
  clockid_t clock = 7;
  // Invalid value.
  int pshared = 42;

  ASSERT_EQ(pthread_condattr_init(&cond), 0);
  ASSERT_EQ(pthread_condattr_setclock(&cond, clock), EINVAL);
  ASSERT_EQ(pthread_condattr_getclock(&cond, &clock), 0);
  ASSERT_EQ(clock, CLOCK_REALTIME);
  ASSERT_EQ(pthread_condattr_setpshared(&cond, pshared), EINVAL);
  ASSERT_EQ(pthread_condattr_getpshared(&cond, &pshared), 0);
  ASSERT_EQ(pshared, PTHREAD_PROCESS_PRIVATE);
  ASSERT_EQ(pthread_condattr_destroy(&cond), 0);
}
