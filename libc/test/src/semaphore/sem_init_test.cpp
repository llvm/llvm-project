//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unittests for sem_init and sem_destroy.
///
//===----------------------------------------------------------------------===//

#include "hdr/errno_macros.h"
#include "hdr/limits_macros.h"
#include "hdr/time_macros.h"
#include "hdr/types/sem_t.h"
#include "hdr/types/struct_timespec.h"
#include "src/semaphore/sem_clockwait.h"
#include "src/semaphore/sem_destroy.h"
#include "src/semaphore/sem_getvalue.h"
#include "src/semaphore/sem_init.h"
#include "src/semaphore/sem_post.h"
#include "src/semaphore/sem_timedwait.h"
#include "src/semaphore/sem_trywait.h"
#include "src/semaphore/sem_wait.h"
#include "test/UnitTest/ErrnoCheckingTest.h"
#include "test/UnitTest/ErrnoSetterMatcher.h"
#include "test/UnitTest/Test.h"

using LIBC_NAMESPACE::testing::ErrnoSetterMatcher::Fails;
using LIBC_NAMESPACE::testing::ErrnoSetterMatcher::Succeeds;
using LlvmLibcSemInitTest = LIBC_NAMESPACE::testing::ErrnoCheckingTest;

TEST_F(LlvmLibcSemInitTest, InitAndDestroy) {
  sem_t sem;
  ASSERT_THAT(LIBC_NAMESPACE::sem_init(&sem, 0, 3), Succeeds());
  ASSERT_THAT(LIBC_NAMESPACE::sem_destroy(&sem), Succeeds());
}

TEST_F(LlvmLibcSemInitTest, InitValueTooLarge) {
  sem_t sem;
  // A count above SEM_VALUE_MAX cannot be represented.
  ASSERT_THAT(LIBC_NAMESPACE::sem_init(
                  &sem, 0, static_cast<unsigned int>(SEM_VALUE_MAX) + 1),
              Fails(EINVAL));
}

TEST_F(LlvmLibcSemInitTest, InitShared) {
  sem_t sem;
  // A non-zero pshared is accepted; the semaphore then uses the shared futex
  // addressing mode.
  ASSERT_THAT(LIBC_NAMESPACE::sem_init(&sem, 1, 1), Succeeds());
  ASSERT_THAT(LIBC_NAMESPACE::sem_wait(&sem), Succeeds());
  ASSERT_THAT(LIBC_NAMESPACE::sem_post(&sem), Succeeds());
  ASSERT_THAT(LIBC_NAMESPACE::sem_destroy(&sem), Succeeds());
}

TEST_F(LlvmLibcSemInitTest, UseAfterDestroy) {
  sem_t sem;
  ASSERT_THAT(LIBC_NAMESPACE::sem_init(&sem, 0, 1), Succeeds());
  ASSERT_THAT(LIBC_NAMESPACE::sem_destroy(&sem), Succeeds());

  // The canary no longer matches, so every operation reports EINVAL rather
  // than acting on a destroyed semaphore.
  int value = -1;
  EXPECT_THAT(LIBC_NAMESPACE::sem_getvalue(&sem, &value), Fails(EINVAL));
  EXPECT_THAT(LIBC_NAMESPACE::sem_post(&sem), Fails(EINVAL));
  EXPECT_THAT(LIBC_NAMESPACE::sem_wait(&sem), Fails(EINVAL));
  EXPECT_THAT(LIBC_NAMESPACE::sem_trywait(&sem), Fails(EINVAL));
  EXPECT_THAT(LIBC_NAMESPACE::sem_destroy(&sem), Fails(EINVAL));

  // The deadline is valid, so EINVAL can only come from the canary check.
  timespec ts{};
  ts.tv_sec = 1;
  ts.tv_nsec = 0;
  EXPECT_THAT(LIBC_NAMESPACE::sem_timedwait(&sem, &ts), Fails(EINVAL));
  EXPECT_THAT(LIBC_NAMESPACE::sem_clockwait(&sem, CLOCK_MONOTONIC, &ts),
              Fails(EINVAL));
}
