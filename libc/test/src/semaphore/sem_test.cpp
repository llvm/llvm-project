//===-- Unittests for unnamed POSIX semaphores ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "hdr/errno_macros.h"
#include "hdr/limits_macros.h"
#include "hdr/time_macros.h"
#include "hdr/types/sem_t.h"
#include "hdr/types/struct_timespec.h"
#include "src/__support/time/clock_gettime.h"
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
using LlvmLibcSemTest = LIBC_NAMESPACE::testing::ErrnoCheckingTest;

// Returns an absolute deadline a few milliseconds past now on the given clock.
static timespec deadline_in_10ms(clockid_t clock_id) {
  timespec ts{};
  LIBC_NAMESPACE::internal::clock_gettime(clock_id, &ts);
  ts.tv_nsec += 10'000'000;
  if (ts.tv_nsec >= 1'000'000'000) {
    ts.tv_sec += 1;
    ts.tv_nsec -= 1'000'000'000;
  }
  return ts;
}

TEST_F(LlvmLibcSemTest, InitGetValueDestroy) {
  sem_t sem;
  ASSERT_THAT(LIBC_NAMESPACE::sem_init(&sem, 0, 3), Succeeds());

  int value = -1;
  ASSERT_THAT(LIBC_NAMESPACE::sem_getvalue(&sem, &value), Succeeds());
  EXPECT_EQ(value, 3);

  ASSERT_THAT(LIBC_NAMESPACE::sem_destroy(&sem), Succeeds());
}

TEST_F(LlvmLibcSemTest, InitValueTooLarge) {
  sem_t sem;
  // A count above SEM_VALUE_MAX cannot be represented.
  ASSERT_THAT(LIBC_NAMESPACE::sem_init(
                  &sem, 0, static_cast<unsigned int>(SEM_VALUE_MAX) + 1),
              Fails(EINVAL));
}

TEST_F(LlvmLibcSemTest, InitShared) {
  sem_t sem;
  // A non-zero pshared is accepted; the semaphore then uses the shared futex
  // addressing mode.
  ASSERT_THAT(LIBC_NAMESPACE::sem_init(&sem, 1, 1), Succeeds());
  ASSERT_THAT(LIBC_NAMESPACE::sem_wait(&sem), Succeeds());
  ASSERT_THAT(LIBC_NAMESPACE::sem_post(&sem), Succeeds());
  ASSERT_THAT(LIBC_NAMESPACE::sem_destroy(&sem), Succeeds());
}

TEST_F(LlvmLibcSemTest, UseAfterDestroy) {
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

  timespec ts = deadline_in_10ms(CLOCK_REALTIME);
  EXPECT_THAT(LIBC_NAMESPACE::sem_timedwait(&sem, &ts), Fails(EINVAL));
  EXPECT_THAT(LIBC_NAMESPACE::sem_clockwait(&sem, CLOCK_MONOTONIC, &ts),
              Fails(EINVAL));
}

TEST_F(LlvmLibcSemTest, PostAndWait) {
  sem_t sem;
  ASSERT_THAT(LIBC_NAMESPACE::sem_init(&sem, 0, 0), Succeeds());

  ASSERT_THAT(LIBC_NAMESPACE::sem_post(&sem), Succeeds());
  ASSERT_THAT(LIBC_NAMESPACE::sem_post(&sem), Succeeds());

  int value = -1;
  ASSERT_THAT(LIBC_NAMESPACE::sem_getvalue(&sem, &value), Succeeds());
  EXPECT_EQ(value, 2);

  // The count is positive, so neither wait blocks.
  ASSERT_THAT(LIBC_NAMESPACE::sem_wait(&sem), Succeeds());
  ASSERT_THAT(LIBC_NAMESPACE::sem_wait(&sem), Succeeds());

  ASSERT_THAT(LIBC_NAMESPACE::sem_getvalue(&sem, &value), Succeeds());
  EXPECT_EQ(value, 0);

  ASSERT_THAT(LIBC_NAMESPACE::sem_destroy(&sem), Succeeds());
}

TEST_F(LlvmLibcSemTest, TryWait) {
  sem_t sem;
  ASSERT_THAT(LIBC_NAMESPACE::sem_init(&sem, 0, 1), Succeeds());

  ASSERT_THAT(LIBC_NAMESPACE::sem_trywait(&sem), Succeeds());

  // The count is now zero, so a non-blocking wait reports EAGAIN.
  ASSERT_THAT(LIBC_NAMESPACE::sem_trywait(&sem), Fails(EAGAIN));

  ASSERT_THAT(LIBC_NAMESPACE::sem_destroy(&sem), Succeeds());
}

TEST_F(LlvmLibcSemTest, PostOverflow) {
  sem_t sem;
  ASSERT_THAT(LIBC_NAMESPACE::sem_init(
                  &sem, 0, static_cast<unsigned int>(SEM_VALUE_MAX)),
              Succeeds());

  // Incrementing past SEM_VALUE_MAX is refused and leaves the count alone.
  ASSERT_THAT(LIBC_NAMESPACE::sem_post(&sem), Fails(EOVERFLOW));

  int value = -1;
  ASSERT_THAT(LIBC_NAMESPACE::sem_getvalue(&sem, &value), Succeeds());
  EXPECT_EQ(value, SEM_VALUE_MAX);

  ASSERT_THAT(LIBC_NAMESPACE::sem_destroy(&sem), Succeeds());
}

TEST_F(LlvmLibcSemTest, TimedWaitLocksImmediately) {
  sem_t sem;
  ASSERT_THAT(LIBC_NAMESPACE::sem_init(&sem, 0, 1), Succeeds());

  // The semaphore can be locked right away, so the deadline is never read.
  ASSERT_THAT(LIBC_NAMESPACE::sem_timedwait(&sem, nullptr), Succeeds());
  ASSERT_THAT(LIBC_NAMESPACE::sem_destroy(&sem), Succeeds());
}

TEST_F(LlvmLibcSemTest, TimedWaitTimeout) {
  sem_t sem;
  ASSERT_THAT(LIBC_NAMESPACE::sem_init(&sem, 0, 0), Succeeds());

  timespec ts = deadline_in_10ms(CLOCK_REALTIME);
  ASSERT_THAT(LIBC_NAMESPACE::sem_timedwait(&sem, &ts), Fails(ETIMEDOUT));

  ASSERT_THAT(LIBC_NAMESPACE::sem_destroy(&sem), Succeeds());
}

TEST_F(LlvmLibcSemTest, TimedWaitInvalidTimespec) {
  sem_t sem;
  ASSERT_THAT(LIBC_NAMESPACE::sem_init(&sem, 0, 0), Succeeds());

  timespec ts{};
  ts.tv_sec = 1;
  ts.tv_nsec = 1'000'000'001;
  ASSERT_THAT(LIBC_NAMESPACE::sem_timedwait(&sem, &ts), Fails(EINVAL));

  ASSERT_THAT(LIBC_NAMESPACE::sem_destroy(&sem), Succeeds());
}

TEST_F(LlvmLibcSemTest, ClockWaitMonotonicTimeout) {
  sem_t sem;
  ASSERT_THAT(LIBC_NAMESPACE::sem_init(&sem, 0, 0), Succeeds());

  timespec ts = deadline_in_10ms(CLOCK_MONOTONIC);
  ASSERT_THAT(LIBC_NAMESPACE::sem_clockwait(&sem, CLOCK_MONOTONIC, &ts),
              Fails(ETIMEDOUT));

  ASSERT_THAT(LIBC_NAMESPACE::sem_destroy(&sem), Succeeds());
}

TEST_F(LlvmLibcSemTest, ClockWaitUnsupportedClock) {
  sem_t sem;
  ASSERT_THAT(LIBC_NAMESPACE::sem_init(&sem, 0, 0), Succeeds());

  timespec ts{};
  ts.tv_sec = 1;
  ts.tv_nsec = 0;
  ASSERT_THAT(
      LIBC_NAMESPACE::sem_clockwait(&sem, static_cast<clockid_t>(99), &ts),
      Fails(EINVAL));

  ASSERT_THAT(LIBC_NAMESPACE::sem_destroy(&sem), Succeeds());
}
