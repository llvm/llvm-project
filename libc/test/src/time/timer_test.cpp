//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unittests for posix timers (timer_create, timer_delete).
///
//===----------------------------------------------------------------------===//

#include "hdr/errno_macros.h"
#include "hdr/signal_macros.h"
#include "hdr/time_macros.h"
#include "hdr/types/clockid_t.h"
#include "hdr/types/struct_itimerspec.h"
#include "hdr/types/struct_sigevent.h"
#include "hdr/types/timer_t.h"
#include "src/time/clock_gettime.h"
#include "src/time/timer_create.h"
#include "src/time/timer_delete.h"
#include "src/time/timer_gettime.h"
#include "src/time/timer_settime.h"
#include "src/unistd/gettid.h"
#include "test/UnitTest/ErrnoCheckingTest.h"
#include "test/UnitTest/ErrnoSetterMatcher.h"
#include "test/UnitTest/Test.h"

using LIBC_NAMESPACE::testing::ErrnoSetterMatcher::any_of;
using LIBC_NAMESPACE::testing::ErrnoSetterMatcher::Fails;
using LIBC_NAMESPACE::testing::ErrnoSetterMatcher::Succeeds;
using LlvmLibcTimerTest = LIBC_NAMESPACE::testing::ErrnoCheckingTest;

TEST_F(LlvmLibcTimerTest, NullSigevent) {
  timer_t timerid;
  ASSERT_THAT(LIBC_NAMESPACE::timer_create(CLOCK_REALTIME, nullptr, &timerid),
              Succeeds(0));
  ASSERT_THAT(LIBC_NAMESPACE::timer_delete(timerid), Succeeds(0));
}

TEST_F(LlvmLibcTimerTest, ValidSigeventSigevNone) {
  sigevent se;
  se.sigev_notify = SIGEV_NONE;
  timer_t timerid;
  ASSERT_THAT(LIBC_NAMESPACE::timer_create(CLOCK_MONOTONIC, &se, &timerid),
              Succeeds(0));
  ASSERT_THAT(LIBC_NAMESPACE::timer_delete(timerid), Succeeds(0));
}

TEST_F(LlvmLibcTimerTest, ValidSigeventSigevSignal) {
  sigevent se;
  se.sigev_notify = SIGEV_SIGNAL;
  se.sigev_signo = SIGALRM;
  se.sigev_value.sival_int = 42;
  timer_t timerid;
  ASSERT_THAT(LIBC_NAMESPACE::timer_create(CLOCK_REALTIME, &se, &timerid),
              Succeeds(0));
  ASSERT_THAT(LIBC_NAMESPACE::timer_delete(timerid), Succeeds(0));
}

#ifdef SIGEV_THREAD_ID
TEST_F(LlvmLibcTimerTest, ValidSigeventSigevThreadId) {
  sigevent se;
  se.sigev_notify = SIGEV_THREAD_ID;
  se.sigev_signo = SIGALRM;
  se.sigev_value.sival_int = 42;
  se.sigev_notify_thread_id = LIBC_NAMESPACE::gettid();
  timer_t timerid;
  ASSERT_THAT(LIBC_NAMESPACE::timer_create(CLOCK_REALTIME, &se, &timerid),
              Succeeds(0));
  ASSERT_THAT(LIBC_NAMESPACE::timer_delete(timerid), Succeeds(0));
}
#endif

TEST_F(LlvmLibcTimerTest, InvalidClockId) {
  timer_t timerid;
  ASSERT_THAT(LIBC_NAMESPACE::timer_create(static_cast<clockid_t>(-1), nullptr,
                                           &timerid),
              Fails(EINVAL));
}

TEST_F(LlvmLibcTimerTest, NullTimerId) {
  ASSERT_THAT(LIBC_NAMESPACE::timer_create(CLOCK_REALTIME, nullptr, nullptr),
              Fails(any_of(EINVAL, EFAULT)));
}

TEST_F(LlvmLibcTimerTest, DeleteInvalidTimerId) {
  ASSERT_THAT(LIBC_NAMESPACE::timer_delete(reinterpret_cast<timer_t>(-1)),
              Fails(EINVAL));
}

TEST_F(LlvmLibcTimerTest, CreateAndDelete) {
  timer_t timerid;
  ASSERT_THAT(LIBC_NAMESPACE::timer_create(CLOCK_REALTIME, nullptr, &timerid),
              Succeeds(0));
  ASSERT_THAT(LIBC_NAMESPACE::timer_delete(timerid), Succeeds(0));
  ASSERT_THAT(LIBC_NAMESPACE::timer_delete(timerid), Fails(EINVAL));
}

TEST_F(LlvmLibcTimerTest, SetTimeRelativeAndDisarm) {
  timer_t timerid;
  ASSERT_THAT(LIBC_NAMESPACE::timer_create(CLOCK_REALTIME, nullptr, &timerid),
              Succeeds(0));

  itimerspec new_val{};
  new_val.it_value.tv_sec = 10;
  new_val.it_value.tv_nsec = 0;
  new_val.it_interval.tv_sec = 1;
  new_val.it_interval.tv_nsec = 0;

  itimerspec old_val{};
  ASSERT_THAT(LIBC_NAMESPACE::timer_settime(timerid, 0, &new_val, &old_val),
              Succeeds(0));
  // Initially, old_val should be 0 because it was disarmed.
  ASSERT_EQ(old_val.it_value.tv_sec, static_cast<time_t>(0));
  ASSERT_EQ(old_val.it_value.tv_nsec,
            static_cast<decltype(old_val.it_value.tv_nsec)>(0));

  // Disarm the timer by setting it_value to 0
  itimerspec disarm_val{};
  ASSERT_THAT(LIBC_NAMESPACE::timer_settime(timerid, 0, &disarm_val, &old_val),
              Succeeds(0));
  // old_val should have been armed previously
  ASSERT_GT(old_val.it_value.tv_sec, static_cast<time_t>(0));

  ASSERT_THAT(LIBC_NAMESPACE::timer_delete(timerid), Succeeds(0));
}

TEST_F(LlvmLibcTimerTest, SetTimeAbsolute) {
  timer_t timerid;
  ASSERT_THAT(LIBC_NAMESPACE::timer_create(CLOCK_REALTIME, nullptr, &timerid),
              Succeeds(0));

  timespec now{};
  ASSERT_THAT(LIBC_NAMESPACE::clock_gettime(CLOCK_REALTIME, &now), Succeeds(0));

  itimerspec new_val{};
  new_val.it_value.tv_sec = now.tv_sec + 100;
  new_val.it_value.tv_nsec = 0;

  ASSERT_THAT(
      LIBC_NAMESPACE::timer_settime(timerid, TIMER_ABSTIME, &new_val, nullptr),
      Succeeds(0));

  ASSERT_THAT(LIBC_NAMESPACE::timer_delete(timerid), Succeeds(0));
}

TEST_F(LlvmLibcTimerTest, SetTimeInvalidTimerId) {
  itimerspec new_val{};
  new_val.it_value.tv_sec = 1;
  ASSERT_THAT(LIBC_NAMESPACE::timer_settime(reinterpret_cast<timer_t>(-1), 0,
                                            &new_val, nullptr),
              Fails(EINVAL));
}

TEST_F(LlvmLibcTimerTest, SetTimeNullNewValue) {
  timer_t timerid;
  ASSERT_THAT(LIBC_NAMESPACE::timer_create(CLOCK_REALTIME, nullptr, &timerid),
              Succeeds(0));

  ASSERT_THAT(LIBC_NAMESPACE::timer_settime(timerid, 0, nullptr, nullptr),
              Fails(any_of(EINVAL, EFAULT)));

  ASSERT_THAT(LIBC_NAMESPACE::timer_delete(timerid), Succeeds(0));
}

TEST_F(LlvmLibcTimerTest, SetTimeInvalidNanoseconds) {
  timer_t timerid;
  ASSERT_THAT(LIBC_NAMESPACE::timer_create(CLOCK_REALTIME, nullptr, &timerid),
              Succeeds(0));

  itimerspec new_val{};
  new_val.it_value.tv_sec = 1;
  new_val.it_value.tv_nsec = 1000000000; // invalid >= 10^9

  ASSERT_THAT(LIBC_NAMESPACE::timer_settime(timerid, 0, &new_val, nullptr),
              Fails(EINVAL));

  new_val.it_value.tv_nsec = -1; // invalid < 0
  ASSERT_THAT(LIBC_NAMESPACE::timer_settime(timerid, 0, &new_val, nullptr),
              Fails(EINVAL));

  ASSERT_THAT(LIBC_NAMESPACE::timer_delete(timerid), Succeeds(0));
}

TEST_F(LlvmLibcTimerTest, GetTimeDisarmedAndArmed) {
  timer_t timerid;
  ASSERT_THAT(LIBC_NAMESPACE::timer_create(CLOCK_REALTIME, nullptr, &timerid),
              Succeeds(0));

  itimerspec curr_val{};
  ASSERT_THAT(LIBC_NAMESPACE::timer_gettime(timerid, &curr_val), Succeeds(0));
  ASSERT_EQ(curr_val.it_value.tv_sec, static_cast<time_t>(0));
  ASSERT_EQ(curr_val.it_value.tv_nsec,
            static_cast<decltype(curr_val.it_value.tv_nsec)>(0));

  itimerspec new_val{};
  new_val.it_value.tv_sec = 20;
  new_val.it_value.tv_nsec = 0;
  new_val.it_interval.tv_sec = 2;
  new_val.it_interval.tv_nsec = 0;

  ASSERT_THAT(LIBC_NAMESPACE::timer_settime(timerid, 0, &new_val, nullptr),
              Succeeds(0));

  ASSERT_THAT(LIBC_NAMESPACE::timer_gettime(timerid, &curr_val), Succeeds(0));
  ASSERT_GT(curr_val.it_value.tv_sec, static_cast<time_t>(0));
  ASSERT_EQ(curr_val.it_interval.tv_sec, static_cast<time_t>(2));

  ASSERT_THAT(LIBC_NAMESPACE::timer_delete(timerid), Succeeds(0));
}

TEST_F(LlvmLibcTimerTest, GetTimeInvalidTimerId) {
  itimerspec curr_val{};
  ASSERT_THAT(
      LIBC_NAMESPACE::timer_gettime(reinterpret_cast<timer_t>(-1), &curr_val),
      Fails(EINVAL));
}

TEST_F(LlvmLibcTimerTest, GetTimeNullPointer) {
  timer_t timerid;
  ASSERT_THAT(LIBC_NAMESPACE::timer_create(CLOCK_REALTIME, nullptr, &timerid),
              Succeeds(0));

  ASSERT_THAT(LIBC_NAMESPACE::timer_gettime(timerid, nullptr),
              Fails(any_of(EINVAL, EFAULT)));

  ASSERT_THAT(LIBC_NAMESPACE::timer_delete(timerid), Succeeds(0));
}
