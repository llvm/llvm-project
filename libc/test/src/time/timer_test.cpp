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
#include "hdr/types/struct_sigevent.h"
#include "hdr/types/timer_t.h"
#include "src/__support/OSUtil/syscall.h"
#include "src/time/timer_create.h"
#include "src/time/timer_delete.h"
#include "test/UnitTest/ErrnoCheckingTest.h"
#include "test/UnitTest/ErrnoSetterMatcher.h"
#include "test/UnitTest/Test.h"
#include <sys/syscall.h>

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
  struct sigevent se;
  se.sigev_notify = SIGEV_NONE;
  timer_t timerid;
  ASSERT_THAT(LIBC_NAMESPACE::timer_create(CLOCK_MONOTONIC, &se, &timerid),
              Succeeds(0));
  ASSERT_THAT(LIBC_NAMESPACE::timer_delete(timerid), Succeeds(0));
}

TEST_F(LlvmLibcTimerTest, ValidSigeventSigevSignal) {
  struct sigevent se;
  se.sigev_notify = SIGEV_SIGNAL;
  se.sigev_signo = SIGALRM;
  se.sigev_value.sival_int = 42;
  timer_t timerid;
  ASSERT_THAT(LIBC_NAMESPACE::timer_create(CLOCK_REALTIME, &se, &timerid),
              Succeeds(0));
  ASSERT_THAT(LIBC_NAMESPACE::timer_delete(timerid), Succeeds(0));
}

#if defined(SYS_gettid) && defined(SIGEV_THREAD_ID)
TEST_F(LlvmLibcTimerTest, ValidSigeventSigevThreadId) {
  struct sigevent se;
  se.sigev_notify = SIGEV_THREAD_ID;
  se.sigev_signo = SIGALRM;
  se.sigev_value.sival_int = 42;
  se.sigev_notify_thread_id = LIBC_NAMESPACE::syscall_impl<pid_t>(SYS_gettid);
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
