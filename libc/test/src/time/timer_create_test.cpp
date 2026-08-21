//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unittests for timer_create.
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
#include "test/UnitTest/ErrnoCheckingTest.h"
#include "test/UnitTest/ErrnoSetterMatcher.h"
#include "test/UnitTest/Test.h"
#include <sys/syscall.h>

using LIBC_NAMESPACE::testing::ErrnoSetterMatcher::any_of;
using LIBC_NAMESPACE::testing::ErrnoSetterMatcher::Fails;
using LIBC_NAMESPACE::testing::ErrnoSetterMatcher::Succeeds;
using LlvmLibcTimerCreateTest = LIBC_NAMESPACE::testing::ErrnoCheckingTest;

TEST_F(LlvmLibcTimerCreateTest, NullSigevent) {
  timer_t timerid;
  ASSERT_THAT(LIBC_NAMESPACE::timer_create(CLOCK_REALTIME, nullptr, &timerid),
              Succeeds(0));

#ifdef SYS_timer_delete
  LIBC_NAMESPACE::syscall_impl<int>(SYS_timer_delete, timerid);
#endif
}

TEST_F(LlvmLibcTimerCreateTest, ValidSigeventSigevNone) {
  struct sigevent se;
  se.sigev_notify = SIGEV_NONE;
  timer_t timerid;
  ASSERT_THAT(LIBC_NAMESPACE::timer_create(CLOCK_MONOTONIC, &se, &timerid),
              Succeeds(0));

#ifdef SYS_timer_delete
  LIBC_NAMESPACE::syscall_impl<int>(SYS_timer_delete, timerid);
#endif
}

TEST_F(LlvmLibcTimerCreateTest, ValidSigeventSigevSignal) {
  struct sigevent se;
  se.sigev_notify = SIGEV_SIGNAL;
  se.sigev_signo = SIGALRM;
  se.sigev_value.sival_int = 42;
  timer_t timerid;
  ASSERT_THAT(LIBC_NAMESPACE::timer_create(CLOCK_REALTIME, &se, &timerid),
              Succeeds(0));

#ifdef SYS_timer_delete
  LIBC_NAMESPACE::syscall_impl<int>(SYS_timer_delete, timerid);
#endif
}

TEST_F(LlvmLibcTimerCreateTest, InvalidClockId) {
  timer_t timerid;
  ASSERT_THAT(LIBC_NAMESPACE::timer_create(static_cast<clockid_t>(-1), nullptr,
                                           &timerid),
              Fails(EINVAL));
}

TEST_F(LlvmLibcTimerCreateTest, NullTimerId) {
  ASSERT_THAT(LIBC_NAMESPACE::timer_create(CLOCK_REALTIME, nullptr, nullptr),
              Fails(any_of(EINVAL, EFAULT)));
}
