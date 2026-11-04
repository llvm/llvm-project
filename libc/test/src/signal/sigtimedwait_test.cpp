//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unittests for sigtimedwait.
///
//===----------------------------------------------------------------------===//

#include "hdr/errno_macros.h"
#include "hdr/signal_macros.h"
#include "hdr/types/siginfo_t.h"
#include "hdr/types/sigset_t.h"
#include "hdr/types/struct_sigaction.h"
#include "hdr/types/struct_timespec.h"
#include "src/signal/raise.h"
#include "src/signal/sigaction.h"
#include "src/signal/sigaddset.h"
#include "src/signal/sigemptyset.h"
#include "src/signal/sigprocmask.h"
#include "src/signal/sigtimedwait.h"
#include "test/UnitTest/ErrnoCheckingTest.h"
#include "test/UnitTest/ErrnoSetterMatcher.h"
#include "test/UnitTest/Test.h"

using LIBC_NAMESPACE::testing::ErrnoSetterMatcher::Fails;
using LIBC_NAMESPACE::testing::ErrnoSetterMatcher::Succeeds;

static void test_sig_handler(int) {}

class LlvmLibcSigtimedwaitTest
    : public LIBC_NAMESPACE::testing::ErrnoCheckingTest {
  struct sigaction old_action{};
  sigset_t old_mask{};

public:
  void SetUp() override {
    ErrnoCheckingTest::SetUp();
    ASSERT_THAT(LIBC_NAMESPACE::sigprocmask(0, nullptr, &old_mask), Succeeds());

    struct sigaction sa{};
    sa.sa_handler = test_sig_handler;
    ASSERT_THAT(LIBC_NAMESPACE::sigemptyset(&sa.sa_mask), Succeeds());
    sa.sa_flags = 0;
    ASSERT_THAT(LIBC_NAMESPACE::sigaction(SIGUSR1, &sa, &old_action),
                Succeeds());

    sigset_t blocked;
    ASSERT_THAT(LIBC_NAMESPACE::sigemptyset(&blocked), Succeeds());
    ASSERT_THAT(LIBC_NAMESPACE::sigaddset(&blocked, SIGUSR1), Succeeds());
    ASSERT_THAT(LIBC_NAMESPACE::sigprocmask(SIG_BLOCK, &blocked, nullptr),
                Succeeds());
  }

  void TearDown() override {
    LIBC_NAMESPACE::sigprocmask(SIG_SETMASK, &old_mask, nullptr);
    LIBC_NAMESPACE::sigaction(SIGUSR1, &old_action, nullptr);
    ErrnoCheckingTest::TearDown();
  }
};

// This tests that a pending signal is returned along with its info.
TEST_F(LlvmLibcSigtimedwaitTest, PendingSignalReturnsSignalAndInfo) {
  sigset_t set;
  ASSERT_THAT(LIBC_NAMESPACE::sigemptyset(&set), Succeeds());
  ASSERT_THAT(LIBC_NAMESPACE::sigaddset(&set, SIGUSR1), Succeeds());
  ASSERT_THAT(LIBC_NAMESPACE::raise(SIGUSR1), Succeeds());

  struct timespec timeout{0, 0};
  siginfo_t info{};
  ASSERT_THAT(LIBC_NAMESPACE::sigtimedwait(&set, &info, &timeout),
              Succeeds(SIGUSR1));
  EXPECT_EQ(info.si_signo, SIGUSR1);

  EXPECT_THAT(LIBC_NAMESPACE::sigtimedwait(&set, nullptr, &timeout),
              Fails(EAGAIN));
}

// This tests that a null info pointer is allowed.
TEST_F(LlvmLibcSigtimedwaitTest, NullInfo) {
  sigset_t set;
  ASSERT_THAT(LIBC_NAMESPACE::sigemptyset(&set), Succeeds());
  ASSERT_THAT(LIBC_NAMESPACE::sigaddset(&set, SIGUSR1), Succeeds());
  ASSERT_THAT(LIBC_NAMESPACE::raise(SIGUSR1), Succeeds());

  struct timespec timeout{0, 0};
  ASSERT_THAT(LIBC_NAMESPACE::sigtimedwait(&set, nullptr, &timeout),
              Succeeds(SIGUSR1));
}

// This tests that a null timeout waits until a pending signal arrives.
TEST_F(LlvmLibcSigtimedwaitTest, NullTimeoutWithPendingSignal) {
  sigset_t set;
  ASSERT_THAT(LIBC_NAMESPACE::sigemptyset(&set), Succeeds());
  ASSERT_THAT(LIBC_NAMESPACE::sigaddset(&set, SIGUSR1), Succeeds());
  ASSERT_THAT(LIBC_NAMESPACE::raise(SIGUSR1), Succeeds());

  siginfo_t info{};
  ASSERT_THAT(LIBC_NAMESPACE::sigtimedwait(&set, &info, nullptr),
              Succeeds(SIGUSR1));
  EXPECT_EQ(info.si_signo, SIGUSR1);
}

// This tests that no pending signal with a zero timeout returns EAGAIN.
TEST_F(LlvmLibcSigtimedwaitTest, NoPendingSignalTimesOut) {
  sigset_t set;
  ASSERT_THAT(LIBC_NAMESPACE::sigemptyset(&set), Succeeds());

  struct timespec timeout{0, 0};
  EXPECT_THAT(LIBC_NAMESPACE::sigtimedwait(&set, nullptr, &timeout),
              Fails(EAGAIN));
}

// This tests that an invalid timeout returns EINVAL.
TEST_F(LlvmLibcSigtimedwaitTest, InvalidTimeout) {
  sigset_t set;
  ASSERT_THAT(LIBC_NAMESPACE::sigemptyset(&set), Succeeds());

  struct timespec timeout{0, 1000000000L};
  EXPECT_THAT(LIBC_NAMESPACE::sigtimedwait(&set, nullptr, &timeout),
              Fails(EINVAL));
}
