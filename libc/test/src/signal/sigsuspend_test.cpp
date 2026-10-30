//===-- Unittests for sigsuspend ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "hdr/errno_macros.h"
#include "hdr/signal_macros.h"
#include "hdr/types/sigset_t.h"
#include "hdr/types/struct_sigaction.h"
#include "src/signal/raise.h"
#include "src/signal/sigaction.h"
#include "src/signal/sigaddset.h"
#include "src/signal/sigemptyset.h"
#include "src/signal/sigprocmask.h"
#include "src/signal/sigsuspend.h"
#include "test/UnitTest/ErrnoCheckingTest.h"
#include "test/UnitTest/ErrnoSetterMatcher.h"
#include "test/UnitTest/Test.h"

static volatile bool signal_caught = false;

static void test_sig_handler(int signo) {
  if (signo == SIGUSR1)
    signal_caught = true;
}

class LlvmLibcSigsuspendInterruptTest
    : public LIBC_NAMESPACE::testing::ErrnoCheckingTest {
  struct sigaction old_action{};
  sigset_t old_mask{};

public:
  void SetUp() override {
    ErrnoCheckingTest::SetUp();
    signal_caught = false;
    ASSERT_THAT(LIBC_NAMESPACE::sigprocmask(0, nullptr, &old_mask), Succeeds());
    struct sigaction sa{};
    sa.sa_handler = test_sig_handler;
    ASSERT_THAT(LIBC_NAMESPACE::sigemptyset(&sa.sa_mask), Succeeds());
    sa.sa_flags = 0;
    ASSERT_THAT(LIBC_NAMESPACE::sigaction(SIGUSR1, &sa, &old_action),
                Succeeds());
  }

  void TearDown() override {
    LIBC_NAMESPACE::sigaction(SIGUSR1, &old_action, nullptr);
    LIBC_NAMESPACE::sigprocmask(SIG_SETMASK, &old_mask, nullptr);
    ErrnoCheckingTest::TearDown();
  }
};

using LIBC_NAMESPACE::testing::ErrnoSetterMatcher::Fails;
using LIBC_NAMESPACE::testing::ErrnoSetterMatcher::Succeeds;

// This tests if it correctly catches a pending signal.
TEST_F(LlvmLibcSigsuspendInterruptTest, SigsuspendCaughtByHandler) {
  sigset_t block, empty;
  EXPECT_THAT(LIBC_NAMESPACE::sigemptyset(&block), Succeeds());
  EXPECT_THAT(LIBC_NAMESPACE::sigaddset(&block, SIGUSR1), Succeeds());
  EXPECT_THAT(LIBC_NAMESPACE::sigprocmask(SIG_BLOCK, &block, nullptr),
              Succeeds());
  EXPECT_THAT(LIBC_NAMESPACE::raise(SIGUSR1), Succeeds());
  EXPECT_THAT(LIBC_NAMESPACE::sigemptyset(&empty), Succeeds());
  EXPECT_THAT(LIBC_NAMESPACE::sigsuspend(&empty), Fails(EINTR));
  EXPECT_TRUE(signal_caught);
}
