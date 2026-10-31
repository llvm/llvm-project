//===-- Unittests for sigpending ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/signal/sigpending.h"

#include "hdr/signal_macros.h"
#include "hdr/types/sigset_t.h"
#include "src/signal/raise.h"
#include "src/signal/sigaddset.h"
#include "src/signal/sigemptyset.h"
#include "src/signal/sigismember.h"
#include "src/signal/sigprocmask.h"
#include "test/UnitTest/ErrnoSetterMatcher.h"
#include "test/UnitTest/Test.h"

using LIBC_NAMESPACE::testing::ErrnoSetterMatcher::Succeeds;

TEST(LlvmLibcSigpendingTest, NoBlockedSignals) {
  sigset_t pending;
  ASSERT_THAT(LIBC_NAMESPACE::sigpending(&pending), Succeeds(0));
}

TEST(LlvmLibcSigpendingTest, BlockAndRaise) {
  // Block SIGUSR1.
  sigset_t blockset;
  ASSERT_EQ(LIBC_NAMESPACE::sigemptyset(&blockset), 0);
  ASSERT_EQ(LIBC_NAMESPACE::sigaddset(&blockset, SIGUSR1), 0);

  sigset_t oldset;
  ASSERT_EQ(LIBC_NAMESPACE::sigprocmask(SIG_BLOCK, &blockset, &oldset), 0);

  // Raise SIGUSR1; it should become pending since it is blocked.
  ASSERT_EQ(LIBC_NAMESPACE::raise(SIGUSR1), 0);

  // Check that SIGUSR1 is now pending.
  sigset_t pending;
  ASSERT_THAT(LIBC_NAMESPACE::sigpending(&pending), Succeeds(0));
  ASSERT_EQ(LIBC_NAMESPACE::sigismember(&pending, SIGUSR1), 1);

  // Restore the old signal mask.
  ASSERT_EQ(LIBC_NAMESPACE::sigprocmask(SIG_SETMASK, &oldset, nullptr), 0);
}