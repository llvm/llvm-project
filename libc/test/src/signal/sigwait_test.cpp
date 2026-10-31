//===-- Unittests for sigwait ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/signal/sigwait.h"

#include "hdr/errno_macros.h"
#include "hdr/signal_macros.h"
#include "hdr/types/sigset_t.h"
#include "src/signal/raise.h"
#include "src/signal/sigaddset.h"
#include "src/signal/sigemptyset.h"
#include "src/signal/sigprocmask.h"
#include "test/UnitTest/Test.h"

TEST(LlvmLibcSigwaitTest, WaitPendingSignal) {
  // Block SIGUSR1 so raising it leaves it pending without terminating.
  sigset_t mask;
  ASSERT_EQ(LIBC_NAMESPACE::sigemptyset(&mask), 0);
  ASSERT_EQ(LIBC_NAMESPACE::sigaddset(&mask, SIGUSR1), 0);

  sigset_t oldset;
  ASSERT_EQ(LIBC_NAMESPACE::sigprocmask(SIG_BLOCK, &mask, &oldset), 0);

  // Raise SIGUSR1. It is now queued/pending for this thread.
  ASSERT_EQ(LIBC_NAMESPACE::raise(SIGUSR1), 0);

  // sigwait should immediately catch and dequeue SIGUSR1.
  int sig = 0;
  ASSERT_EQ(LIBC_NAMESPACE::sigwait(&mask, &sig), 0);
  ASSERT_EQ(sig, SIGUSR1);

  // Restore the original signal mask.
  ASSERT_EQ(LIBC_NAMESPACE::sigprocmask(SIG_SETMASK, &oldset, nullptr), 0);
}