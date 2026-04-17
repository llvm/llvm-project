//===-- Unittests for ucontext routines -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/ucontext/getcontext.h"
#include "src/ucontext/setcontext.h"

#include "src/signal/sigaddset.h"
#include "src/signal/sigemptyset.h"
#include "src/signal/sigprocmask.h"

#include "test/UnitTest/Test.h"

#include "include/llvm-libc-macros/signal-macros.h"

namespace LIBC_NAMESPACE {

static bool is_signal_set(const sigset_t *set, int signum) {
  // TODO: Replace this with sigismember once it is implemented.
  // NSIG is 64, sigset_t is an array of unsigned long.
  // Signum is 1-indexed.
  int word = (signum - 1) / (sizeof(unsigned long) * 8);
  int bit = (signum - 1) % (sizeof(unsigned long) * 8);
  return (set->__signals[word] & (1UL << bit)) != 0;
}

TEST(LlvmLibcUcontextTest, BasicStubTest) {
  static volatile int jumped = 0;
  ucontext_t ctx;
  ASSERT_EQ(getcontext(&ctx), 0);
  if (!jumped) {
    jumped = 1;
    setcontext(&ctx);
    ASSERT_TRUE(false && "setcontext should not return on success");
  }
}

TEST(LlvmLibcUcontextTest, SignalMaskTest) {
  sigset_t set, old_set;
  sigemptyset(&set);
  sigaddset(&set, SIGUSR1);

  // Set mask to [SIGUSR1] using sigprocmask
  sigprocmask(SIG_SETMASK, &set, &old_set);

  ucontext_t ctx;
  getcontext(&ctx);

  // Verify that getcontext captured the mask
  ASSERT_TRUE(is_signal_set(&ctx.uc_sigmask, SIGUSR1));
  ASSERT_FALSE(is_signal_set(&ctx.uc_sigmask, SIGUSR2));

  sigset_t new_set;
  static volatile int mask_jumped = 0;
  if (mask_jumped == 0) {
    mask_jumped = 1;
    sigemptyset(&new_set);
    sigaddset(&new_set, SIGUSR2);
    sigprocmask(SIG_SETMASK, &new_set, nullptr);

    setcontext(&ctx);
  }

  // Check current mask
  sigset_t current;
  sigprocmask(SIG_BLOCK, nullptr, &current);

  // Restore original mask for clean state
  sigprocmask(SIG_SETMASK, &old_set, nullptr);

  ASSERT_TRUE(is_signal_set(&current, SIGUSR1));
  ASSERT_FALSE(is_signal_set(&current, SIGUSR2));
}

} // namespace LIBC_NAMESPACE
