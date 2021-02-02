//===-- Unittests for sigaction -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "include/errno.h"
#define __LLVM_LIBC_INTERNAL_SIGACTION
#include "include/signal.h"
#include "src/signal/raise.h"
#include "src/signal/sigaction.h"

#include "test/ErrnoSetterMatcher.h"
#include "utils/UnitTest/Test.h"

using __llvm_libc::testing::ErrnoSetterMatcher::Fails;
using __llvm_libc::testing::ErrnoSetterMatcher::Succeeds;

TEST(LlvmLibcSigaction, Invalid) {
  // -1 is a much larger signal that NSIG, so this should fail.
  EXPECT_THAT(__llvm_libc::sigaction(-1, nullptr, nullptr), Fails(EINVAL));
}

// SIGKILL cannot have its action changed, but it can be examined.
TEST(LlvmLibcSigaction, Sigkill) {
  struct __sigaction action;
  EXPECT_THAT(__llvm_libc::sigaction(SIGKILL, nullptr, &action), Succeeds());
  EXPECT_THAT(__llvm_libc::sigaction(SIGKILL, &action, nullptr), Fails(EINVAL));
}

static int sigusr1Count;
static bool correctSignal;

TEST(LlvmLibcSigaction, CustomAction) {
  // Zero this incase tests get run multiple times in the future.
  sigusr1Count = 0;

  struct __sigaction action;
  EXPECT_THAT(__llvm_libc::sigaction(SIGUSR1, nullptr, &action), Succeeds());

  action.sa_handler = +[](int signal) {
    correctSignal = signal == SIGUSR1;
    sigusr1Count++;
  };
  EXPECT_THAT(__llvm_libc::sigaction(SIGUSR1, &action, nullptr), Succeeds());

  __llvm_libc::raise(SIGUSR1);
  EXPECT_EQ(sigusr1Count, 1);
  EXPECT_TRUE(correctSignal);

  action.sa_handler = SIG_DFL;
  EXPECT_THAT(__llvm_libc::sigaction(SIGUSR1, &action, nullptr), Succeeds());

  EXPECT_DEATH([] { __llvm_libc::raise(SIGUSR1); }, WITH_SIGNAL(SIGUSR1));
}

TEST(LlvmLibcSigaction, Ignore) {
  struct __sigaction action;
  EXPECT_THAT(__llvm_libc::sigaction(SIGUSR1, nullptr, &action), Succeeds());
  action.sa_handler = SIG_IGN;
  EXPECT_THAT(__llvm_libc::sigaction(SIGUSR1, &action, nullptr), Succeeds());

  EXPECT_EXITS([] { __llvm_libc::raise(SIGUSR1); }, 0);
}
