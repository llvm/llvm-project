//===-- Unittests for sigaction -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/signal/raise.h"
#include "src/signal/sigaction.h"

#include "test/UnitTest/ErrnoSetterMatcher.h"
#include "test/UnitTest/Test.h"

#include <errno.h>
#include <signal.h>

using LIBC_NAMESPACE::testing::ErrnoSetterMatcher::Fails;
using LIBC_NAMESPACE::testing::ErrnoSetterMatcher::Succeeds;

TEST(LlvmLibcSigaction, Invalid) {
  // -1 is a much larger signal that NSIG, so this should fail.
  EXPECT_THAT(LIBC_NAMESPACE::sigaction(-1, nullptr, nullptr), Fails(EINVAL));
}

// SIGKILL cannot have its action changed, but it can be examined.
TEST(LlvmLibcSigaction, Sigkill) {
  struct sigaction action;
  EXPECT_THAT(LIBC_NAMESPACE::sigaction(SIGKILL, nullptr, &action), Succeeds());
  EXPECT_THAT(LIBC_NAMESPACE::sigaction(SIGKILL, &action, nullptr),
              Fails(EINVAL));
}

static int sigusr1Count;
static bool correctSignal;

TEST(LlvmLibcSigaction, CustomAction) {
  // Zero this incase tests get run multiple times in the future.
  sigusr1Count = 0;

  struct sigaction action;
  EXPECT_THAT(LIBC_NAMESPACE::sigaction(SIGUSR1, nullptr, &action), Succeeds());

  action.sa_handler = +[](int signal) {
    correctSignal = signal == SIGUSR1;
    sigusr1Count++;
  };
  EXPECT_THAT(LIBC_NAMESPACE::sigaction(SIGUSR1, &action, nullptr), Succeeds());

  LIBC_NAMESPACE::raise(SIGUSR1);
  EXPECT_EQ(sigusr1Count, 1);
  EXPECT_TRUE(correctSignal);

  action.sa_handler = SIG_DFL;
  EXPECT_THAT(LIBC_NAMESPACE::sigaction(SIGUSR1, &action, nullptr), Succeeds());

  EXPECT_DEATH([] { LIBC_NAMESPACE::raise(SIGUSR1); }, WITH_SIGNAL(SIGUSR1));
}

TEST(LlvmLibcSigaction, Ignore) {
  struct sigaction action;
  EXPECT_THAT(LIBC_NAMESPACE::sigaction(SIGUSR1, nullptr, &action), Succeeds());
  action.sa_handler = SIG_IGN;
  EXPECT_THAT(LIBC_NAMESPACE::sigaction(SIGUSR1, &action, nullptr), Succeeds());

  EXPECT_EXITS([] { LIBC_NAMESPACE::raise(SIGUSR1); }, 0);
}
