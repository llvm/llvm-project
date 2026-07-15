//===-- Unittests for sigaction -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "hdr/errno_macros.h"
#include "hdr/signal_macros.h"
#include "src/signal/raise.h"
#include "src/signal/sigaction.h"
#include "test/UnitTest/ErrnoSetterMatcher.h"
#include "test/UnitTest/Test.h"

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

struct ResethandTest {
  inline static int counter = 0;
  static void handler(int) { counter++; }
};

// Verify that SA_RESETHAND resets the signal handler to SIG_DFL after one
// execution.
TEST(LlvmLibcSigaction, Resethand) {
  ResethandTest::counter = 0;
  struct sigaction action;
  EXPECT_THAT(LIBC_NAMESPACE::sigaction(SIGUSR1, nullptr, &action), Succeeds());
  action.sa_handler = ResethandTest::handler;
  action.sa_flags = SA_RESETHAND;
  EXPECT_THAT(LIBC_NAMESPACE::sigaction(SIGUSR1, &action, nullptr), Succeeds());

  LIBC_NAMESPACE::raise(SIGUSR1);
  EXPECT_EQ(ResethandTest::counter, 1);

  // The handler should have been reset to SIG_DFL.
  struct sigaction old_action;
  EXPECT_THAT(LIBC_NAMESPACE::sigaction(SIGUSR1, nullptr, &old_action),
              Succeeds());
  EXPECT_EQ(old_action.sa_handler, SIG_DFL);
}

struct NodeferTest {
  inline static int max_depth = 0;
  inline static int depth = 0;
  static void handler(int sig) {
    depth++;
    if (depth > max_depth)
      max_depth = depth;
    if (depth < 3)
      LIBC_NAMESPACE::raise(sig);
    depth--;
  }
};

struct WithoutNodeferTest {
  inline static int calls = 0;
  inline static int max_depth = 0;
  inline static int depth = 0;
  static void handler(int sig) {
    depth++;
    calls++;
    if (depth > max_depth)
      max_depth = depth;
    if (calls == 1)
      LIBC_NAMESPACE::raise(sig);
    depth--;
  }
};

// Verify that SA_NODEFER allows recursive/nested signal delivery.
TEST(LlvmLibcSigaction, Nodefer) {
  NodeferTest::max_depth = 0;
  NodeferTest::depth = 0;
  struct sigaction action;
  EXPECT_THAT(LIBC_NAMESPACE::sigaction(SIGUSR1, nullptr, &action), Succeeds());
  action.sa_handler = NodeferTest::handler;
  action.sa_flags = SA_NODEFER;
  EXPECT_THAT(LIBC_NAMESPACE::sigaction(SIGUSR1, &action, nullptr), Succeeds());

  LIBC_NAMESPACE::raise(SIGUSR1);
  EXPECT_EQ(NodeferTest::max_depth, 3);

  // Test without SA_NODEFER (default behavior: signal is blocked)
  WithoutNodeferTest::max_depth = 0;
  WithoutNodeferTest::depth = 0;
  WithoutNodeferTest::calls = 0;
  action.sa_handler = WithoutNodeferTest::handler;
  action.sa_flags = 0;
  EXPECT_THAT(LIBC_NAMESPACE::sigaction(SIGUSR1, &action, nullptr), Succeeds());

  LIBC_NAMESPACE::raise(SIGUSR1);
  EXPECT_EQ(WithoutNodeferTest::max_depth, 1);
  EXPECT_EQ(WithoutNodeferTest::calls, 2);
}

struct SiginfoTest {
  inline static bool received = false;
  inline static int signo = 0;
  static void handler(int, siginfo_t *info, void *) {
    received = true;
    if (!info)
      return;
    signo = info->si_signo;
  }
};

// Verify that SA_SIGINFO invokes the sa_sigaction handler with siginfo_t.
TEST(LlvmLibcSigaction, Siginfo) {
  SiginfoTest::received = false;
  SiginfoTest::signo = 0;
  struct sigaction action;
  EXPECT_THAT(LIBC_NAMESPACE::sigaction(SIGUSR1, nullptr, &action), Succeeds());
  action.sa_sigaction = SiginfoTest::handler;
  action.sa_flags = SA_SIGINFO;
  EXPECT_THAT(LIBC_NAMESPACE::sigaction(SIGUSR1, &action, nullptr), Succeeds());

  LIBC_NAMESPACE::raise(SIGUSR1);
  EXPECT_TRUE(SiginfoTest::received);
  EXPECT_EQ(SiginfoTest::signo, SIGUSR1);
}
