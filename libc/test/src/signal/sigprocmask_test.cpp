//===-- Unittests for sigprocmask -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/libc_errno.h"
#include "src/signal/raise.h"
#include "src/signal/sigaddset.h"
#include "src/signal/sigemptyset.h"
#include "src/signal/sigprocmask.h"

#include "test/UnitTest/ErrnoSetterMatcher.h"
#include "test/UnitTest/Test.h"

#include <signal.h>

class LlvmLibcSignalTest : public LIBC_NAMESPACE::testing::Test {
  sigset_t oldSet;

public:
  void SetUp() override { LIBC_NAMESPACE::sigprocmask(0, nullptr, &oldSet); }

  void TearDown() override {
    LIBC_NAMESPACE::sigprocmask(SIG_SETMASK, &oldSet, nullptr);
  }
};

using LIBC_NAMESPACE::testing::ErrnoSetterMatcher::Fails;
using LIBC_NAMESPACE::testing::ErrnoSetterMatcher::Succeeds;

// This tests for invalid input.
TEST_F(LlvmLibcSignalTest, SigprocmaskInvalid) {
  libc_errno = 0;

  sigset_t valid;
  // 17 and -4 are out of the range for sigprocmask's how paramater.
  EXPECT_THAT(LIBC_NAMESPACE::sigprocmask(17, &valid, nullptr), Fails(EINVAL));
  EXPECT_THAT(LIBC_NAMESPACE::sigprocmask(-4, &valid, nullptr), Fails(EINVAL));

  // This pointer is out of this processes address range.
  sigset_t *invalid = reinterpret_cast<sigset_t *>(-1);
  EXPECT_THAT(LIBC_NAMESPACE::sigprocmask(SIG_SETMASK, invalid, nullptr),
              Fails(EFAULT));
  EXPECT_THAT(LIBC_NAMESPACE::sigprocmask(-4, nullptr, invalid), Fails(EFAULT));
}

// This tests that when nothing is blocked, a process gets killed and alse tests
// that when signals are blocked they are not delivered to the process.
TEST_F(LlvmLibcSignalTest, BlockUnblock) {
  sigset_t sigset;
  EXPECT_EQ(LIBC_NAMESPACE::sigemptyset(&sigset), 0);
  EXPECT_EQ(LIBC_NAMESPACE::sigprocmask(SIG_SETMASK, &sigset, nullptr), 0);
  EXPECT_DEATH([] { LIBC_NAMESPACE::raise(SIGUSR1); }, WITH_SIGNAL(SIGUSR1));
  EXPECT_EQ(LIBC_NAMESPACE::sigaddset(&sigset, SIGUSR1), 0);
  EXPECT_EQ(LIBC_NAMESPACE::sigprocmask(SIG_SETMASK, &sigset, nullptr), 0);
  EXPECT_EXITS([] { LIBC_NAMESPACE::raise(SIGUSR1); }, 0);
}
