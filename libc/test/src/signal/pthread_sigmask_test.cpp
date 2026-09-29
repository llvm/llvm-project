//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unittests for pthread_sigmask.
///
//===----------------------------------------------------------------------===//

#include "hdr/signal_macros.h"
#include "hdr/types/sigset_t.h"
#include "src/signal/pthread_sigmask.h"
#include "src/signal/raise.h"
#include "src/signal/sigaddset.h"
#include "src/signal/sigemptyset.h"
#include "test/UnitTest/ErrnoCheckingTest.h"
#include "test/UnitTest/Test.h"

class LlvmLibcPthreadSigmaskTest
    : public LIBC_NAMESPACE::testing::ErrnoCheckingTest {
  sigset_t old_set;

public:
  void SetUp() override {
    ErrnoCheckingTest::SetUp();
    LIBC_NAMESPACE::pthread_sigmask(0, nullptr, &old_set);
  }

  void TearDown() override {
    LIBC_NAMESPACE::pthread_sigmask(SIG_SETMASK, &old_set, nullptr);
    ErrnoCheckingTest::TearDown();
  }
};

// This tests for invalid input.
TEST_F(LlvmLibcPthreadSigmaskTest, PthreadSigmaskInvalid) {
  sigset_t valid;
  // 17 and -4 are out of the range for pthread_sigmask's how parameter.
  EXPECT_EQ(LIBC_NAMESPACE::pthread_sigmask(17, &valid, nullptr), EINVAL);
  EXPECT_EQ(LIBC_NAMESPACE::pthread_sigmask(-4, &valid, nullptr), EINVAL);

  // This pointer is out of this process's address range.
  sigset_t *invalid = reinterpret_cast<sigset_t *>(-1);
  EXPECT_EQ(LIBC_NAMESPACE::pthread_sigmask(SIG_SETMASK, invalid, nullptr),
            EFAULT);
  EXPECT_EQ(LIBC_NAMESPACE::pthread_sigmask(SIG_SETMASK, nullptr, invalid),
            EFAULT);
}

// This tests that when nothing is blocked, a process gets killed and also tests
// that when signals are blocked they are not delivered to the process.
TEST_F(LlvmLibcPthreadSigmaskTest, BlockUnblock) {
  sigset_t sigset;
  EXPECT_EQ(LIBC_NAMESPACE::sigemptyset(&sigset), 0);
  EXPECT_EQ(LIBC_NAMESPACE::pthread_sigmask(SIG_SETMASK, &sigset, nullptr), 0);
  EXPECT_DEATH([] { LIBC_NAMESPACE::raise(SIGUSR1); }, WITH_SIGNAL(SIGUSR1));
  EXPECT_EQ(LIBC_NAMESPACE::sigaddset(&sigset, SIGUSR1), 0);
  EXPECT_EQ(LIBC_NAMESPACE::pthread_sigmask(SIG_SETMASK, &sigset, nullptr), 0);
  EXPECT_EXITS([] { LIBC_NAMESPACE::raise(SIGUSR1); }, 0);
}
