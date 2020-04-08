//===-- Unittests for sigprocmask -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "include/errno.h"
#include "include/signal.h"
#include "src/errno/llvmlibc_errno.h"
#include "src/signal/raise.h"
#include "src/signal/sigaddset.h"
#include "src/signal/sigemptyset.h"
#include "src/signal/sigprocmask.h"

#include "utils/UnitTest/ErrnoSetterMatcher.h"
#include "utils/UnitTest/Test.h"

class SignalTest : public __llvm_libc::testing::Test {
  sigset_t oldSet;

public:
  void SetUp() override { __llvm_libc::sigprocmask(0, nullptr, &oldSet); }

  void TearDown() override {
    __llvm_libc::sigprocmask(SIG_SETMASK, &oldSet, nullptr);
  }
};

using __llvm_libc::testing::ErrnoSetterMatcher::Fails;
using __llvm_libc::testing::ErrnoSetterMatcher::Succeeds;

// This tests for invalid input.
TEST_F(SignalTest, SigprocmaskInvalid) {
  llvmlibc_errno = 0;

  sigset_t valid;
  // 17 and -4 are out of the range for sigprocmask's how paramater.
  EXPECT_THAT(__llvm_libc::sigprocmask(17, &valid, nullptr), Fails(EINVAL));
  EXPECT_THAT(__llvm_libc::sigprocmask(-4, &valid, nullptr), Fails(EINVAL));

  // This pointer is out of this processes address range.
  sigset_t *invalid = reinterpret_cast<sigset_t *>(-1);
  EXPECT_THAT(__llvm_libc::sigprocmask(SIG_SETMASK, invalid, nullptr),
              Fails(EFAULT));
  EXPECT_THAT(__llvm_libc::sigprocmask(-4, nullptr, invalid), Fails(EFAULT));
}

// This tests that when nothing is blocked, a process gets killed and alse tests
// that when signals are blocked they are not delivered to the process.
TEST_F(SignalTest, BlockUnblock) {
  sigset_t sigset;
  EXPECT_EQ(__llvm_libc::sigemptyset(&sigset), 0);
  EXPECT_EQ(__llvm_libc::sigprocmask(SIG_SETMASK, &sigset, nullptr), 0);
  EXPECT_DEATH([] { __llvm_libc::raise(SIGUSR1); }, SIGUSR1);
  EXPECT_EQ(__llvm_libc::sigaddset(&sigset, SIGUSR1), 0);
  EXPECT_EQ(__llvm_libc::sigprocmask(SIG_SETMASK, &sigset, nullptr), 0);
  EXPECT_EXITS([] { __llvm_libc::raise(SIGUSR1); }, 0);
}
