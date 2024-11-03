//===-- Unittests for sigfillset ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "include/errno.h"
#include "include/signal.h"
#include "src/signal/raise.h"
#include "src/signal/sigfillset.h"
#include "src/signal/sigprocmask.h"

#include "test/ErrnoSetterMatcher.h"
#include "test/UnitTest/Test.h"

TEST(LlvmLibcSigfillset, Invalid) {
  using __llvm_libc::testing::ErrnoSetterMatcher::Fails;
  EXPECT_THAT(__llvm_libc::sigfillset(nullptr), Fails(EINVAL));
}

TEST(LlvmLibcSigfillset, BlocksAll) {
  using __llvm_libc::testing::ErrnoSetterMatcher::Succeeds;
  sigset_t set;
  EXPECT_THAT(__llvm_libc::sigfillset(&set), Succeeds());
  EXPECT_THAT(__llvm_libc::sigprocmask(SIG_SETMASK, &set, nullptr), Succeeds());
  EXPECT_EXITS([] { __llvm_libc::raise(SIGUSR1); }, 0);
}
