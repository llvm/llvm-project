//===-- Unittests for sigfillset ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/signal/raise.h"
#include "src/signal/sigfillset.h"
#include "src/signal/sigprocmask.h"

#include "test/UnitTest/ErrnoSetterMatcher.h"
#include "test/UnitTest/Test.h"

#include <signal.h>

TEST(LlvmLibcSigfillset, Invalid) {
  using LIBC_NAMESPACE::testing::ErrnoSetterMatcher::Fails;
  EXPECT_THAT(LIBC_NAMESPACE::sigfillset(nullptr), Fails(EINVAL));
}

TEST(LlvmLibcSigfillset, BlocksAll) {
  using LIBC_NAMESPACE::testing::ErrnoSetterMatcher::Succeeds;
  sigset_t set;
  EXPECT_THAT(LIBC_NAMESPACE::sigfillset(&set), Succeeds());
  EXPECT_THAT(LIBC_NAMESPACE::sigprocmask(SIG_SETMASK, &set, nullptr),
              Succeeds());
  EXPECT_EXITS([] { LIBC_NAMESPACE::raise(SIGUSR1); }, 0);
}
