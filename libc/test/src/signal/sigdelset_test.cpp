//===-- Unittests for sigdelset -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/signal/raise.h"
#include "src/signal/sigdelset.h"
#include "src/signal/sigfillset.h"
#include "src/signal/sigprocmask.h"

#include "test/UnitTest/ErrnoSetterMatcher.h"
#include "test/UnitTest/Test.h"

#include <errno.h>
#include <signal.h>

TEST(LlvmLibcSigdelset, Invalid) {
  using LIBC_NAMESPACE::testing::ErrnoSetterMatcher::Fails;
  // Invalid set.
  EXPECT_THAT(LIBC_NAMESPACE::sigdelset(nullptr, SIGUSR1), Fails(EINVAL));

  sigset_t set;
  // Valid set, invalid signum.
  EXPECT_THAT(LIBC_NAMESPACE::sigdelset(&set, -1), Fails(EINVAL));
}

TEST(LlvmLibcSigdelset, UnblockOne) {
  using LIBC_NAMESPACE::testing::ErrnoSetterMatcher::Succeeds;
  sigset_t set;
  EXPECT_THAT(LIBC_NAMESPACE::sigfillset(&set), Succeeds());
  EXPECT_THAT(LIBC_NAMESPACE::sigdelset(&set, SIGUSR1), Succeeds());
  EXPECT_THAT(LIBC_NAMESPACE::sigprocmask(SIG_SETMASK, &set, nullptr),
              Succeeds());
  EXPECT_DEATH([] { LIBC_NAMESPACE::raise(SIGUSR1); }, WITH_SIGNAL(SIGUSR1));
}
