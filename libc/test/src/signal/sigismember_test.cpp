//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unittests for sigismember.
///
//===----------------------------------------------------------------------===//

#include "src/signal/sigismember.h"

#include "hdr/signal_macros.h"
#include "hdr/types/sigset_t.h"
#include "src/signal/sigaddset.h"
#include "src/signal/sigdelset.h"
#include "src/signal/sigemptyset.h"
#include "src/signal/sigfillset.h"
#include "test/UnitTest/ErrnoSetterMatcher.h"
#include "test/UnitTest/Test.h"

TEST(LlvmLibcSigismember, Invalid) {
  using LIBC_NAMESPACE::testing::ErrnoSetterMatcher::Fails;
  EXPECT_THAT(LIBC_NAMESPACE::sigismember(nullptr, SIGSEGV), Fails(EINVAL));

  sigset_t sigset;
  EXPECT_THAT(LIBC_NAMESPACE::sigismember(&sigset, -1), Fails(EINVAL));
  EXPECT_THAT(LIBC_NAMESPACE::sigismember(&sigset, 0), Fails(EINVAL));

#if defined(__APPLE__)
  EXPECT_THAT(LIBC_NAMESPACE::sigismember(&sigset, NSIG), Fails(EINVAL));
#else
  constexpr int bitsInSigsetT = 8 * sizeof(sigset_t);
  EXPECT_THAT(LIBC_NAMESPACE::sigismember(&sigset, bitsInSigsetT + 1),
              Fails(EINVAL));
#endif
}

TEST(LlvmLibcSigismember, BasicOperations) {
  using LIBC_NAMESPACE::testing::ErrnoSetterMatcher::Succeeds;
  sigset_t sigset;

  EXPECT_THAT(LIBC_NAMESPACE::sigemptyset(&sigset), Succeeds());
  EXPECT_EQ(LIBC_NAMESPACE::sigismember(&sigset, SIGUSR1), 0);
  EXPECT_EQ(LIBC_NAMESPACE::sigismember(&sigset, SIGUSR2), 0);
  EXPECT_EQ(LIBC_NAMESPACE::sigismember(&sigset, SIGINT), 0);

  EXPECT_THAT(LIBC_NAMESPACE::sigaddset(&sigset, SIGUSR1), Succeeds());
  EXPECT_EQ(LIBC_NAMESPACE::sigismember(&sigset, SIGUSR1), 1);
  EXPECT_EQ(LIBC_NAMESPACE::sigismember(&sigset, SIGUSR2), 0);

  EXPECT_THAT(LIBC_NAMESPACE::sigaddset(&sigset, SIGINT), Succeeds());
  EXPECT_EQ(LIBC_NAMESPACE::sigismember(&sigset, SIGUSR1), 1);
  EXPECT_EQ(LIBC_NAMESPACE::sigismember(&sigset, SIGINT), 1);
  EXPECT_EQ(LIBC_NAMESPACE::sigismember(&sigset, SIGUSR2), 0);

  EXPECT_THAT(LIBC_NAMESPACE::sigdelset(&sigset, SIGUSR1), Succeeds());
  EXPECT_EQ(LIBC_NAMESPACE::sigismember(&sigset, SIGUSR1), 0);
  EXPECT_EQ(LIBC_NAMESPACE::sigismember(&sigset, SIGINT), 1);

  EXPECT_THAT(LIBC_NAMESPACE::sigfillset(&sigset), Succeeds());
  EXPECT_EQ(LIBC_NAMESPACE::sigismember(&sigset, SIGUSR1), 1);
  EXPECT_EQ(LIBC_NAMESPACE::sigismember(&sigset, SIGUSR2), 1);
  EXPECT_EQ(LIBC_NAMESPACE::sigismember(&sigset, SIGINT), 1);

  EXPECT_THAT(LIBC_NAMESPACE::sigdelset(&sigset, SIGUSR2), Succeeds());
  EXPECT_EQ(LIBC_NAMESPACE::sigismember(&sigset, SIGUSR1), 1);
  EXPECT_EQ(LIBC_NAMESPACE::sigismember(&sigset, SIGUSR2), 0);
}
