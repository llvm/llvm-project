//===-- Unittests for sigaddset -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/signal/sigaddset.h"

#include "test/UnitTest/ErrnoSetterMatcher.h"
#include "test/UnitTest/Test.h"

#include <signal.h>

// This tests invalid inputs and ensures errno is properly set.
TEST(LlvmLibcSignalTest, SigaddsetInvalid) {
  using LIBC_NAMESPACE::testing::ErrnoSetterMatcher::Fails;
  using LIBC_NAMESPACE::testing::ErrnoSetterMatcher::Succeeds;
  EXPECT_THAT(LIBC_NAMESPACE::sigaddset(nullptr, SIGSEGV), Fails(EINVAL));

  sigset_t sigset;
  EXPECT_THAT(LIBC_NAMESPACE::sigaddset(&sigset, -1), Fails(EINVAL));

  // This doesn't use NSIG because LIBC_NAMESPACE::sigaddset error checking is
  // against sizeof(sigset_t) not NSIG.
  constexpr int bitsInSigsetT = 8 * sizeof(sigset_t);

  EXPECT_THAT(LIBC_NAMESPACE::sigaddset(&sigset, bitsInSigsetT + 1),
              Fails(EINVAL));
  EXPECT_THAT(LIBC_NAMESPACE::sigaddset(&sigset, 0), Fails(EINVAL));
  EXPECT_THAT(LIBC_NAMESPACE::sigaddset(&sigset, bitsInSigsetT), Succeeds());
}
