//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unittests for killpg.
///
//===----------------------------------------------------------------------===//

#include "src/signal/killpg.h"

#include "hdr/errno_macros.h"
#include "hdr/signal_macros.h"
#include "src/unistd/getpgrp.h"
#include "src/unistd/setpgid.h"
#include "test/UnitTest/ErrnoSetterMatcher.h"
#include "test/UnitTest/Test.h"

using LIBC_NAMESPACE::testing::ErrnoSetterMatcher::Fails;
using LIBC_NAMESPACE::testing::ErrnoSetterMatcher::Succeeds;

TEST(LlvmLibcKillpgTest, InvalidPgrp) {
  // Negative process group ID must fail with EINVAL.
  ASSERT_THAT(LIBC_NAMESPACE::killpg(-1, 0), Fails(EINVAL));
}

TEST(LlvmLibcKillpgTest, ZeroPgrpTargetsCallerGroup) {
  // Signal 0 performs an existence check on the caller's process group.
  ASSERT_THAT(LIBC_NAMESPACE::killpg(0, 0), Succeeds(0));
}

TEST(LlvmLibcKillpgTest, ExplicitPgrp) {
  // Signal 0 performs an existence check on our own process group.
  pid_t pgrp = LIBC_NAMESPACE::getpgrp();
  ASSERT_THAT(LIBC_NAMESPACE::killpg(pgrp, 0), Succeeds(0));
}

TEST(LlvmLibcKillpgTest, KillProcessGroup) {
  EXPECT_DEATH(
      [] {
        ASSERT_EQ(LIBC_NAMESPACE::setpgid(0, 0), 0);
        pid_t pgrp = LIBC_NAMESPACE::getpgrp();
        LIBC_NAMESPACE::killpg(pgrp, SIGKILL);
      },
      WITH_SIGNAL(SIGKILL));
}
