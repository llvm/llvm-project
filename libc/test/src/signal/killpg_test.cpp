//===-- Unittests for killpg ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/signal/killpg.h"

#include "hdr/errno_macros.h"
#include "hdr/signal_macros.h"
#include "src/__support/OSUtil/syscall.h"
#include "test/UnitTest/ErrnoSetterMatcher.h"
#include "test/UnitTest/Test.h"

#include <sys/syscall.h>

using LIBC_NAMESPACE::testing::ErrnoSetterMatcher::Fails;
using LIBC_NAMESPACE::testing::ErrnoSetterMatcher::Succeeds;

TEST(LlvmLibcKillpgTest, InvalidPgrp) {
  // Negative process group ID must fail with EINVAL.
  ASSERT_THAT(LIBC_NAMESPACE::killpg(-1, 0), Fails(EINVAL));
}

TEST(LlvmLibcKillpgTest, TargetSelf) {
  // pgrp = 0 represents the caller's process group.
  ASSERT_THAT(LIBC_NAMESPACE::killpg(0, 0), Succeeds(0));

  // Sending signal 0 to our own process group should succeed.
  pid_t pgrp = LIBC_NAMESPACE::syscall_impl<pid_t>(SYS_getpgrp);
  ASSERT_THAT(LIBC_NAMESPACE::killpg(pgrp, 0), Succeeds(0));
}