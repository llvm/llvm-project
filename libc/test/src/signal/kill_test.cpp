//===-- Unittests for kill -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/signal/kill.h"

#include "src/__support/OSUtil/syscall.h" // For internal syscall function.
#include "test/UnitTest/ErrnoSetterMatcher.h"
#include "test/UnitTest/Test.h"

#include <signal.h>
#include <sys/syscall.h> // For syscall numbers.

using LIBC_NAMESPACE::testing::ErrnoSetterMatcher::Succeeds;

TEST(LlvmLibcKillTest, TargetSelf) {
  pid_t parent_pid = LIBC_NAMESPACE::syscall_impl<pid_t>(SYS_getpid);
  ASSERT_THAT(LIBC_NAMESPACE::kill(parent_pid, 0), Succeeds(0));

  EXPECT_DEATH(
      [] {
        pid_t child_pid = LIBC_NAMESPACE::syscall_impl<pid_t>(SYS_getpid);
        LIBC_NAMESPACE::kill(child_pid, SIGKILL);
      },
      WITH_SIGNAL(SIGKILL));
}
