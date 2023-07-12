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

using __llvm_libc::testing::ErrnoSetterMatcher::Succeeds;

TEST(LlvmLibcKillTest, TargetSelf) {
  pid_t parent_pid = __llvm_libc::syscall_impl(SYS_getpid);
  ASSERT_THAT(__llvm_libc::kill(parent_pid, 0), Succeeds(0));

  EXPECT_DEATH(
      [] {
        pid_t child_pid = __llvm_libc::syscall_impl(SYS_getpid);
        __llvm_libc::kill(child_pid, SIGKILL);
      },
      WITH_SIGNAL(SIGKILL));
}
