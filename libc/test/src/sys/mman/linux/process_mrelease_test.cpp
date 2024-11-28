//===-- Unittests for process_mrelease ------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/OSUtil/syscall.h" // For internal syscall function.
#include "src/__support/threads/sleep.h"
#include "src/errno/libc_errno.h"
#include "src/signal/kill.h"
#include "src/stdlib/exit.h"
#include "src/sys/mman/process_mrelease.h"
#include "src/unistd/close.h"
#include "src/unistd/fork.h"
#include "test/UnitTest/ErrnoSetterMatcher.h"
#include "test/UnitTest/LibcTest.h"

#include <sys/syscall.h>

using namespace LIBC_NAMESPACE::testing::ErrnoSetterMatcher;

int pidfd_open(pid_t pid, unsigned int flags) {
  return LIBC_NAMESPACE::syscall_impl(SYS_pidfd_open, pid, flags);
}

TEST(LlvmLibcProcessMReleaseTest, NoError) {
  pid_t child_pid = fork();
  EXPECT_GE(child_pid, 0);

  if (child_pid == 0) {
    // Child process: wait a bit then exit gracefully.
    LIBC_NAMESPACE::sleep_briefly();
    LIBC_NAMESPACE::exit(0);
  } else {
    // Parent process: wait a bit and then kill the child.
    // Give child process some time to start.
    LIBC_NAMESPACE::sleep_briefly();
    int pidfd = pidfd_open(child_pid, 0);
    EXPECT_GE(pidfd, 0);

    // Send SIGKILL to child process
    LIBC_NAMESPACE::kill(child_pid, SIGKILL);

    EXPECT_THAT(LIBC_NAMESPACE::process_mrelease(pidfd, 0), Succeeds());

    LIBC_NAMESPACE::close(pidfd);
  }
}

TEST(LlvmLibcProcessMReleaseTest, ErrorNotKilled) {
  pid_t child_pid = fork();
  EXPECT_GE(child_pid, 0);

  if (child_pid == 0) {
    // Child process: wait a bit then exit gracefully.
    LIBC_NAMESPACE::sleep_briefly();
    LIBC_NAMESPACE::exit(0);
  } else {
    // Give child process some time to start.
    LIBC_NAMESPACE::sleep_briefly();
    int pidfd = pidfd_open(child_pid, 0);
    EXPECT_GE(pidfd, 0);

    EXPECT_THAT(LIBC_NAMESPACE::process_mrelease(pidfd, 0), Fails(EINVAL));

    LIBC_NAMESPACE::close(pidfd);
  }
}

TEST(LlvmLibcProcessMReleaseTest, ErrorNonExistingPidfd) {
  EXPECT_THAT(LIBC_NAMESPACE::process_mrelease(-1, 0), Fails(EBADF));
}
