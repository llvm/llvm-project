//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Integration test for lockf.
///
//===----------------------------------------------------------------------===//

#include "hdr/errno_macros.h"
#include "hdr/fcntl_macros.h"
#include "hdr/sys_stat_macros.h"
#include "hdr/sys_wait_macros.h"
#include "hdr/unistd_macros.h"
#include "src/__support/CPP/scope.h"
#include "src/fcntl/open.h"
#include "src/stdlib/exit.h"
#include "src/sys/wait/waitpid.h"
#include "src/unistd/close.h"
#include "src/unistd/fork.h"
#include "src/unistd/lockf.h"
#include "src/unistd/unlink.h"

#include "test/IntegrationTest/test.h"

#include <errno.h>

TEST_MAIN([[maybe_unused]] int argc, [[maybe_unused]] char **argv,
          [[maybe_unused]] char **envp) {
  constexpr const char TEST_FILE[] = "lockf_integration.test";

  int fd = LIBC_NAMESPACE::open(TEST_FILE, O_CREAT | O_TRUNC | O_RDWR, S_IRWXU);
  ASSERT_TRUE(fd > 0);

  LIBC_NAMESPACE::cpp::scope_exit cleanup([&] {
    ASSERT_EQ(LIBC_NAMESPACE::close(fd), 0);
    ASSERT_EQ(LIBC_NAMESPACE::unlink(TEST_FILE), 0);
  });

  // Acquire an exclusive lock on the file in the parent process.
  ASSERT_EQ(LIBC_NAMESPACE::lockf(fd, F_LOCK, 0), 0);

  pid_t pid = LIBC_NAMESPACE::fork();
  ASSERT_TRUE(pid >= 0);

  if (pid == 0) {
    // Child process:
    int child_fd = LIBC_NAMESPACE::open(TEST_FILE, O_RDWR, 0);
    if (child_fd < 0)
      LIBC_NAMESPACE::exit(1);

    // F_TEST should fail because the section is locked by another process
    // (parent).
    if (LIBC_NAMESPACE::lockf(child_fd, F_TEST, 0) == 0)
      LIBC_NAMESPACE::exit(2);
    if (errno != EACCES && errno != EAGAIN)
      LIBC_NAMESPACE::exit(3);

    // F_TLOCK should fail because non-blocking lock cannot be acquired.
    if (LIBC_NAMESPACE::lockf(child_fd, F_TLOCK, 0) == 0)
      LIBC_NAMESPACE::exit(4);
    if (errno != EACCES && errno != EAGAIN)
      LIBC_NAMESPACE::exit(5);

    LIBC_NAMESPACE::close(child_fd);
    LIBC_NAMESPACE::exit(0);
  }

  // Parent process waits for child.
  int status = 0;
  pid_t cpid = LIBC_NAMESPACE::waitpid(pid, &status, 0);
  ASSERT_EQ(cpid, pid);
  ASSERT_TRUE(WIFEXITED(status));
  ASSERT_EQ(WEXITSTATUS(status), 0);

  return 0;
}
