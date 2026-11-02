//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unit tests for popen and pclose.
///
//===----------------------------------------------------------------------===//

#include "src/stdio/popen.h"

#include "hdr/errno_macros.h"
#include "hdr/fcntl_macros.h"
#include "hdr/sys_wait_macros.h"
#include "hdr/types/FILE.h"
#include "src/fcntl/fcntl.h"
#include "src/stdio/fgets.h"
#include "src/stdio/fileno.h"
#include "src/stdio/fputs.h"
#include "src/stdio/pclose.h"
#include "src/stdio/snprintf.h"
#include "src/sys/wait/wait.h"
#include "test/UnitTest/ErrnoCheckingTest.h"
#include "test/UnitTest/ErrnoSetterMatcher.h"
#include "test/UnitTest/Test.h"

using LIBC_NAMESPACE::testing::ErrnoSetterMatcher::Fails;
using LlvmLibcPopenTest = LIBC_NAMESPACE::testing::ErrnoCheckingTest;

TEST_F(LlvmLibcPopenTest, ReadMode) {
  ::FILE *stream = LIBC_NAMESPACE::popen("echo hello", "r");
  ASSERT_ERRNO_SUCCESS();
  ASSERT_NE(stream, nullptr);

  char buf[16];
  ASSERT_NE(LIBC_NAMESPACE::fgets(buf, sizeof(buf), stream), nullptr);
  ASSERT_STREQ(buf, "hello\n");

  int status = LIBC_NAMESPACE::pclose(stream);
  ASSERT_ERRNO_SUCCESS();
  ASSERT_TRUE(WIFEXITED(status));
  ASSERT_EQ(WEXITSTATUS(status), 0);
}

TEST_F(LlvmLibcPopenTest, WriteMode) {
  // The command exits with 0 only if it reads "hello" from its stdin.
  ::FILE *stream = LIBC_NAMESPACE::popen("read x; test \"$x\" = hello", "w");
  ASSERT_ERRNO_SUCCESS();
  ASSERT_NE(stream, nullptr);

  ASSERT_GE(LIBC_NAMESPACE::fputs("hello\n", stream), 0);

  int status = LIBC_NAMESPACE::pclose(stream);
  ASSERT_ERRNO_SUCCESS();
  ASSERT_TRUE(WIFEXITED(status));
  ASSERT_EQ(WEXITSTATUS(status), 0);
}

TEST_F(LlvmLibcPopenTest, ExitStatus) {
  ::FILE *stream = LIBC_NAMESPACE::popen("exit 42", "r");
  ASSERT_ERRNO_SUCCESS();
  ASSERT_NE(stream, nullptr);

  int status = LIBC_NAMESPACE::pclose(stream);
  ASSERT_ERRNO_SUCCESS();
  ASSERT_TRUE(WIFEXITED(status));
  ASSERT_EQ(WEXITSTATUS(status), 42);
}

TEST_F(LlvmLibcPopenTest, InvalidMode) {
  ::FILE *stream = LIBC_NAMESPACE::popen("true", "x");
  ASSERT_ERRNO_EQ(EINVAL);
  ASSERT_EQ(stream, nullptr);
}

TEST_F(LlvmLibcPopenTest, PreviousStreamsClosedInChild) {
  // Keep a popen stream open while a second command is started.
  ::FILE *first = LIBC_NAMESPACE::popen("cat >/dev/null", "w");
  ASSERT_ERRNO_SUCCESS();
  ASSERT_NE(first, nullptr);
  int first_fd = LIBC_NAMESPACE::fileno(first);
  ASSERT_GE(first_fd, 0);

  // The second command reports whether the first stream's fd is open in its
  // own process. Redirecting to a closed fd fails with EBADF.
  char command[128];
  ASSERT_GT(LIBC_NAMESPACE::snprintf(
                command, sizeof(command),
                "{ true >&%d; } 2>/dev/null && echo open || echo closed",
                first_fd),
            0);
  ::FILE *second = LIBC_NAMESPACE::popen(command, "r");
  ASSERT_ERRNO_SUCCESS();
  ASSERT_NE(second, nullptr);

  char buf[16];
  ASSERT_NE(LIBC_NAMESPACE::fgets(buf, sizeof(buf), second), nullptr);
  ASSERT_STREQ(buf, "closed\n");

  int status = LIBC_NAMESPACE::pclose(second);
  ASSERT_ERRNO_SUCCESS();
  ASSERT_TRUE(WIFEXITED(status));
  ASSERT_EQ(WEXITSTATUS(status), 0);

  status = LIBC_NAMESPACE::pclose(first);
  ASSERT_ERRNO_SUCCESS();
  ASSERT_TRUE(WIFEXITED(status));
  ASSERT_EQ(WEXITSTATUS(status), 0);
}

TEST_F(LlvmLibcPopenTest, StatusUnavailable) {
  ::FILE *stream = LIBC_NAMESPACE::popen("true", "r");
  ASSERT_ERRNO_SUCCESS();
  ASSERT_NE(stream, nullptr);

  // Reap the child before pclose can, which makes its status unavailable.
  int status = 0;
  ASSERT_GT(LIBC_NAMESPACE::wait(&status), 0);
  ASSERT_TRUE(WIFEXITED(status));

  ASSERT_THAT(LIBC_NAMESPACE::pclose(stream), Fails(ECHILD));
}

TEST_F(LlvmLibcPopenTest, CloseOnExecMode) {
  ::FILE *stream = LIBC_NAMESPACE::popen("true", "re");
  ASSERT_ERRNO_SUCCESS();
  ASSERT_NE(stream, nullptr);

  int fd_flags = LIBC_NAMESPACE::fcntl(LIBC_NAMESPACE::fileno(stream), F_GETFD);
  ASSERT_ERRNO_SUCCESS();
  ASSERT_GE(fd_flags, 0);
  ASSERT_NE(fd_flags & FD_CLOEXEC, 0);

  int status = LIBC_NAMESPACE::pclose(stream);
  ASSERT_ERRNO_SUCCESS();
  ASSERT_TRUE(WIFEXITED(status));
  ASSERT_EQ(WEXITSTATUS(status), 0);
}

TEST_F(LlvmLibcPopenTest, NoCloseOnExecByDefault) {
  ::FILE *stream = LIBC_NAMESPACE::popen("true", "r");
  ASSERT_ERRNO_SUCCESS();
  ASSERT_NE(stream, nullptr);

  int fd_flags = LIBC_NAMESPACE::fcntl(LIBC_NAMESPACE::fileno(stream), F_GETFD);
  ASSERT_ERRNO_SUCCESS();
  ASSERT_GE(fd_flags, 0);
  ASSERT_EQ(fd_flags & FD_CLOEXEC, 0);

  int status = LIBC_NAMESPACE::pclose(stream);
  ASSERT_ERRNO_SUCCESS();
  ASSERT_TRUE(WIFEXITED(status));
  ASSERT_EQ(WEXITSTATUS(status), 0);
}
