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
#include "src/__support/CPP/scope.h"
#include "src/fcntl/fcntl.h"
#include "src/stdio/fclose.h"
#include "src/stdio/fgets.h"
#include "src/stdio/fileno.h"
#include "src/stdio/fopen.h"
#include "src/stdio/fputs.h"
#include "src/stdio/freopen.h"
#include "src/stdio/pclose.h"
#include "src/stdio/remove.h"
#include "src/stdio/snprintf.h"
#include "src/sys/wait/wait.h"
#include "test/UnitTest/ErrnoCheckingTest.h"
#include "test/UnitTest/ErrnoSetterMatcher.h"
#include "test/UnitTest/Test.h"

using LIBC_NAMESPACE::cpp::scope_exit;
using LIBC_NAMESPACE::testing::ErrnoSetterMatcher::Fails;
using LIBC_NAMESPACE::testing::ErrnoSetterMatcher::Succeeds;
using LlvmLibcPopenTest = LIBC_NAMESPACE::testing::ErrnoCheckingTest;

TEST_F(LlvmLibcPopenTest, ReadMode) {
  ::FILE *stream = LIBC_NAMESPACE::popen("echo hello", "r");
  ASSERT_ERRNO_SUCCESS();
  ASSERT_NE(stream, nullptr);
  scope_exit close_stream([&] { (void)LIBC_NAMESPACE::pclose(stream); });

  char buf[16];
  ASSERT_NE(LIBC_NAMESPACE::fgets(buf, sizeof(buf), stream), nullptr);
  ASSERT_STREQ(buf, "hello\n");

  close_stream.release();
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
  scope_exit close_stream([&] { (void)LIBC_NAMESPACE::pclose(stream); });

  ASSERT_GE(LIBC_NAMESPACE::fputs("hello\n", stream), 0);

  close_stream.release();
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
  scope_exit close_first([&] { (void)LIBC_NAMESPACE::pclose(first); });
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
  scope_exit close_second([&] { (void)LIBC_NAMESPACE::pclose(second); });

  char buf[16];
  ASSERT_NE(LIBC_NAMESPACE::fgets(buf, sizeof(buf), second), nullptr);
  ASSERT_STREQ(buf, "closed\n");

  close_second.release();
  int status = LIBC_NAMESPACE::pclose(second);
  ASSERT_ERRNO_SUCCESS();
  ASSERT_TRUE(WIFEXITED(status));
  ASSERT_EQ(WEXITSTATUS(status), 0);

  close_first.release();
  status = LIBC_NAMESPACE::pclose(first);
  ASSERT_ERRNO_SUCCESS();
  ASSERT_TRUE(WIFEXITED(status));
  ASSERT_EQ(WEXITSTATUS(status), 0);
}

TEST_F(LlvmLibcPopenTest, StatusUnavailable) {
  ::FILE *stream = LIBC_NAMESPACE::popen("true", "r");
  ASSERT_ERRNO_SUCCESS();
  ASSERT_NE(stream, nullptr);
  scope_exit close_stream([&] { (void)LIBC_NAMESPACE::pclose(stream); });

  // Reap the child before pclose can, which makes its status unavailable.
  int status = 0;
  ASSERT_GT(LIBC_NAMESPACE::wait(&status), 0);
  ASSERT_TRUE(WIFEXITED(status));

  close_stream.release();
  ASSERT_THAT(LIBC_NAMESPACE::pclose(stream), Fails(ECHILD));
}

TEST_F(LlvmLibcPopenTest, CloseOnExecMode) {
  ::FILE *stream = LIBC_NAMESPACE::popen("true", "re");
  ASSERT_ERRNO_SUCCESS();
  ASSERT_NE(stream, nullptr);
  scope_exit close_stream([&] { (void)LIBC_NAMESPACE::pclose(stream); });

  int fd_flags = LIBC_NAMESPACE::fcntl(LIBC_NAMESPACE::fileno(stream), F_GETFD);
  ASSERT_ERRNO_SUCCESS();
  ASSERT_GE(fd_flags, 0);
  ASSERT_NE(fd_flags & FD_CLOEXEC, 0);

  close_stream.release();
  int status = LIBC_NAMESPACE::pclose(stream);
  ASSERT_ERRNO_SUCCESS();
  ASSERT_TRUE(WIFEXITED(status));
  ASSERT_EQ(WEXITSTATUS(status), 0);
}

TEST_F(LlvmLibcPopenTest, NoCloseOnExecByDefault) {
  ::FILE *stream = LIBC_NAMESPACE::popen("true", "r");
  ASSERT_ERRNO_SUCCESS();
  ASSERT_NE(stream, nullptr);
  scope_exit close_stream([&] { (void)LIBC_NAMESPACE::pclose(stream); });

  int fd_flags = LIBC_NAMESPACE::fcntl(LIBC_NAMESPACE::fileno(stream), F_GETFD);
  ASSERT_ERRNO_SUCCESS();
  ASSERT_GE(fd_flags, 0);
  ASSERT_EQ(fd_flags & FD_CLOEXEC, 0);

  close_stream.release();
  int status = LIBC_NAMESPACE::pclose(stream);
  ASSERT_ERRNO_SUCCESS();
  ASSERT_TRUE(WIFEXITED(status));
  ASSERT_EQ(WEXITSTATUS(status), 0);
}

TEST_F(LlvmLibcPopenTest, FreopenNullPath) {
  ::FILE *stream = LIBC_NAMESPACE::popen("echo reopened", "r");
  ASSERT_ERRNO_SUCCESS();
  ASSERT_NE(stream, nullptr);
  scope_exit close_stream([&] { (void)LIBC_NAMESPACE::pclose(stream); });

  ASSERT_THAT(LIBC_NAMESPACE::freopen(nullptr, "invalid", stream),
              Fails(EINVAL, static_cast<void *>(nullptr)));
  ASSERT_THAT(LIBC_NAMESPACE::freopen(nullptr, "w", stream),
              Fails(EBADF, static_cast<void *>(nullptr)));
  ASSERT_THAT(LIBC_NAMESPACE::freopen(nullptr, "rb", stream),
              Succeeds(static_cast<void *>(stream)));

  char buf[16] = {0};
  ASSERT_NE(LIBC_NAMESPACE::fgets(buf, sizeof(buf), stream), nullptr);
  EXPECT_STREQ(buf, "reopened\n");

  close_stream.release();
  int status = LIBC_NAMESPACE::pclose(stream);
  ASSERT_ERRNO_SUCCESS();
  ASSERT_TRUE(WIFEXITED(status));
  ASSERT_EQ(WEXITSTATUS(status), 0);
}

TEST_F(LlvmLibcPopenTest, FreopenPath) {
  const auto FILENAME = libc_make_test_file_path("popen_freopen_path.test");
  const auto NON_EXISTENT =
      libc_make_test_file_path("popen_freopen_missing.test");

  ::FILE *stream = LIBC_NAMESPACE::popen("true", "r");
  ASSERT_ERRNO_SUCCESS();
  ASSERT_NE(stream, nullptr);
  scope_exit close_stream([&] { (void)LIBC_NAMESPACE::pclose(stream); });

  // A failed freopen detaches the pipe fd and sets fd = -1, but a subsequent
  // freopen onto a valid path must still succeed.
  ASSERT_THAT(LIBC_NAMESPACE::freopen(NON_EXISTENT, "r", stream),
              Fails(ENOENT, static_cast<void *>(nullptr)));
  ASSERT_THAT(LIBC_NAMESPACE::freopen(FILENAME, "w+", stream),
              Succeeds(static_cast<void *>(stream)));
  scope_exit remove_file([&] { (void)LIBC_NAMESPACE::remove(FILENAME); });

  // Spawning another popen child while |stream| is open must not affect or
  // inherit the reopened file stream.
  ::FILE *other = LIBC_NAMESPACE::popen("true", "r");
  ASSERT_NE(other, nullptr);
  ASSERT_GE(LIBC_NAMESPACE::pclose(other), 0);
  ASSERT_ERRNO_SUCCESS();

  ASSERT_GE(LIBC_NAMESPACE::fputs("reopened popen\n", stream), 0);

  close_stream.release();
  int status = LIBC_NAMESPACE::pclose(stream);
  ASSERT_ERRNO_SUCCESS();
  ASSERT_TRUE(WIFEXITED(status));
  ASSERT_EQ(WEXITSTATUS(status), 0);

  ::FILE *verify = LIBC_NAMESPACE::fopen(FILENAME, "r");
  ASSERT_NE(verify, nullptr);
  scope_exit close_verify([&] { (void)LIBC_NAMESPACE::fclose(verify); });
  char buf[32] = {0};
  ASSERT_NE(LIBC_NAMESPACE::fgets(buf, sizeof(buf), verify), nullptr);
  EXPECT_STREQ(buf, "reopened popen\n");
  close_verify.release();
  ASSERT_THAT(LIBC_NAMESPACE::fclose(verify), Succeeds(0));
  remove_file.release();
  ASSERT_THAT(LIBC_NAMESPACE::remove(FILENAME), Succeeds(0));
}
