//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unit tests for tcsetwinsize.
///
//===----------------------------------------------------------------------===//

#include "hdr/fcntl_macros.h"
#include "hdr/types/struct_winsize.h"
#include "src/__support/CPP/scope.h"
#include "src/__support/libc_errno.h"
#include "src/fcntl/open.h"
#include "src/termios/tcsetwinsize.h"
#include "src/unistd/close.h"
#include "src/unistd/pipe.h"
#include "test/UnitTest/ErrnoCheckingTest.h"
#include "test/UnitTest/ErrnoSetterMatcher.h"
#include "test/UnitTest/Test.h"

using LlvmLibcTcSetWinSizeTest = LIBC_NAMESPACE::testing::ErrnoCheckingTest;
using namespace LIBC_NAMESPACE::testing::ErrnoSetterMatcher;

TEST_F(LlvmLibcTcSetWinSizeTest, InvalidFileDescriptor) {
  constexpr unsigned short TEST_ROWS = 24;
  constexpr unsigned short TEST_COLS = 80;
  struct winsize ws = {TEST_ROWS, TEST_COLS, 0, 0};
  ASSERT_THAT(LIBC_NAMESPACE::tcsetwinsize(-1, &ws), Fails(EBADF));
}

TEST_F(LlvmLibcTcSetWinSizeTest, NonTerminalFileDescriptor) {
  int pipefd[2];
  ASSERT_THAT(LIBC_NAMESPACE::pipe(pipefd), Succeeds(0));
  LIBC_NAMESPACE::cpp::scope_exit close_pipe([&] {
    ASSERT_THAT(LIBC_NAMESPACE::close(pipefd[0]), Succeeds(0));
    ASSERT_THAT(LIBC_NAMESPACE::close(pipefd[1]), Succeeds(0));
  });

  constexpr unsigned short TEST_ROWS = 24;
  constexpr unsigned short TEST_COLS = 80;
  struct winsize ws = {TEST_ROWS, TEST_COLS, 0, 0};
  ASSERT_THAT(LIBC_NAMESPACE::tcsetwinsize(pipefd[0], &ws), Fails(ENOTTY));
  ASSERT_THAT(LIBC_NAMESPACE::tcsetwinsize(pipefd[1], &ws), Fails(ENOTTY));
}

TEST_F(LlvmLibcTcSetWinSizeTest, TerminalSmokeTest) {
  // Use a pseudo-terminal master rather than /dev/tty so running tests in an
  // interactive terminal does not resize the user's window.
  int fd = LIBC_NAMESPACE::open("/dev/ptmx", O_RDWR);
  if (fd < 0) {
    // When /dev/ptmx is not available, gracefully skip the test.
    libc_errno = 0;
    return;
  }
  ASSERT_ERRNO_SUCCESS();
  LIBC_NAMESPACE::cpp::scope_exit close_fd(
      [&] { ASSERT_THAT(LIBC_NAMESPACE::close(fd), Succeeds(0)); });

  constexpr unsigned short TEST_ROWS = 30;
  constexpr unsigned short TEST_COLS = 100;
  struct winsize ws = {TEST_ROWS, TEST_COLS, 0, 0};
  int ret = LIBC_NAMESPACE::tcsetwinsize(fd, &ws);
  if (ret < 0)
    ASSERT_ERRNO_EQ(ENOTTY);
  else
    ASSERT_ERRNO_SUCCESS();
}

TEST_F(LlvmLibcTcSetWinSizeTest, NullPointer) {
  int fd = LIBC_NAMESPACE::open("/dev/ptmx", O_RDWR);
  if (fd < 0) {
    // When /dev/ptmx is not available, gracefully skip the test.
    libc_errno = 0;
    return;
  }
  ASSERT_ERRNO_SUCCESS();
  LIBC_NAMESPACE::cpp::scope_exit close_fd(
      [&] { ASSERT_THAT(LIBC_NAMESPACE::close(fd), Succeeds(0)); });

  constexpr unsigned short TEST_ROWS = 24;
  constexpr unsigned short TEST_COLS = 80;
  struct winsize ws = {TEST_ROWS, TEST_COLS, 0, 0};
  int ret = LIBC_NAMESPACE::tcsetwinsize(fd, &ws);
  if (ret < 0) {
    ASSERT_ERRNO_EQ(ENOTTY);
    return;
  }

  ASSERT_THAT(LIBC_NAMESPACE::tcsetwinsize(fd, nullptr), Fails(EFAULT));
}
