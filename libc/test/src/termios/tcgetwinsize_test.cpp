//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unit tests for tcgetwinsize.
///
//===----------------------------------------------------------------------===//

#include "hdr/fcntl_macros.h"
#include "hdr/types/struct_winsize.h"
#include "src/__support/libc_errno.h"
#include "src/fcntl/open.h"
#include "src/termios/tcgetwinsize.h"
#include "src/unistd/close.h"
#include "src/unistd/pipe.h"
#include "test/UnitTest/ErrnoCheckingTest.h"
#include "test/UnitTest/ErrnoSetterMatcher.h"
#include "test/UnitTest/Test.h"

using LlvmLibcTcGetWinSizeTest = LIBC_NAMESPACE::testing::ErrnoCheckingTest;
using namespace LIBC_NAMESPACE::testing::ErrnoSetterMatcher;

TEST_F(LlvmLibcTcGetWinSizeTest, InvalidFileDescriptor) {
  struct winsize ws;
  ASSERT_THAT(LIBC_NAMESPACE::tcgetwinsize(-1, &ws), Fails(EBADF));
}

TEST_F(LlvmLibcTcGetWinSizeTest, NonTerminalFileDescriptor) {
  int pipefd[2];
  ASSERT_THAT(LIBC_NAMESPACE::pipe(pipefd), Succeeds(0));

  struct winsize ws;
  ASSERT_THAT(LIBC_NAMESPACE::tcgetwinsize(pipefd[0], &ws), Fails(ENOTTY));
  ASSERT_THAT(LIBC_NAMESPACE::tcgetwinsize(pipefd[1], &ws), Fails(ENOTTY));

  ASSERT_THAT(LIBC_NAMESPACE::close(pipefd[0]), Succeeds(0));
  ASSERT_THAT(LIBC_NAMESPACE::close(pipefd[1]), Succeeds(0));
}

TEST_F(LlvmLibcTcGetWinSizeTest, TerminalSmokeTest) {
  int fd = LIBC_NAMESPACE::open("/dev/tty", O_RDONLY);
  if (fd < 0) {
    // When /dev/tty is not available, gracefully skip the test.
    libc_errno = 0;
    return;
  }
  ASSERT_ERRNO_SUCCESS();

  constexpr unsigned short SENTINEL_VAL = 0xFFFF;
  struct winsize ws = {SENTINEL_VAL, SENTINEL_VAL, SENTINEL_VAL, SENTINEL_VAL};
  int ret = LIBC_NAMESPACE::tcgetwinsize(fd, &ws);
  if (ret < 0) {
    ASSERT_ERRNO_EQ(ENOTTY);
  } else {
    ASSERT_ERRNO_SUCCESS();
    EXPECT_NE(ws.ws_row, SENTINEL_VAL);
    EXPECT_NE(ws.ws_col, SENTINEL_VAL);
  }
  ASSERT_THAT(LIBC_NAMESPACE::close(fd), Succeeds(0));
}
