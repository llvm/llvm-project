//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unittests for posix_close.
///
//===----------------------------------------------------------------------===//

#include "hdr/errno_macros.h"
#include "hdr/unistd_macros.h"
#include "src/unistd/close.h"
#include "src/unistd/pipe.h"
#include "src/unistd/posix_close.h"
#include "src/unistd/read.h"
#include "test/UnitTest/ErrnoCheckingTest.h"
#include "test/UnitTest/ErrnoSetterMatcher.h"
#include "test/UnitTest/Test.h"

using namespace LIBC_NAMESPACE::testing::ErrnoSetterMatcher;
using LlvmLibcPosixCloseTest = LIBC_NAMESPACE::testing::ErrnoCheckingTest;

TEST_F(LlvmLibcPosixCloseTest, InvalidFlag) {
  int pipefd[2];
  ASSERT_THAT(LIBC_NAMESPACE::pipe(pipefd), Succeeds(0));

  // If flag is invalid, posix_close() may fail with EINVAL, but shall otherwise
  // behave as if flag had been 0 and close fd.
  EXPECT_THAT(LIBC_NAMESPACE::posix_close(pipefd[0], ~0), Fails<int>(EINVAL));
  EXPECT_THAT(LIBC_NAMESPACE::close(pipefd[1]), Succeeds(0));

  // The descriptor pipefd[0] should now be closed.
  char buf[1];
  EXPECT_THAT(LIBC_NAMESPACE::read(pipefd[0], buf, 1), Fails<ssize_t>(EBADF));
}

TEST_F(LlvmLibcPosixCloseTest, InvalidFd) {
  // Valid flags (0 and POSIX_CLOSE_RESTART) with invalid fd should fail with
  // EBADF.
  EXPECT_THAT(LIBC_NAMESPACE::posix_close(-1, 0), Fails<int>(EBADF));
  EXPECT_THAT(LIBC_NAMESPACE::posix_close(-1, POSIX_CLOSE_RESTART),
              Fails<int>(EBADF));
}

TEST_F(LlvmLibcPosixCloseTest, CloseSuccess) {
  int pipefd[2];
  ASSERT_THAT(LIBC_NAMESPACE::pipe(pipefd), Succeeds(0));

  EXPECT_THAT(LIBC_NAMESPACE::posix_close(pipefd[0], 0), Succeeds(0));
  EXPECT_THAT(LIBC_NAMESPACE::close(pipefd[1]), Succeeds(0));

  // Subsequent operation on closed fd should fail with EBADF.
  char buf[1];
  EXPECT_THAT(LIBC_NAMESPACE::read(pipefd[0], buf, 1), Fails<ssize_t>(EBADF));
}

TEST_F(LlvmLibcPosixCloseTest, CloseWithRestartFlag) {
  int pipefd[2];
  ASSERT_THAT(LIBC_NAMESPACE::pipe(pipefd), Succeeds(0));

  EXPECT_THAT(LIBC_NAMESPACE::posix_close(pipefd[0], POSIX_CLOSE_RESTART),
              Succeeds(0));
  EXPECT_THAT(LIBC_NAMESPACE::close(pipefd[1]), Succeeds(0));

  // Subsequent operation on closed fd should fail with EBADF.
  char buf[1];
  EXPECT_THAT(LIBC_NAMESPACE::read(pipefd[0], buf, 1), Fails<ssize_t>(EBADF));
}
