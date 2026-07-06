//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unittests for sockatmark.
///
//===----------------------------------------------------------------------===//

#include "hdr/sys_socket_macros.h" // For AF_UNIX and SOCK_DGRAM
#include "src/__support/CPP/scope.h"
#include "src/sys/socket/sockatmark.h"
#include "src/sys/socket/socketpair.h"
#include "src/unistd/close.h"
#include "src/unistd/pipe.h"
#include "test/UnitTest/ErrnoCheckingTest.h"
#include "test/UnitTest/ErrnoSetterMatcher.h"

using LIBC_NAMESPACE::testing::ErrnoSetterMatcher::Fails;
using LIBC_NAMESPACE::testing::ErrnoSetterMatcher::Succeeds;
using LlvmLibcSockatmarkTest = LIBC_NAMESPACE::testing::ErrnoCheckingTest;
using LIBC_NAMESPACE::cpp::scope_exit;

TEST_F(LlvmLibcSockatmarkTest, SocketpairReturnsFalse) {
  int sockpair[2] = {-1, -1};
  ASSERT_THAT(LIBC_NAMESPACE::socketpair(AF_UNIX, SOCK_DGRAM, 0, sockpair),
              Succeeds(0));
  scope_exit close_sockpair([&] {
    ASSERT_THAT(LIBC_NAMESPACE::close(sockpair[0]), Succeeds(0));
    ASSERT_THAT(LIBC_NAMESPACE::close(sockpair[1]), Succeeds(0));
  });

  ASSERT_THAT(LIBC_NAMESPACE::sockatmark(sockpair[0]), Succeeds(0));
  ASSERT_THAT(LIBC_NAMESPACE::sockatmark(sockpair[1]), Succeeds(0));
}

TEST_F(LlvmLibcSockatmarkTest, InvalidFdFails) {
  ASSERT_THAT(LIBC_NAMESPACE::sockatmark(-1), Fails(EBADF));
}

TEST_F(LlvmLibcSockatmarkTest, NonSocketFdFails) {
  int pipefd[2] = {-1, -1};
  ASSERT_THAT(LIBC_NAMESPACE::pipe(pipefd), Succeeds(0));
  scope_exit close_pipe([&] {
    ASSERT_THAT(LIBC_NAMESPACE::close(pipefd[0]), Succeeds(0));
    ASSERT_THAT(LIBC_NAMESPACE::close(pipefd[1]), Succeeds(0));
  });

  ASSERT_THAT(LIBC_NAMESPACE::sockatmark(pipefd[0]), Fails(ENOTTY));
}
