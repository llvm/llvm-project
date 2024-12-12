//===-- Unittests for socketpair ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/sys/socket/socketpair.h"

#include "src/unistd/close.h"

#include "src/errno/libc_errno.h"
#include "test/UnitTest/Test.h"

#include <sys/socket.h> // For AF_UNIX and SOCK_DGRAM

TEST(LlvmLibcSocketPairTest, LocalSocket) {
  int sockpair[2] = {-1, -1};
  int result = LIBC_NAMESPACE::socketpair(AF_UNIX, SOCK_DGRAM, 0, sockpair);
  ASSERT_EQ(result, 0);
  ASSERT_ERRNO_SUCCESS();

  ASSERT_GE(sockpair[0], 0);
  ASSERT_GE(sockpair[1], 0);

  LIBC_NAMESPACE::close(sockpair[0]);
  LIBC_NAMESPACE::close(sockpair[1]);
  ASSERT_ERRNO_SUCCESS();
}

TEST(LlvmLibcSocketPairTest, SocketFails) {
  int sockpair[2] = {-1, -1};
  int result = LIBC_NAMESPACE::socketpair(-1, -1, -1, sockpair);
  ASSERT_EQ(result, -1);
  ASSERT_ERRNO_FAILURE();
}
