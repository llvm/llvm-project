//===-- Unittests for socketpair ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/sys/socket/socketpair.h"

#include "src/unistd/close.h"

#include "test/UnitTest/ErrnoCheckingTest.h"
#include "test/UnitTest/ErrnoSetterMatcher.h"
#include "test/UnitTest/Test.h"

#include <sys/socket.h> // For AF_UNIX and SOCK_DGRAM

using LIBC_NAMESPACE::testing::ErrnoSetterMatcher::Fails;
using LIBC_NAMESPACE::testing::ErrnoSetterMatcher::Succeeds;
using LlvmLibcSocketPairTest = LIBC_NAMESPACE::testing::ErrnoCheckingTest;

TEST_F(LlvmLibcSocketPairTest, LocalSocket) {
  int sockpair[2] = {-1, -1};
  ASSERT_THAT(LIBC_NAMESPACE::socketpair(AF_UNIX, SOCK_DGRAM, 0, sockpair),
              Succeeds(0));

  ASSERT_GE(sockpair[0], 0);
  ASSERT_GE(sockpair[1], 0);

  ASSERT_THAT(LIBC_NAMESPACE::close(sockpair[0]), Succeeds(0));
  ASSERT_THAT(LIBC_NAMESPACE::close(sockpair[1]), Succeeds(0));
}

TEST_F(LlvmLibcSocketPairTest, SocketFails) {
  int sockpair[2] = {-1, -1};
  ASSERT_THAT(LIBC_NAMESPACE::socketpair(-1, -1, -1, sockpair), Fails(EINVAL));
}
