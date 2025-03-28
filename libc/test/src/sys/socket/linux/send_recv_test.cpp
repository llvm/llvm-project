//===-- Unittests for send/recv -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/sys/socket/recv.h"
#include "src/sys/socket/send.h"
#include "src/sys/socket/socketpair.h"

#include "src/unistd/close.h"

#include "test/UnitTest/ErrnoCheckingTest.h"
#include "test/UnitTest/ErrnoSetterMatcher.h"
#include "test/UnitTest/Test.h"

#include <sys/socket.h> // For AF_UNIX and SOCK_DGRAM

using LIBC_NAMESPACE::testing::ErrnoSetterMatcher::Fails;
using LIBC_NAMESPACE::testing::ErrnoSetterMatcher::Succeeds;
using LlvmLibcSendRecvTest = LIBC_NAMESPACE::testing::ErrnoCheckingTest;

TEST_F(LlvmLibcSendRecvTest, SucceedsWithSocketPair) {
  const char TEST_MESSAGE[] = "connection successful";
  const size_t MESSAGE_LEN = sizeof(TEST_MESSAGE);

  int sockpair[2] = {0, 0};

  ASSERT_THAT(LIBC_NAMESPACE::socketpair(AF_UNIX, SOCK_STREAM, 0, sockpair),
              Succeeds(0));

  ASSERT_THAT(LIBC_NAMESPACE::send(sockpair[0], TEST_MESSAGE, MESSAGE_LEN, 0),
              Succeeds(static_cast<ssize_t>(MESSAGE_LEN)));

  char buffer[256];

  ASSERT_THAT(LIBC_NAMESPACE::recv(sockpair[1], buffer, sizeof(buffer), 0),
              Succeeds(static_cast<ssize_t>(MESSAGE_LEN)));

  ASSERT_STREQ(buffer, TEST_MESSAGE);

  // close both ends of the socket
  ASSERT_THAT(LIBC_NAMESPACE::close(sockpair[0]), Succeeds(0));
  ASSERT_THAT(LIBC_NAMESPACE::close(sockpair[1]), Succeeds(0));
}

TEST_F(LlvmLibcSendRecvTest, SendFails) {
  const char TEST_MESSAGE[] = "connection terminated";
  const size_t MESSAGE_LEN = sizeof(TEST_MESSAGE);

  ASSERT_THAT(LIBC_NAMESPACE::send(-1, TEST_MESSAGE, MESSAGE_LEN, 0),
              Fails(EBADF));
}

TEST_F(LlvmLibcSendRecvTest, RecvFails) {
  char buffer[256];

  ASSERT_THAT(LIBC_NAMESPACE::recv(-1, buffer, sizeof(buffer), 0),
              Fails(EBADF));
}
