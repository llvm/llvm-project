//===-- Unittests for sendto/recvfrom -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/sys/socket/recvfrom.h"
#include "src/sys/socket/sendto.h"
#include "src/sys/socket/socketpair.h"

#include "src/unistd/close.h"

#include "src/errno/libc_errno.h"
#include "test/UnitTest/Test.h"

#include <sys/socket.h> // For AF_UNIX and SOCK_DGRAM

TEST(LlvmLibcSendToRecvFromTest, SucceedsWithSocketPair) {
  const char TEST_MESSAGE[] = "connection successful";
  const size_t MESSAGE_LEN = sizeof(TEST_MESSAGE);

  int sockpair[2] = {0, 0};

  int result = LIBC_NAMESPACE::socketpair(AF_UNIX, SOCK_STREAM, 0, sockpair);
  ASSERT_EQ(result, 0);
  ASSERT_ERRNO_SUCCESS();

  ssize_t send_result = LIBC_NAMESPACE::sendto(sockpair[0], TEST_MESSAGE,
                                               MESSAGE_LEN, 0, nullptr, 0);
  EXPECT_EQ(send_result, static_cast<ssize_t>(MESSAGE_LEN));
  ASSERT_ERRNO_SUCCESS();

  char buffer[256];

  ssize_t recv_result = LIBC_NAMESPACE::recvfrom(sockpair[1], buffer,
                                                 sizeof(buffer), 0, nullptr, 0);
  ASSERT_EQ(recv_result, static_cast<ssize_t>(MESSAGE_LEN));
  ASSERT_ERRNO_SUCCESS();

  ASSERT_STREQ(buffer, TEST_MESSAGE);

  // close both ends of the socket
  result = LIBC_NAMESPACE::close(sockpair[0]);
  ASSERT_EQ(result, 0);
  ASSERT_ERRNO_SUCCESS();

  result = LIBC_NAMESPACE::close(sockpair[1]);
  ASSERT_EQ(result, 0);
  ASSERT_ERRNO_SUCCESS();
}

TEST(LlvmLibcSendToRecvFromTest, SendToFails) {
  const char TEST_MESSAGE[] = "connection terminated";
  const size_t MESSAGE_LEN = sizeof(TEST_MESSAGE);

  ssize_t send_result =
      LIBC_NAMESPACE::sendto(-1, TEST_MESSAGE, MESSAGE_LEN, 0, nullptr, 0);
  EXPECT_EQ(send_result, ssize_t(-1));
  ASSERT_ERRNO_FAILURE();

  LIBC_NAMESPACE::libc_errno = 0; // reset errno to avoid test ordering issues.
}

TEST(LlvmLibcSendToRecvFromTest, RecvFromFails) {
  char buffer[256];

  ssize_t recv_result =
      LIBC_NAMESPACE::recvfrom(-1, buffer, sizeof(buffer), 0, nullptr, 0);
  ASSERT_EQ(recv_result, ssize_t(-1));
  ASSERT_ERRNO_FAILURE();

  LIBC_NAMESPACE::libc_errno = 0; // reset errno to avoid test ordering issues.
}
