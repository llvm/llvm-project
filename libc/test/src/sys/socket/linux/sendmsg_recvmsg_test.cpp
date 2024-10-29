//===-- Unittests for sendmsg/recvmsg -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/sys/socket/recvmsg.h"
#include "src/sys/socket/sendmsg.h"
#include "src/sys/socket/socketpair.h"

#include "src/unistd/close.h"

#include "src/errno/libc_errno.h"
#include "test/UnitTest/Test.h"

#include <sys/socket.h> // For AF_UNIX and SOCK_DGRAM

TEST(LlvmLibcSendMsgRecvMsgTest, SucceedsWithSocketPair) {
  const char TEST_MESSAGE[] = "connection successful";
  const size_t MESSAGE_LEN = sizeof(TEST_MESSAGE);

  int sockpair[2] = {0, 0};

  int result = LIBC_NAMESPACE::socketpair(AF_UNIX, SOCK_STREAM, 0, sockpair);
  ASSERT_EQ(result, 0);
  ASSERT_ERRNO_SUCCESS();

  iovec send_msg_text;
  send_msg_text.iov_base =
      reinterpret_cast<void *>(const_cast<char *>(TEST_MESSAGE));
  send_msg_text.iov_len = MESSAGE_LEN;

  msghdr send_message;
  send_message.msg_name = nullptr;
  send_message.msg_namelen = 0;
  send_message.msg_iov = &send_msg_text;
  send_message.msg_iovlen = 1;
  send_message.msg_control = nullptr;
  send_message.msg_controllen = 0;
  send_message.msg_flags = 0;

  ssize_t send_result = LIBC_NAMESPACE::sendmsg(sockpair[0], &send_message, 0);
  EXPECT_EQ(send_result, static_cast<ssize_t>(MESSAGE_LEN));
  ASSERT_ERRNO_SUCCESS();

  char buffer[256];

  iovec recv_msg_text;
  recv_msg_text.iov_base = reinterpret_cast<void *>(buffer);
  recv_msg_text.iov_len = sizeof(buffer);

  msghdr recv_message;
  recv_message.msg_name = nullptr;
  recv_message.msg_namelen = 0;
  recv_message.msg_iov = &recv_msg_text;
  recv_message.msg_iovlen = 1;
  recv_message.msg_control = nullptr;
  recv_message.msg_controllen = 0;
  recv_message.msg_flags = 0;

  ssize_t recv_result = LIBC_NAMESPACE::recvmsg(sockpair[1], &recv_message, 0);
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

TEST(LlvmLibcSendMsgRecvMsgTest, SendFails) {
  const char TEST_MESSAGE[] = "connection terminated";
  const size_t MESSAGE_LEN = sizeof(TEST_MESSAGE);

  iovec send_msg_text;
  send_msg_text.iov_base =
      reinterpret_cast<void *>(const_cast<char *>(TEST_MESSAGE));
  send_msg_text.iov_len = MESSAGE_LEN;

  msghdr send_message;
  send_message.msg_name = nullptr;
  send_message.msg_namelen = 0;
  send_message.msg_iov = &send_msg_text;
  send_message.msg_iovlen = 1;
  send_message.msg_control = nullptr;
  send_message.msg_controllen = 0;
  send_message.msg_flags = 0;

  ssize_t send_result = LIBC_NAMESPACE::sendmsg(-1, &send_message, 0);
  EXPECT_EQ(send_result, ssize_t(-1));
  ASSERT_ERRNO_FAILURE();

  LIBC_NAMESPACE::libc_errno = 0; // reset errno to avoid test ordering issues.
}

TEST(LlvmLibcSendMsgRecvMsgTest, RecvFails) {
  char buffer[256];

  iovec recv_msg_text;
  recv_msg_text.iov_base = reinterpret_cast<void *>(buffer);
  recv_msg_text.iov_len = sizeof(buffer);

  msghdr recv_message;
  recv_message.msg_name = nullptr;
  recv_message.msg_namelen = 0;
  recv_message.msg_iov = &recv_msg_text;
  recv_message.msg_iovlen = 1;
  recv_message.msg_control = nullptr;
  recv_message.msg_controllen = 0;
  recv_message.msg_flags = 0;

  ssize_t recv_result = LIBC_NAMESPACE::recvmsg(-1, &recv_message, 0);
  ASSERT_EQ(recv_result, ssize_t(-1));
  ASSERT_ERRNO_FAILURE();

  LIBC_NAMESPACE::libc_errno = 0; // reset errno to avoid test ordering issues.
}
