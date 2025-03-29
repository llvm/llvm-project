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

#include "test/UnitTest/ErrnoCheckingTest.h"
#include "test/UnitTest/ErrnoSetterMatcher.h"
#include "test/UnitTest/Test.h"

#include <sys/socket.h> // For AF_UNIX and SOCK_DGRAM

using LIBC_NAMESPACE::testing::ErrnoSetterMatcher::Fails;
using LIBC_NAMESPACE::testing::ErrnoSetterMatcher::Succeeds;
using LlvmLibcSendMsgRecvMsgTest = LIBC_NAMESPACE::testing::ErrnoCheckingTest;

TEST_F(LlvmLibcSendMsgRecvMsgTest, SucceedsWithSocketPair) {
  const char TEST_MESSAGE[] = "connection successful";
  const size_t MESSAGE_LEN = sizeof(TEST_MESSAGE);

  int sockpair[2] = {0, 0};

  ASSERT_THAT(LIBC_NAMESPACE::socketpair(AF_UNIX, SOCK_STREAM, 0, sockpair),
              Succeeds(0));

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

  ASSERT_THAT(LIBC_NAMESPACE::sendmsg(sockpair[0], &send_message, 0),
              Succeeds(static_cast<ssize_t>(MESSAGE_LEN)));

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

  ASSERT_THAT(LIBC_NAMESPACE::recvmsg(sockpair[1], &recv_message, 0),
              Succeeds(static_cast<ssize_t>(MESSAGE_LEN)));

  ASSERT_STREQ(buffer, TEST_MESSAGE);

  // close both ends of the socket
  ASSERT_THAT(LIBC_NAMESPACE::close(sockpair[0]), Succeeds(0));
  ASSERT_THAT(LIBC_NAMESPACE::close(sockpair[1]), Succeeds(0));
}

TEST_F(LlvmLibcSendMsgRecvMsgTest, SendFails) {
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

  ASSERT_THAT(LIBC_NAMESPACE::sendmsg(-1, &send_message, 0), Fails(EBADF));
}

TEST_F(LlvmLibcSendMsgRecvMsgTest, RecvFails) {
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

  ASSERT_THAT(LIBC_NAMESPACE::recvmsg(-1, &recv_message, 0), Fails(EBADF));
}
