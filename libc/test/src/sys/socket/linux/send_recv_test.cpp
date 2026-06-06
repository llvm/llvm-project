//===-- Unittests for send/recv -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "hdr/sys_socket_macros.h"
#include "src/sys/socket/recv.h"
#include "src/sys/socket/send.h"
#include "src/sys/socket/socketpair.h"

#include "src/unistd/close.h"

#include "src/__support/CPP/scope.h"
#include "test/UnitTest/ErrnoCheckingTest.h"
#include "test/UnitTest/ErrnoSetterMatcher.h"
#include "test/UnitTest/Test.h"

using LIBC_NAMESPACE::testing::ErrnoSetterMatcher::Fails;
using LIBC_NAMESPACE::testing::ErrnoSetterMatcher::Succeeds;
using LlvmLibcSendRecvTest = LIBC_NAMESPACE::testing::ErrnoCheckingTest;

TEST_F(LlvmLibcSendRecvTest, SucceedsWithSocketPair) {
  const char TEST_MESSAGE[] = "connection successful";
  const size_t MESSAGE_LEN = sizeof(TEST_MESSAGE);

  int sockpair[2] = {0, 0};

  ASSERT_THAT(LIBC_NAMESPACE::socketpair(AF_UNIX, SOCK_STREAM, 0, sockpair),
              Succeeds(0));
  LIBC_NAMESPACE::cpp::scope_exit close_sockpair([&] {
    ASSERT_THAT(LIBC_NAMESPACE::close(sockpair[0]), Succeeds(0));
    ASSERT_THAT(LIBC_NAMESPACE::close(sockpair[1]), Succeeds(0));
  });

  ASSERT_THAT(LIBC_NAMESPACE::send(sockpair[0], TEST_MESSAGE, MESSAGE_LEN, 0),
              Succeeds(static_cast<ssize_t>(MESSAGE_LEN)));

  char buffer[256];

  ASSERT_THAT(LIBC_NAMESPACE::recv(sockpair[1], buffer, sizeof(buffer), 0),
              Succeeds(static_cast<ssize_t>(MESSAGE_LEN)));

  ASSERT_STREQ(buffer, TEST_MESSAGE);
}

TEST_F(LlvmLibcSendRecvTest, MsgFlagsTest) {
  int sockpair[2] = {0, 0};

  ASSERT_THAT(LIBC_NAMESPACE::socketpair(AF_UNIX, SOCK_SEQPACKET, 0, sockpair),
              Succeeds(0));

  char buffer[256] = {};

  // MSG_DONTWAIT on an empty socket returns EAGAIN
  ASSERT_THAT(
      LIBC_NAMESPACE::recv(sockpair[1], buffer, sizeof(buffer), MSG_DONTWAIT),
      Fails<ssize_t>(EAGAIN));

  const char TEST_MESSAGE[] = "this is a long message";
  const size_t MESSAGE_LEN = sizeof(TEST_MESSAGE);

  ASSERT_THAT(LIBC_NAMESPACE::send(sockpair[0], TEST_MESSAGE, MESSAGE_LEN, 0),
              Succeeds(static_cast<ssize_t>(MESSAGE_LEN)));

  // MSG_PEEK does not remove the message from the socket
  ASSERT_THAT(
      LIBC_NAMESPACE::recv(sockpair[1], buffer, sizeof(buffer), MSG_PEEK),
      Succeeds(static_cast<ssize_t>(MESSAGE_LEN)));
  ASSERT_STREQ(buffer, TEST_MESSAGE);

  // Read the message again, but use a smaller buffer to test MSG_TRUNC. Return
  // value should be real length.
  char small_buffer[6] = {};
  ASSERT_THAT(LIBC_NAMESPACE::recv(sockpair[1], small_buffer,
                                   sizeof(small_buffer) - 1, MSG_TRUNC),
              Succeeds(static_cast<ssize_t>(MESSAGE_LEN)));
  ASSERT_STREQ(small_buffer, "this ");

  // Sending with MSG_NOSIGNAL to a closed socket should fail with EPIPE
  ASSERT_THAT(LIBC_NAMESPACE::close(sockpair[1]), Succeeds(0));
  ASSERT_THAT(LIBC_NAMESPACE::send(sockpair[0], "x", 1, MSG_NOSIGNAL),
              Fails<ssize_t>(EPIPE));

  ASSERT_THAT(LIBC_NAMESPACE::close(sockpair[0]), Succeeds(0));
}

TEST_F(LlvmLibcSendRecvTest, SendFails) {
  const char TEST_MESSAGE[] = "connection terminated";
  const size_t MESSAGE_LEN = sizeof(TEST_MESSAGE);

  ASSERT_THAT(LIBC_NAMESPACE::send(-1, TEST_MESSAGE, MESSAGE_LEN, 0),
              Fails(EBADF, static_cast<ssize_t>(-1)));
}

TEST_F(LlvmLibcSendRecvTest, RecvFails) {
  char buffer[256];

  ASSERT_THAT(LIBC_NAMESPACE::recv(-1, buffer, sizeof(buffer), 0),
              Fails(EBADF, static_cast<ssize_t>(-1)));
}
