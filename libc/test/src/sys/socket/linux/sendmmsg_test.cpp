//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unit tests for sendmmsg.
///
//===----------------------------------------------------------------------===//

#include "hdr/sys_socket_macros.h"
#include "hdr/types/struct_mmsghdr.h"
#include "src/__support/CPP/scope.h"
#include "src/string/strlen.h"
#include "src/sys/socket/recvmsg.h"
#include "src/sys/socket/sendmmsg.h"
#include "src/sys/socket/socketpair.h"
#include "src/unistd/close.h"
#include "test/UnitTest/ErrnoCheckingTest.h"
#include "test/UnitTest/ErrnoSetterMatcher.h"
#include "test/UnitTest/LibcTest.h"
#include "test/UnitTest/Test.h"

using LIBC_NAMESPACE::testing::ErrnoSetterMatcher::Fails;
using LIBC_NAMESPACE::testing::ErrnoSetterMatcher::Succeeds;
using LlvmLibcSendMmsgTest = LIBC_NAMESPACE::testing::ErrnoCheckingTest;

TEST_F(LlvmLibcSendMmsgTest, SendMmsgSucceedsWithSocketPair) {
  const char *const TEST_MESSAGES[] = {"message one", "message two"};
  const size_t MESSAGES_COUNT = 2;

  int sockpair[2] = {0, 0};

  ASSERT_THAT(LIBC_NAMESPACE::socketpair(AF_UNIX, SOCK_DGRAM, 0, sockpair),
              Succeeds(0));
  LIBC_NAMESPACE::cpp::scope_exit close_sockpair([&] {
    ASSERT_THAT(LIBC_NAMESPACE::close(sockpair[0]), Succeeds(0));
    ASSERT_THAT(LIBC_NAMESPACE::close(sockpair[1]), Succeeds(0));
  });

  struct iovec send_msg_vec[MESSAGES_COUNT] = {};
  struct mmsghdr send_msg_hdr[MESSAGES_COUNT] = {};
  for (size_t i = 0; i < MESSAGES_COUNT; ++i) {
    send_msg_vec[i].iov_base =
        reinterpret_cast<void *>(const_cast<char *>(TEST_MESSAGES[i]));
    send_msg_vec[i].iov_len = LIBC_NAMESPACE::strlen(TEST_MESSAGES[i]) + 1;
    send_msg_hdr[i].msg_hdr.msg_iov = &send_msg_vec[i];
    send_msg_hdr[i].msg_hdr.msg_iovlen = 1;
  }

  ASSERT_THAT(
      LIBC_NAMESPACE::sendmmsg(sockpair[0], send_msg_hdr, MESSAGES_COUNT, 0),
      Succeeds(static_cast<int>(MESSAGES_COUNT)));

  for (size_t i = 0; i < MESSAGES_COUNT; ++i) {
    ASSERT_EQ(static_cast<size_t>(send_msg_hdr[i].msg_len),
              LIBC_NAMESPACE::strlen(TEST_MESSAGES[i]) + 1);
  }

  for (size_t i = 0; i < MESSAGES_COUNT; ++i) {
    char recv_buffer[256] = {};
    struct iovec recv_msg_vec;
    recv_msg_vec.iov_base = reinterpret_cast<void *>(recv_buffer);
    recv_msg_vec.iov_len = sizeof(recv_buffer);

    struct msghdr recv_msg_hdr = {};
    recv_msg_hdr.msg_iov = &recv_msg_vec;
    recv_msg_hdr.msg_iovlen = 1;

    ASSERT_THAT(LIBC_NAMESPACE::recvmsg(sockpair[1], &recv_msg_hdr, 0),
                Succeeds(static_cast<ssize_t>(
                    LIBC_NAMESPACE::strlen(TEST_MESSAGES[i]) + 1)));
    ASSERT_STREQ(recv_buffer, TEST_MESSAGES[i]);
  }
}

TEST_F(LlvmLibcSendMmsgTest, SendMmsgFails) {
  struct mmsghdr msg_hdrs = {};

  ASSERT_THAT(LIBC_NAMESPACE::sendmmsg(-1, &msg_hdrs, 1, 0), Fails(EBADF, -1));
}
