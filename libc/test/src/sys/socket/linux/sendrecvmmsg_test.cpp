//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unit tests for sendmmsg and recvmmsg.
///
//===----------------------------------------------------------------------===//

#include "hdr/sys_socket_macros.h"
#include "hdr/types/struct_mmsghdr.h"
#include "hdr/types/struct_timespec.h"
#include "src/__support/CPP/scope.h"
#include "src/string/strlen.h"
#include "src/sys/socket/recvmmsg.h"
#include "src/sys/socket/sendmmsg.h"
#include "src/sys/socket/socketpair.h"
#include "src/unistd/close.h"
#include "test/UnitTest/ErrnoCheckingTest.h"
#include "test/UnitTest/ErrnoSetterMatcher.h"
#include "test/UnitTest/LibcTest.h"
#include "test/UnitTest/Test.h"

using LIBC_NAMESPACE::testing::ErrnoSetterMatcher::Fails;
using LIBC_NAMESPACE::testing::ErrnoSetterMatcher::Succeeds;
using LlvmLibcSendRecvMmsgTest = LIBC_NAMESPACE::testing::ErrnoCheckingTest;

TEST_F(LlvmLibcSendRecvMmsgTest, SendRecvMmsgSucceedsWithSocketPair) {
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

  char recv_buffers[MESSAGES_COUNT][256] = {};
  struct iovec recv_msg_vec[MESSAGES_COUNT] = {};
  struct mmsghdr recv_msg_hdr[MESSAGES_COUNT] = {};
  for (size_t i = 0; i < MESSAGES_COUNT; ++i) {
    recv_msg_vec[i].iov_base = reinterpret_cast<void *>(recv_buffers[i]);
    recv_msg_vec[i].iov_len = sizeof(recv_buffers[i]);
    recv_msg_hdr[i].msg_hdr.msg_iov = &recv_msg_vec[i];
    recv_msg_hdr[i].msg_hdr.msg_iovlen = 1;
  }

  struct timespec invalid_timeout = {-1, 0};
  ASSERT_THAT(LIBC_NAMESPACE::recvmmsg(sockpair[1], recv_msg_hdr,
                                       MESSAGES_COUNT, 0, &invalid_timeout),
              Fails<int>(EINVAL));

  ASSERT_THAT(LIBC_NAMESPACE::recvmmsg(sockpair[1], recv_msg_hdr,
                                       MESSAGES_COUNT, 0, nullptr),
              Succeeds(static_cast<int>(MESSAGES_COUNT)));

  for (size_t i = 0; i < MESSAGES_COUNT; ++i) {
    ASSERT_EQ(static_cast<size_t>(recv_msg_hdr[i].msg_len),
              LIBC_NAMESPACE::strlen(TEST_MESSAGES[i]) + 1);
    ASSERT_STREQ(recv_buffers[i], TEST_MESSAGES[i]);
  }
}

TEST_F(LlvmLibcSendRecvMmsgTest, SendMmsgFails) {
  struct mmsghdr msg_hdrs = {};
  ASSERT_THAT(LIBC_NAMESPACE::sendmmsg(-1, &msg_hdrs, 1, 0), Fails(EBADF, -1));
}

TEST_F(LlvmLibcSendRecvMmsgTest, RecvmmsgFails) {
  struct mmsghdr msg_hdrs = {};
  ASSERT_THAT(LIBC_NAMESPACE::recvmmsg(-1, &msg_hdrs, 1, 0, nullptr),
              Fails(EBADF, -1));
}
