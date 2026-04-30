//===-- Unittests for sendmsg/recvmsg -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "hdr/fcntl_macros.h"
#include "hdr/sys_socket_macros.h"
#include "hdr/types/struct_cmsghdr.h"
#include "src/fcntl/fcntl.h"
#include "src/string/memcpy.h"
#include "src/string/memset.h"
#include "src/sys/socket/getsockopt.h"
#include "src/sys/socket/recvmsg.h"
#include "src/sys/socket/sendmsg.h"
#include "src/sys/socket/socketpair.h"

#include "src/unistd/close.h"

#include "test/UnitTest/ErrnoCheckingTest.h"
#include "test/UnitTest/ErrnoSetterMatcher.h"
#include "test/UnitTest/LibcTest.h"
#include "test/UnitTest/Test.h"

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

TEST_F(LlvmLibcSendMsgRecvMsgTest, CmsgDetails) {
  ASSERT_EQ(CMSG_ALIGN(0), static_cast<size_t>(0));
  ASSERT_EQ(CMSG_ALIGN(1), sizeof(size_t));

  // Some implementations align struct cmsghdr in various size computations, but
  // this is a noop. This verifies that.
  ASSERT_EQ(CMSG_ALIGN(sizeof(struct cmsghdr)), sizeof(struct cmsghdr));

  char buf[0x100] = {};

  struct msghdr msg;
  msg.msg_control = buf;

  // We shouldn't be able to get the first header if there's not enough space
  // for it.
  msg.msg_controllen = 0;
  ASSERT_EQ(CMSG_FIRSTHDR(&msg), nullptr);
  msg.msg_controllen = sizeof(struct cmsghdr) - 1;
  ASSERT_EQ(CMSG_FIRSTHDR(&msg), nullptr);
  msg.msg_controllen = sizeof(struct cmsghdr);
  ASSERT_EQ(CMSG_FIRSTHDR(&msg), reinterpret_cast<struct cmsghdr *>(buf));
  msg.msg_controllen = sizeof(buf);
  struct cmsghdr *cmsg = CMSG_FIRSTHDR(&msg);
  ASSERT_EQ(cmsg, reinterpret_cast<struct cmsghdr *>(buf));

  // We shouldn't be able to get the next header if this one is too big.
  cmsg->cmsg_len = 0x1000;
  ASSERT_EQ(CMSG_NXTHDR(&msg, cmsg), nullptr);
  cmsg->cmsg_len = sizeof(buf) - sizeof(struct cmsghdr) + 1;
  ASSERT_EQ(CMSG_NXTHDR(&msg, cmsg), nullptr);

  cmsg->cmsg_len = sizeof(buf) - sizeof(struct cmsghdr);
  struct cmsghdr *cmsg2 = CMSG_NXTHDR(&msg, cmsg);
  ASSERT_LT(buf, reinterpret_cast<char *>(cmsg2));
  ASSERT_LT(reinterpret_cast<char *>(cmsg2), buf + sizeof(buf));

  // POSIX explicitly does not specify whether CMSG_NXTHDR returns the
  // next header if its data array would extend beyond the end of the buffer.
  // Our implementation does.
#ifdef LIBC_FULL_BUILD
  cmsg2->cmsg_len = sizeof(struct cmsghdr) + 1;
  ASSERT_EQ(CMSG_NXTHDR(&msg, cmsg), cmsg2);
#endif
}

TEST_F(LlvmLibcSendMsgRecvMsgTest, SendAndReceiveFileDescriptor) {
  int sockpair[2] = {0, 0};

  ASSERT_THAT(LIBC_NAMESPACE::socketpair(AF_UNIX, SOCK_STREAM, 0, sockpair),
              Succeeds(0));

  struct iovec iov;
  iov.iov_base = reinterpret_cast<void *>(const_cast<char *>("x"));
  iov.iov_len = 1;

  char control_buf[CMSG_SPACE(sizeof(int))] = {};

  struct msghdr msg;
  msg.msg_name = nullptr;
  msg.msg_namelen = 0;
  msg.msg_iov = &iov;
  msg.msg_iovlen = 1;
  msg.msg_flags = 0;
  msg.msg_control = control_buf;
  msg.msg_controllen = CMSG_LEN(sizeof(int));

  struct cmsghdr *cmsg = CMSG_FIRSTHDR(&msg);
  cmsg->cmsg_level = SOL_SOCKET;
  cmsg->cmsg_type = SCM_RIGHTS;
  cmsg->cmsg_len = CMSG_LEN(sizeof(int));
  LIBC_NAMESPACE::memcpy(CMSG_DATA(cmsg), sockpair + 1, sizeof(int));

  ASSERT_THAT(LIBC_NAMESPACE::sendmsg(sockpair[0], &msg, 0),
              Succeeds(static_cast<ssize_t>(1)));

  char buffer[256];

  iov.iov_base = buffer;
  iov.iov_len = sizeof(buffer);

  LIBC_NAMESPACE::memset(control_buf, 0, sizeof(control_buf));

  msg.msg_name = nullptr;
  msg.msg_namelen = 0;
  msg.msg_iov = &iov;
  msg.msg_iovlen = 1;
  msg.msg_control = control_buf;
  msg.msg_controllen = sizeof(control_buf);
  msg.msg_flags = 0;

  ASSERT_THAT(LIBC_NAMESPACE::recvmsg(sockpair[1], &msg, 0),
              Succeeds(static_cast<ssize_t>(1)));

  ASSERT_EQ(buffer[0], 'x');

  cmsg = CMSG_FIRSTHDR(&msg);

  ASSERT_TRUE(cmsg != nullptr);
  ASSERT_EQ(cmsg->cmsg_level, SOL_SOCKET);
  // Use ASSERT_TRUE, as ASSERT_EQ requires SCM_RIGHTS to be an int,
  // which is not true on all systems (e.g. glibc).
  ASSERT_TRUE(cmsg->cmsg_type == SCM_RIGHTS);
  ASSERT_EQ(cmsg->cmsg_len, CMSG_LEN(sizeof(int)));
  ASSERT_EQ(CMSG_NXTHDR(&msg, cmsg), nullptr);

  int new_fd;
  LIBC_NAMESPACE::memcpy(&new_fd, CMSG_DATA(cmsg), sizeof(int));

  int new_sock_type = 0;
  socklen_t optlen = sizeof(int);
  ASSERT_THAT(LIBC_NAMESPACE::getsockopt(new_fd, SOL_SOCKET, SO_TYPE,
                                         &new_sock_type, &optlen),
              Succeeds(0));
  // Use ASSERT_TRUE, as ASSERT_EQ requires SOCK_STREAM to be an int,
  // which is not true on all systems (e.g. glibc).
  ASSERT_TRUE(new_sock_type == SOCK_STREAM);

  ASSERT_THAT(LIBC_NAMESPACE::close(sockpair[0]), Succeeds(0));
  ASSERT_THAT(LIBC_NAMESPACE::close(sockpair[1]), Succeeds(0));
  ASSERT_THAT(LIBC_NAMESPACE::close(new_fd), Succeeds(0));
}

TEST_F(LlvmLibcSendMsgRecvMsgTest, MsgCmsgCloexec) {
  int sockpair[2] = {0, 0};

  ASSERT_THAT(LIBC_NAMESPACE::socketpair(AF_UNIX, SOCK_STREAM, 0, sockpair),
              Succeeds(0));

  struct iovec iov;
  iov.iov_base = reinterpret_cast<void *>(const_cast<char *>("x"));
  iov.iov_len = 1;

  char control_buf[CMSG_SPACE(sizeof(int))] = {};

  struct msghdr msg;
  msg.msg_name = nullptr;
  msg.msg_namelen = 0;
  msg.msg_iov = &iov;
  msg.msg_iovlen = 1;
  msg.msg_flags = 0;
  msg.msg_control = control_buf;
  msg.msg_controllen = CMSG_LEN(sizeof(int));

  struct cmsghdr *cmsg = CMSG_FIRSTHDR(&msg);
  cmsg->cmsg_level = SOL_SOCKET;
  cmsg->cmsg_type = SCM_RIGHTS;
  cmsg->cmsg_len = CMSG_LEN(sizeof(int));
  LIBC_NAMESPACE::memcpy(CMSG_DATA(cmsg), sockpair + 1, sizeof(int));

  ASSERT_THAT(LIBC_NAMESPACE::sendmsg(sockpair[0], &msg, 0),
              Succeeds(static_cast<ssize_t>(1)));

  char buffer[256];

  iov.iov_base = buffer;
  iov.iov_len = sizeof(buffer);

  LIBC_NAMESPACE::memset(control_buf, 0, sizeof(control_buf));

  msg.msg_name = nullptr;
  msg.msg_namelen = 0;
  msg.msg_iov = &iov;
  msg.msg_iovlen = 1;
  msg.msg_control = control_buf;
  msg.msg_controllen = sizeof(control_buf);
  msg.msg_flags = 0;

  // Receive with MSG_CMSG_CLOEXEC
  ASSERT_THAT(LIBC_NAMESPACE::recvmsg(sockpair[1], &msg, MSG_CMSG_CLOEXEC),
              Succeeds(static_cast<ssize_t>(1)));

  cmsg = CMSG_FIRSTHDR(&msg);
  ASSERT_TRUE(cmsg != nullptr);

  int new_fd;
  LIBC_NAMESPACE::memcpy(&new_fd, CMSG_DATA(cmsg), sizeof(int));

  // Check FD_CLOEXEC
  int flags = LIBC_NAMESPACE::fcntl(new_fd, F_GETFD);
  ASSERT_GE(flags, 0);
  ASSERT_NE(flags & FD_CLOEXEC, 0);

  ASSERT_THAT(LIBC_NAMESPACE::close(sockpair[0]), Succeeds(0));
  ASSERT_THAT(LIBC_NAMESPACE::close(sockpair[1]), Succeeds(0));
  ASSERT_THAT(LIBC_NAMESPACE::close(new_fd), Succeeds(0));
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

  ASSERT_THAT(LIBC_NAMESPACE::sendmsg(-1, &send_message, 0),
              Fails(EBADF, static_cast<ssize_t>(-1)));
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

  ASSERT_THAT(LIBC_NAMESPACE::recvmsg(-1, &recv_message, 0),
              Fails(EBADF, static_cast<ssize_t>(-1)));
}
