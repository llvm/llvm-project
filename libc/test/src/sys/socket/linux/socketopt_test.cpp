//===-- Unittests for getsockopt and setsockopt ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "hdr/sys_socket_macros.h"
#include "hdr/types/struct_linger.h"
#include "src/sys/socket/getsockopt.h"
#include "src/sys/socket/setsockopt.h"
#include "src/sys/socket/socket.h"

#include "src/unistd/close.h"
#include "src/unistd/pipe.h"

#include "test/UnitTest/ErrnoCheckingTest.h"
#include "test/UnitTest/ErrnoSetterMatcher.h"
#include "test/UnitTest/Test.h"
#include <sys/socket.h>

using LIBC_NAMESPACE::testing::ErrnoSetterMatcher::Fails;
using LIBC_NAMESPACE::testing::ErrnoSetterMatcher::Succeeds;
using LlvmLibcSocketOptTest = LIBC_NAMESPACE::testing::ErrnoCheckingTest;

TEST_F(LlvmLibcSocketOptTest, BasicSocketOpt) {
  int sock = LIBC_NAMESPACE::socket(AF_UNIX, SOCK_STREAM, 0);
  ASSERT_GE(sock, 0);

  int optval = 0;
  socklen_t optlen = sizeof(optval);

  // Test a boolean-like option
  optval = 1;
  ASSERT_THAT(LIBC_NAMESPACE::setsockopt(sock, SOL_SOCKET, SO_KEEPALIVE,
                                         &optval, optlen),
              Succeeds(0));

  optval = 0;
  ASSERT_THAT(LIBC_NAMESPACE::getsockopt(sock, SOL_SOCKET, SO_KEEPALIVE,
                                         &optval, &optlen),
              Succeeds(0));
  ASSERT_EQ(optval, 1);
  ASSERT_EQ(optlen, static_cast<socklen_t>(sizeof(optval)));

  // Test SO_TYPE (read-only)
  ASSERT_THAT(
      LIBC_NAMESPACE::getsockopt(sock, SOL_SOCKET, SO_TYPE, &optval, &optlen),
      Succeeds(0));
  ASSERT_EQ(optval, SOCK_STREAM);
  ASSERT_EQ(optlen, static_cast<socklen_t>(sizeof(optval)));

  optval = SOCK_DGRAM;
  ASSERT_THAT(
      LIBC_NAMESPACE::setsockopt(sock, SOL_SOCKET, SO_TYPE, &optval, optlen),
      Fails(ENOPROTOOPT));

  // Test SO_LINGER (uses a struct)
  struct linger lin;
  lin.l_onoff = 1;
  lin.l_linger = 5;
  optlen = sizeof(lin);
  ASSERT_THAT(
      LIBC_NAMESPACE::setsockopt(sock, SOL_SOCKET, SO_LINGER, &lin, optlen),
      Succeeds(0));

  lin = {};
  optlen = sizeof(lin);
  ASSERT_THAT(
      LIBC_NAMESPACE::getsockopt(sock, SOL_SOCKET, SO_LINGER, &lin, &optlen),
      Succeeds(0));
  ASSERT_EQ(lin.l_onoff, 1);
  ASSERT_EQ(lin.l_linger, 5);
  ASSERT_EQ(optlen, static_cast<socklen_t>(sizeof(lin)));

  ASSERT_THAT(LIBC_NAMESPACE::close(sock), Succeeds(0));
}

TEST_F(LlvmLibcSocketOptTest, NotASocket) {
  int fds[2];
  ASSERT_THAT(LIBC_NAMESPACE::pipe(fds), Succeeds(0));
  ASSERT_THAT(LIBC_NAMESPACE::close(fds[1]), Succeeds(0));

  int optval = 1;
  socklen_t optlen = sizeof(optval);
  ASSERT_THAT(LIBC_NAMESPACE::setsockopt(fds[0], SOL_SOCKET, SO_KEEPALIVE,
                                         &optval, optlen),
              Fails(ENOTSOCK));

  ASSERT_THAT(LIBC_NAMESPACE::getsockopt(fds[0], SOL_SOCKET, SO_KEEPALIVE,
                                         &optval, &optlen),
              Fails(ENOTSOCK));
  ASSERT_THAT(LIBC_NAMESPACE::close(fds[0]), Succeeds(0));
}

TEST_F(LlvmLibcSocketOptTest, InvalidSocket) {
  int optval = 1;
  socklen_t optlen = sizeof(optval);
  ASSERT_THAT(
      LIBC_NAMESPACE::setsockopt(-1, SOL_SOCKET, SO_KEEPALIVE, &optval, optlen),
      Fails(EBADF));

  ASSERT_THAT(LIBC_NAMESPACE::getsockopt(-1, SOL_SOCKET, SO_KEEPALIVE, &optval,
                                         &optlen),
              Fails(EBADF));
}
