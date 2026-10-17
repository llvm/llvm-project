//===-- Unittests for getsockopt and setsockopt ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "hdr/sys_socket_macros.h"
#include "hdr/time_macros.h"
#include "hdr/types/struct_linger.h"
#include "hdr/types/struct_timespec.h"
#include "hdr/types/struct_timeval.h"
#include "src/__support/time/clock_gettime.h"
#include "src/sys/socket/getsockopt.h"
#include "src/sys/socket/recv.h"
#include "src/sys/socket/setsockopt.h"
#include "src/sys/socket/socket.h"
#include "src/sys/socket/socketpair.h"
#include "src/unistd/close.h"
#include "src/unistd/pipe.h"

#include "src/__support/CPP/scope.h"
#include "test/UnitTest/ErrnoCheckingTest.h"
#include "test/UnitTest/ErrnoSetterMatcher.h"
#include "test/UnitTest/Test.h"

using LIBC_NAMESPACE::testing::ErrnoSetterMatcher::Fails;
using LIBC_NAMESPACE::testing::ErrnoSetterMatcher::Succeeds;
using LlvmLibcSocketOptTest = LIBC_NAMESPACE::testing::ErrnoCheckingTest;
using LIBC_NAMESPACE::cpp::scope_exit;

TEST_F(LlvmLibcSocketOptTest, BasicSocketOpt) {
  int sock = LIBC_NAMESPACE::socket(AF_UNIX, SOCK_STREAM, 0);
  ASSERT_GE(sock, 0);
  scope_exit close_sock(
      [&] { ASSERT_THAT(LIBC_NAMESPACE::close(sock), Succeeds(0)); });

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
  ASSERT_EQ(optval, static_cast<int>(SOCK_STREAM));
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
}

TEST_F(LlvmLibcSocketOptTest, NotASocket) {
  int fds[2];
  ASSERT_THAT(LIBC_NAMESPACE::pipe(fds), Succeeds(0));
  scope_exit close_fd0(
      [&] { ASSERT_THAT(LIBC_NAMESPACE::close(fds[0]), Succeeds(0)); });
  ASSERT_THAT(LIBC_NAMESPACE::close(fds[1]), Succeeds(0));

  int optval = 1;
  socklen_t optlen = sizeof(optval);
  ASSERT_THAT(LIBC_NAMESPACE::setsockopt(fds[0], SOL_SOCKET, SO_KEEPALIVE,
                                         &optval, optlen),
              Fails(ENOTSOCK));

  ASSERT_THAT(LIBC_NAMESPACE::getsockopt(fds[0], SOL_SOCKET, SO_KEEPALIVE,
                                         &optval, &optlen),
              Fails(ENOTSOCK));
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

TEST_F(LlvmLibcSocketOptTest, ReceiveTimeout) {
  int sv[2] = {0, 0};
  ASSERT_THAT(LIBC_NAMESPACE::socketpair(AF_UNIX, SOCK_STREAM, 0, sv),
              Succeeds(0));
  scope_exit close_sv([&] {
    ASSERT_THAT(LIBC_NAMESPACE::close(sv[0]), Succeeds(0));
    ASSERT_THAT(LIBC_NAMESPACE::close(sv[1]), Succeeds(0));
  });

  struct timeval tv;
  tv.tv_sec = 1;
  tv.tv_usec = 0;
  socklen_t optlen = sizeof(tv);
  ASSERT_THAT(
      LIBC_NAMESPACE::setsockopt(sv[0], SOL_SOCKET, SO_RCVTIMEO, &tv, optlen),
      Succeeds(0));

  // Retrieve option to verify it was set correctly.
  struct timeval retrieved_tv;
  retrieved_tv.tv_sec = 0;
  retrieved_tv.tv_usec = 0;
  socklen_t retrieved_optlen = sizeof(retrieved_tv);
  ASSERT_THAT(LIBC_NAMESPACE::getsockopt(sv[0], SOL_SOCKET, SO_RCVTIMEO,
                                         &retrieved_tv, &retrieved_optlen),
              Succeeds(0));
#ifdef LIBC_TEST_UNDER_EMULATOR
  // Under QEMU getsockopt(SO_RCVTIMEO) may return a length of 0.
  ASSERT_TRUE(retrieved_optlen == optlen || retrieved_optlen == 0);
  if (retrieved_optlen == optlen) {
    ASSERT_EQ(retrieved_tv.tv_sec, tv.tv_sec);
  }
#else
  ASSERT_EQ(retrieved_optlen, optlen);
  ASSERT_EQ(retrieved_tv.tv_sec, tv.tv_sec);
#endif

  char buffer[10];
  struct timespec start, end;
  ASSERT_TRUE(LIBC_NAMESPACE::internal::clock_gettime(CLOCK_MONOTONIC, &start)
                  .has_value());
  // Read/recv on empty socket should block for ~1s and fail with EAGAIN.
  ASSERT_THAT(LIBC_NAMESPACE::recv(sv[0], buffer, sizeof(buffer), 0),
              Fails<ssize_t>(EAGAIN));
  ASSERT_TRUE(LIBC_NAMESPACE::internal::clock_gettime(CLOCK_MONOTONIC, &end)
                  .has_value());

  int64_t elapsed_seconds = end.tv_sec - start.tv_sec;
  int64_t elapsed_nseconds = end.tv_nsec - start.tv_nsec;
  int64_t elapsed_ms = elapsed_seconds * 1000 + elapsed_nseconds / 1000000;
  ASSERT_GE(elapsed_ms, static_cast<int64_t>(1000));
  ASSERT_LT(elapsed_ms, static_cast<int64_t>(10000));
}
