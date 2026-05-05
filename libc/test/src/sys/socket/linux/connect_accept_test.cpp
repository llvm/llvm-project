//===-- Unittests for connect and accept ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "hdr/fcntl_macros.h"
#include "hdr/sys_socket_macros.h"
#include "hdr/types/size_t.h"
#include "hdr/types/struct_sockaddr_un.h"
#include "src/fcntl/fcntl.h"
#include "src/stdio/remove.h"
#include "src/sys/socket/accept.h"
#include "src/sys/socket/accept4.h"
#include "src/sys/socket/bind.h"
#include "src/sys/socket/connect.h"
#include "src/sys/socket/listen.h"
#include "src/sys/socket/socket.h"
#include "src/unistd/close.h"

#include "test/UnitTest/ErrnoCheckingTest.h"
#include "test/UnitTest/ErrnoSetterMatcher.h"
#include "test/src/sys/socket/linux/socket_test_support.h"

using LIBC_NAMESPACE::testing::make_sockaddr_un;
using LIBC_NAMESPACE::testing::ErrnoSetterMatcher::Fails;
using LIBC_NAMESPACE::testing::ErrnoSetterMatcher::Succeeds;
using LlvmLibcConnectAcceptTest = LIBC_NAMESPACE::testing::ErrnoCheckingTest;

TEST_F(LlvmLibcConnectAcceptTest, ConnectLocalSocket) {
  const char *CONNECT_FILE = "connect_file.test";
  const auto CONNECT_PATH = libc_make_test_file_path(CONNECT_FILE);
  struct sockaddr_un connect_addr;
  ASSERT_TRUE(make_sockaddr_un(CONNECT_PATH, connect_addr));

  const char *ACCEPT_FILE = "accept_file.test";
  const auto ACCEPT_PATH = libc_make_test_file_path(ACCEPT_FILE);
  struct sockaddr_un accept_addr;
  ASSERT_TRUE(make_sockaddr_un(ACCEPT_PATH, accept_addr));

  int accepting_socket = LIBC_NAMESPACE::socket(AF_UNIX, SOCK_STREAM, 0);
  ASSERT_GE(accepting_socket, 0);
  ASSERT_ERRNO_SUCCESS();

  ASSERT_THAT(LIBC_NAMESPACE::bind(
                  accepting_socket,
                  reinterpret_cast<const struct sockaddr *>(&accept_addr),
                  sizeof(struct sockaddr_un)),
              Succeeds(0));

  int connecting_socket = LIBC_NAMESPACE::socket(AF_UNIX, SOCK_STREAM, 0);
  ASSERT_GE(connecting_socket, 0);
  ASSERT_ERRNO_SUCCESS();

  // These should fail as the other side is not listen()ing yet.
  ASSERT_THAT(LIBC_NAMESPACE::accept(accepting_socket, nullptr, nullptr),
              Fails(EINVAL));
  ASSERT_THAT(LIBC_NAMESPACE::connect(
                  connecting_socket,
                  reinterpret_cast<const struct sockaddr *>(&accept_addr),
                  sizeof(struct sockaddr_un)),
              Fails(ECONNREFUSED));

  ASSERT_THAT(LIBC_NAMESPACE::listen(accepting_socket, 1), Succeeds(0));

  ASSERT_THAT(LIBC_NAMESPACE::connect(
                  connecting_socket,
                  reinterpret_cast<const struct sockaddr *>(&accept_addr),
                  sizeof(struct sockaddr_un)),
              Succeeds(0));

  int accepted_socket =
      LIBC_NAMESPACE::accept(accepting_socket, nullptr, nullptr);
  ASSERT_GE(accepted_socket, 0);
  ASSERT_ERRNO_SUCCESS();
  ASSERT_THAT(LIBC_NAMESPACE::close(accepted_socket), Succeeds(0));
  ASSERT_THAT(LIBC_NAMESPACE::close(connecting_socket), Succeeds(0));

  // Now try connecting again, but pass a non-nullptr address to accept().
  connecting_socket = LIBC_NAMESPACE::socket(AF_UNIX, SOCK_STREAM, 0);
  ASSERT_GE(connecting_socket, 0);
  ASSERT_ERRNO_SUCCESS();

  ASSERT_THAT(LIBC_NAMESPACE::bind(
                  connecting_socket,
                  reinterpret_cast<const struct sockaddr *>(&connect_addr),
                  sizeof(struct sockaddr_un)),
              Succeeds(0));

  ASSERT_THAT(LIBC_NAMESPACE::connect(
                  connecting_socket,
                  reinterpret_cast<const struct sockaddr *>(&accept_addr),
                  sizeof(struct sockaddr_un)),
              Succeeds(0));

  struct sockaddr_un accepted_addr;
  socklen_t accepted_addr_len = sizeof(accepted_addr);
  accepted_socket = LIBC_NAMESPACE::accept(
      accepting_socket, reinterpret_cast<struct sockaddr *>(&accepted_addr),
      &accepted_addr_len);
  ASSERT_GE(accepted_socket, 0);
  ASSERT_ERRNO_SUCCESS();
  ASSERT_THAT(LIBC_NAMESPACE::close(accepted_socket), Succeeds(0));
  ASSERT_THAT((LIBC_NAMESPACE::testing::SocketAddress{accepted_addr,
                                                      accepted_addr_len}),
              LIBC_NAMESPACE::testing::MatchesAddress(CONNECT_PATH));

  ASSERT_THAT(LIBC_NAMESPACE::close(accepting_socket), Succeeds(0));
  ASSERT_THAT(LIBC_NAMESPACE::close(connecting_socket), Succeeds(0));
  ASSERT_THAT(LIBC_NAMESPACE::remove(ACCEPT_PATH), Succeeds(0));
  ASSERT_THAT(LIBC_NAMESPACE::remove(CONNECT_PATH), Succeeds(0));
}

TEST_F(LlvmLibcConnectAcceptTest, Accept4Flags) {
  const char *ACCEPT_FILE = "accept4_file.test";
  auto ACCEPT_PATH = libc_make_test_file_path(ACCEPT_FILE);
  struct sockaddr_un accept_addr;
  ASSERT_TRUE(make_sockaddr_un(ACCEPT_PATH, accept_addr));

  int accepting_socket = LIBC_NAMESPACE::socket(AF_UNIX, SOCK_STREAM, 0);
  ASSERT_GE(accepting_socket, 0);
  ASSERT_ERRNO_SUCCESS();

  ASSERT_THAT(LIBC_NAMESPACE::bind(
                  accepting_socket,
                  reinterpret_cast<const struct sockaddr *>(&accept_addr),
                  sizeof(struct sockaddr_un)),
              Succeeds(0));

  // This should fail as the other side is not listen()ing yet.
  ASSERT_THAT(LIBC_NAMESPACE::accept4(accepting_socket, nullptr, nullptr, 0),
              Fails(EINVAL));

  ASSERT_THAT(LIBC_NAMESPACE::listen(accepting_socket, 1), Succeeds(0));

  int connecting_socket = LIBC_NAMESPACE::socket(AF_UNIX, SOCK_STREAM, 0);
  ASSERT_GE(connecting_socket, 0);
  ASSERT_ERRNO_SUCCESS();

  ASSERT_THAT(LIBC_NAMESPACE::connect(
                  connecting_socket,
                  reinterpret_cast<const struct sockaddr *>(&accept_addr),
                  sizeof(struct sockaddr_un)),
              Succeeds(0));

  struct sockaddr_un accepted_addr;
  socklen_t accepted_addr_len = sizeof(accepted_addr);
  int accepted_socket = LIBC_NAMESPACE::accept4(
      accepting_socket, reinterpret_cast<struct sockaddr *>(&accepted_addr),
      &accepted_addr_len, SOCK_CLOEXEC | SOCK_NONBLOCK);
  ASSERT_GE(accepted_socket, 0);
  ASSERT_ERRNO_SUCCESS();

  ASSERT_THAT(LIBC_NAMESPACE::close(connecting_socket), Succeeds(0));
  ASSERT_THAT(LIBC_NAMESPACE::close(accepting_socket), Succeeds(0));
  ASSERT_THAT(LIBC_NAMESPACE::remove(ACCEPT_PATH), Succeeds(0));

  ASSERT_EQ(accepted_addr_len, static_cast<socklen_t>(sizeof(sa_family_t)));
  ASSERT_EQ(accepted_addr.sun_family, static_cast<sa_family_t>(AF_UNIX));

  // Check FD_CLOEXEC
  int fd_flags = LIBC_NAMESPACE::fcntl(accepted_socket, F_GETFD);
  ASSERT_GE(fd_flags, 0);
  ASSERT_NE(fd_flags & FD_CLOEXEC, 0);

  // Check O_NONBLOCK
  int fl_flags = LIBC_NAMESPACE::fcntl(accepted_socket, F_GETFL);
  ASSERT_GE(fl_flags, 0);
  ASSERT_NE(fl_flags & O_NONBLOCK, 0);

  ASSERT_THAT(LIBC_NAMESPACE::close(accepted_socket), Succeeds(0));
}
