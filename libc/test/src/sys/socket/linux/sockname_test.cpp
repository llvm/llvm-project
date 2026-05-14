//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unittests for getsockname and getpeername.
///
//===----------------------------------------------------------------------===//

#include "hdr/sys_socket_macros.h"
#include "hdr/types/struct_sockaddr_un.h"
#include "src/__support/CPP/scope.h"
#include "src/stdio/remove.h"
#include "src/sys/socket/accept.h"
#include "src/sys/socket/bind.h"
#include "src/sys/socket/connect.h"
#include "src/sys/socket/getpeername.h"
#include "src/sys/socket/getsockname.h"
#include "src/sys/socket/listen.h"
#include "src/sys/socket/socket.h"
#include "src/unistd/close.h"
#include "test/UnitTest/ErrnoCheckingTest.h"
#include "test/UnitTest/ErrnoSetterMatcher.h"
#include "test/src/sys/socket/linux/socket_test_support.h"

using LIBC_NAMESPACE::testing::make_sockaddr_un;
using LIBC_NAMESPACE::testing::ErrnoSetterMatcher::Fails;
using LIBC_NAMESPACE::testing::ErrnoSetterMatcher::Succeeds;
using LlvmLibcSockNameTest = LIBC_NAMESPACE::testing::ErrnoCheckingTest;
using LIBC_NAMESPACE::cpp::scope_exit;

TEST_F(LlvmLibcSockNameTest, GetSockNameAndPeerName) {
  // 1. Invalid Socket
  struct sockaddr_un addr;
  socklen_t addr_len = sizeof(addr);
  ASSERT_THAT(LIBC_NAMESPACE::getsockname(
                  -1, reinterpret_cast<struct sockaddr *>(&addr), &addr_len),
              Fails(EBADF));
  ASSERT_THAT(LIBC_NAMESPACE::getpeername(
                  -1, reinterpret_cast<struct sockaddr *>(&addr), &addr_len),
              Fails(EBADF));

  // 2. Unbound Socket
  int sock = LIBC_NAMESPACE::socket(AF_UNIX, SOCK_STREAM, 0);
  ASSERT_GE(sock, 0);
  ASSERT_ERRNO_SUCCESS();
  scope_exit close_sock(
      [&] { ASSERT_THAT(LIBC_NAMESPACE::close(sock), Succeeds(0)); });

  // getsockname on unbound socket should succeed
  addr_len = sizeof(addr);
  ASSERT_THAT(LIBC_NAMESPACE::getsockname(
                  sock, reinterpret_cast<struct sockaddr *>(&addr), &addr_len),
              Succeeds(0));
  ASSERT_GE(addr_len, static_cast<socklen_t>(sizeof(sa_family_t)));
  ASSERT_EQ(addr.sun_family, static_cast<sa_family_t>(AF_UNIX));

  // getpeername on unbound/unconnected socket should fail with ENOTCONN
  addr_len = sizeof(addr);
  ASSERT_THAT(LIBC_NAMESPACE::getpeername(
                  sock, reinterpret_cast<struct sockaddr *>(&addr), &addr_len),
              Fails(ENOTCONN));

  // 3. Connected Sockets
  const char *client_file = "getsockname_client.test";
  const auto client_path = libc_make_test_file_path(client_file);
  struct sockaddr_un client_addr;
  ASSERT_TRUE(make_sockaddr_un(client_path, client_addr));

  const char *server_file = "getsockname_server.test";
  const auto server_path = libc_make_test_file_path(server_file);
  struct sockaddr_un server_addr;
  ASSERT_TRUE(make_sockaddr_un(server_path, server_addr));

  int server_sock = LIBC_NAMESPACE::socket(AF_UNIX, SOCK_STREAM, 0);
  ASSERT_GE(server_sock, 0);
  ASSERT_ERRNO_SUCCESS();
  scope_exit close_server_sock(
      [&] { ASSERT_THAT(LIBC_NAMESPACE::close(server_sock), Succeeds(0)); });

  ASSERT_THAT(LIBC_NAMESPACE::bind(
                  server_sock,
                  reinterpret_cast<const struct sockaddr *>(&server_addr),
                  sizeof(struct sockaddr_un)),
              Succeeds(0));
  scope_exit remove_server_path(
      [&] { ASSERT_THAT(LIBC_NAMESPACE::remove(server_path), Succeeds(0)); });

  ASSERT_THAT(LIBC_NAMESPACE::listen(server_sock, 1), Succeeds(0));

  int client_sock = LIBC_NAMESPACE::socket(AF_UNIX, SOCK_STREAM, 0);
  ASSERT_GE(client_sock, 0);
  ASSERT_ERRNO_SUCCESS();
  scope_exit close_client_sock(
      [&] { ASSERT_THAT(LIBC_NAMESPACE::close(client_sock), Succeeds(0)); });

  ASSERT_THAT(LIBC_NAMESPACE::bind(
                  client_sock,
                  reinterpret_cast<const struct sockaddr *>(&client_addr),
                  sizeof(struct sockaddr_un)),
              Succeeds(0));
  scope_exit remove_client_path(
      [&] { ASSERT_THAT(LIBC_NAMESPACE::remove(client_path), Succeeds(0)); });

  ASSERT_THAT(LIBC_NAMESPACE::connect(
                  client_sock,
                  reinterpret_cast<const struct sockaddr *>(&server_addr),
                  sizeof(struct sockaddr_un)),
              Succeeds(0));

  int accepted_sock = LIBC_NAMESPACE::accept(server_sock, nullptr, nullptr);
  ASSERT_GE(accepted_sock, 0);
  ASSERT_ERRNO_SUCCESS();
  scope_exit close_accepted_sock(
      [&] { ASSERT_THAT(LIBC_NAMESPACE::close(accepted_sock), Succeeds(0)); });

  // Test getsockname on client_sock (should be client_path)
  addr_len = sizeof(addr);
  ASSERT_THAT(
      LIBC_NAMESPACE::getsockname(
          client_sock, reinterpret_cast<struct sockaddr *>(&addr), &addr_len),
      Succeeds(0));
  ASSERT_THAT((LIBC_NAMESPACE::testing::SocketAddress{addr, addr_len}),
              LIBC_NAMESPACE::testing::MatchesAddress(client_path));

  // Test getpeername on client_sock (should be server_path)
  addr_len = sizeof(addr);
  ASSERT_THAT(
      LIBC_NAMESPACE::getpeername(
          client_sock, reinterpret_cast<struct sockaddr *>(&addr), &addr_len),
      Succeeds(0));
  ASSERT_THAT((LIBC_NAMESPACE::testing::SocketAddress{addr, addr_len}),
              LIBC_NAMESPACE::testing::MatchesAddress(server_path));

  // Test getsockname on accepted_sock (should be server_path)
  addr_len = sizeof(addr);
  ASSERT_THAT(
      LIBC_NAMESPACE::getsockname(
          accepted_sock, reinterpret_cast<struct sockaddr *>(&addr), &addr_len),
      Succeeds(0));
  ASSERT_THAT((LIBC_NAMESPACE::testing::SocketAddress{addr, addr_len}),
              LIBC_NAMESPACE::testing::MatchesAddress(server_path));

  // Test getpeername on accepted_sock (should be client_path)
  addr_len = sizeof(addr);
  ASSERT_THAT(
      LIBC_NAMESPACE::getpeername(
          accepted_sock, reinterpret_cast<struct sockaddr *>(&addr), &addr_len),
      Succeeds(0));
  ASSERT_THAT((LIBC_NAMESPACE::testing::SocketAddress{addr, addr_len}),
              LIBC_NAMESPACE::testing::MatchesAddress(client_path));
}
