//===-- Unittests for connect and accept ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "hdr/sys_socket_macros.h"
#include "hdr/types/size_t.h"
#include "hdr/types/struct_sockaddr_un.h"
#include "src/sys/socket/accept.h"
#include "src/sys/socket/bind.h"
#include "src/sys/socket/connect.h"
#include "src/sys/socket/listen.h"
#include "src/sys/socket/socket.h"

#include "src/stdio/remove.h"
#include "src/string/strlen.h"
#include "src/string/strncpy.h"
#include "src/unistd/close.h"

#include "test/UnitTest/ErrnoCheckingTest.h"
#include "test/UnitTest/ErrnoSetterMatcher.h"
#include "test/UnitTest/Test.h"

using LIBC_NAMESPACE::testing::ErrnoSetterMatcher::Fails;
using LIBC_NAMESPACE::testing::ErrnoSetterMatcher::Succeeds;
using LlvmLibcConnectAcceptTest = LIBC_NAMESPACE::testing::ErrnoCheckingTest;

constexpr size_t MAX_SOCKET_PATH =
    sizeof(struct sockaddr_un) - sizeof(sa_family_t);

struct sockaddr_un make_sockaddr(const char *path) {
  struct sockaddr_un result;
  result.sun_family = AF_UNIX;
  LIBC_NAMESPACE::strncpy(result.sun_path, path, MAX_SOCKET_PATH);
  return result;
}

TEST_F(LlvmLibcConnectAcceptTest, ConnectLocalSocket) {
  const char *CONNECT_FILE = "connect_file.test";
  auto CONNECT_PATH = libc_make_test_file_path(CONNECT_FILE);
  // It's important that the path fits in the struct, if it doesn't then we
  // can't try to bind to the file.
  ASSERT_LT(LIBC_NAMESPACE::strlen(CONNECT_PATH), MAX_SOCKET_PATH);
  const struct sockaddr_un CONNECT_ADDR = make_sockaddr(CONNECT_PATH);

  const char *ACCEPT_FILE = "accept_file.test";
  auto ACCEPT_PATH = libc_make_test_file_path(ACCEPT_FILE);
  ASSERT_LT(LIBC_NAMESPACE::strlen(ACCEPT_PATH), MAX_SOCKET_PATH);
  const struct sockaddr_un ACCEPT_ADDR = make_sockaddr(ACCEPT_PATH);

  int accepting_socket = LIBC_NAMESPACE::socket(AF_UNIX, SOCK_STREAM, 0);
  ASSERT_GE(accepting_socket, 0);
  ASSERT_ERRNO_SUCCESS();

  ASSERT_THAT(LIBC_NAMESPACE::bind(
                  accepting_socket,
                  reinterpret_cast<const struct sockaddr *>(&ACCEPT_ADDR),
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
                  reinterpret_cast<const struct sockaddr *>(&ACCEPT_ADDR),
                  sizeof(struct sockaddr_un)),
              Fails(ECONNREFUSED));

  ASSERT_THAT(LIBC_NAMESPACE::listen(accepting_socket, 1), Succeeds(0));

  ASSERT_THAT(LIBC_NAMESPACE::connect(
                  connecting_socket,
                  reinterpret_cast<const struct sockaddr *>(&ACCEPT_ADDR),
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
                  reinterpret_cast<const struct sockaddr *>(&CONNECT_ADDR),
                  sizeof(struct sockaddr_un)),
              Succeeds(0));

  ASSERT_THAT(LIBC_NAMESPACE::connect(
                  connecting_socket,
                  reinterpret_cast<const struct sockaddr *>(&ACCEPT_ADDR),
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
  ASSERT_EQ(accepted_addr.sun_family, static_cast<sa_family_t>(AF_UNIX));
  for (size_t i = 0; i < accepted_addr_len - sizeof(sa_family_t); ++i)
    ASSERT_EQ(accepted_addr.sun_path[i], CONNECT_ADDR.sun_path[i]);

  ASSERT_THAT(LIBC_NAMESPACE::close(accepting_socket), Succeeds(0));
  ASSERT_THAT(LIBC_NAMESPACE::close(connecting_socket), Succeeds(0));
  ASSERT_THAT(LIBC_NAMESPACE::remove(ACCEPT_PATH), Succeeds(0));
  ASSERT_THAT(LIBC_NAMESPACE::remove(CONNECT_PATH), Succeeds(0));
}
