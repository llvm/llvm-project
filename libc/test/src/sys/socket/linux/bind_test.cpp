//===-- Unittests for bind ------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/CPP/scope.h"
#include "src/arpa/inet/htonl.h"
#include "src/stdio/remove.h"
#include "src/sys/socket/bind.h"
#include "src/sys/socket/getsockname.h"
#include "src/sys/socket/socket.h"
#include "src/unistd/close.h"
#include "test/UnitTest/ErrnoCheckingTest.h"
#include "test/UnitTest/ErrnoSetterMatcher.h"
#include "test/UnitTest/Test.h"
#include "test/src/sys/socket/linux/socket_test_support.h"

#include "hdr/netinet_in_macros.h"
#include "hdr/sys_socket_macros.h"
#include "hdr/types/struct_sockaddr_in.h"
#include "hdr/types/struct_sockaddr_in6.h"
#include "hdr/types/struct_sockaddr_un.h"

using LIBC_NAMESPACE::testing::ErrnoSetterMatcher::Succeeds;
using LlvmLibcBindTest = LIBC_NAMESPACE::testing::ErrnoCheckingTest;

TEST_F(LlvmLibcBindTest, BindLocalSocket) {

  const char *FILENAME = "bind_file.test";
  auto SOCK_PATH = libc_make_test_file_path(FILENAME);

  int sock = LIBC_NAMESPACE::socket(AF_UNIX, SOCK_DGRAM, 0);
  ASSERT_GE(sock, 0);
  ASSERT_ERRNO_SUCCESS();
  LIBC_NAMESPACE::cpp::scope_exit close_sock(
      [&] { ASSERT_THAT(LIBC_NAMESPACE::close(sock), Succeeds(0)); });

  struct sockaddr_un my_addr;
  ASSERT_TRUE(LIBC_NAMESPACE::testing::make_sockaddr_un(SOCK_PATH, my_addr));

  ASSERT_THAT(
      LIBC_NAMESPACE::bind(sock, reinterpret_cast<struct sockaddr *>(&my_addr),
                           sizeof(struct sockaddr_un)),
      Succeeds(0));
  ASSERT_THAT(LIBC_NAMESPACE::remove(SOCK_PATH), Succeeds(0));
}

TEST_F(LlvmLibcBindTest, BindInetSocket) {
  int sock = LIBC_NAMESPACE::socket(AF_INET, SOCK_DGRAM, 0);
  ASSERT_GE(sock, 0);
  ASSERT_ERRNO_SUCCESS();
  LIBC_NAMESPACE::cpp::scope_exit close_sock(
      [&] { ASSERT_THAT(LIBC_NAMESPACE::close(sock), Succeeds(0)); });

  struct sockaddr_in my_addr;
  my_addr.sin_family = AF_INET;
  my_addr.sin_port = 0;
  // Avoid expanding the htonl macro in overlay mode.
  my_addr.sin_addr.s_addr = (LIBC_NAMESPACE::htonl)(INADDR_LOOPBACK);

  ASSERT_THAT(
      LIBC_NAMESPACE::bind(sock, reinterpret_cast<struct sockaddr *>(&my_addr),
                           sizeof(struct sockaddr_in)),
      Succeeds(0));

  my_addr = {};
  socklen_t len = sizeof(my_addr);
  ASSERT_THAT(LIBC_NAMESPACE::getsockname(
                  sock, reinterpret_cast<struct sockaddr *>(&my_addr), &len),
              Succeeds(0));
  ASSERT_EQ(len, static_cast<socklen_t>(sizeof(struct sockaddr_in)));
  EXPECT_EQ(my_addr.sin_family, static_cast<sa_family_t>(AF_INET));
  EXPECT_NE(my_addr.sin_port, static_cast<in_port_t>(0));
  EXPECT_EQ(my_addr.sin_addr.s_addr, (LIBC_NAMESPACE::htonl)(INADDR_LOOPBACK));
}

TEST_F(LlvmLibcBindTest, BindInet6Socket) {
  int sock = LIBC_NAMESPACE::socket(AF_INET6, SOCK_DGRAM, 0);
  ASSERT_GE(sock, 0);
  ASSERT_ERRNO_SUCCESS();
  LIBC_NAMESPACE::cpp::scope_exit close_sock(
      [&] { ASSERT_THAT(LIBC_NAMESPACE::close(sock), Succeeds(0)); });

  struct sockaddr_in6 my_addr = {};
  my_addr.sin6_family = AF_INET6;
  my_addr.sin6_addr = IN6ADDR_LOOPBACK_INIT;

  ASSERT_THAT(
      LIBC_NAMESPACE::bind(sock, reinterpret_cast<struct sockaddr *>(&my_addr),
                           sizeof(struct sockaddr_in6)),
      Succeeds(0));

  my_addr = {};
  socklen_t len = sizeof(my_addr);
  ASSERT_THAT(LIBC_NAMESPACE::getsockname(
                  sock, reinterpret_cast<struct sockaddr *>(&my_addr), &len),
              Succeeds(0));
  ASSERT_EQ(len, static_cast<socklen_t>(sizeof(struct sockaddr_in6)));
  EXPECT_EQ(my_addr.sin6_family, static_cast<sa_family_t>(AF_INET6));
  EXPECT_NE(my_addr.sin6_port, static_cast<in_port_t>(0));
}
