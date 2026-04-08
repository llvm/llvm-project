//===-- Unittests for connect ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "hdr/sys_socket_macros.h"
#include "hdr/types/struct_sockaddr_un.h"
#include "src/sys/socket/bind.h"
#include "src/sys/socket/connect.h"
#include "src/sys/socket/socket.h"

#include "src/stdio/remove.h"
#include "src/unistd/close.h"

#include "test/UnitTest/ErrnoCheckingTest.h"
#include "test/UnitTest/ErrnoSetterMatcher.h"
#include "test/UnitTest/Test.h"

using LIBC_NAMESPACE::testing::ErrnoSetterMatcher::Fails;
using LIBC_NAMESPACE::testing::ErrnoSetterMatcher::Succeeds;
using LlvmLibcConnectTest = LIBC_NAMESPACE::testing::ErrnoCheckingTest;

TEST_F(LlvmLibcConnectTest, ConnectLocalSocket) {

  const char *FILENAME = "connect_file.test";
  auto SOCK_PATH = libc_make_test_file_path(FILENAME);

  int sock1 = LIBC_NAMESPACE::socket(AF_UNIX, SOCK_STREAM, 0);
  ASSERT_GE(sock1, 0);
  ASSERT_ERRNO_SUCCESS();

  struct sockaddr_un my_addr;

  my_addr.sun_family = AF_UNIX;
  unsigned int i = 0;
  for (;
       SOCK_PATH[i] != '\0' && (i < sizeof(sockaddr_un) - sizeof(sa_family_t));
       ++i)
    my_addr.sun_path[i] = SOCK_PATH[i];

  // It's important that the path fits in the struct, if it doesn't then we
  // can't try to bind to the file.
  ASSERT_LT(
      i, static_cast<unsigned int>(sizeof(sockaddr_un) - sizeof(sa_family_t)));
  my_addr.sun_path[i] = '\0';

  ASSERT_THAT(
      LIBC_NAMESPACE::bind(sock1, reinterpret_cast<struct sockaddr *>(&my_addr),
                           sizeof(struct sockaddr_un)),
      Succeeds(0));

  int sock2 = LIBC_NAMESPACE::socket(AF_UNIX, SOCK_STREAM, 0);
  ASSERT_GE(sock2, 0);
  ASSERT_ERRNO_SUCCESS();

  ASSERT_THAT(
      LIBC_NAMESPACE::connect(sock2,
                              reinterpret_cast<struct sockaddr *>(&my_addr),
                              sizeof(struct sockaddr_un)),
      Fails(ECONNREFUSED)); // Because the other side is not listen()ing.

  ASSERT_THAT(LIBC_NAMESPACE::close(sock1), Succeeds(0));
  ASSERT_THAT(LIBC_NAMESPACE::close(sock2), Succeeds(0));
  ASSERT_THAT(LIBC_NAMESPACE::remove(SOCK_PATH), Succeeds(0));
}
