//===-- Unittests for listen ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "hdr/sys_socket_macros.h"
#include "hdr/types/struct_sockaddr_un.h"
#include "src/sys/socket/bind.h"
#include "src/sys/socket/listen.h"
#include "src/sys/socket/socket.h"

#include "src/stdio/remove.h"
#include "src/unistd/close.h"

#include "src/__support/CPP/scope.h"
#include "test/UnitTest/ErrnoCheckingTest.h"
#include "test/UnitTest/ErrnoSetterMatcher.h"
#include "test/src/sys/socket/linux/socket_test_support.h"

using LIBC_NAMESPACE::testing::ErrnoSetterMatcher::Fails;
using LIBC_NAMESPACE::testing::ErrnoSetterMatcher::Succeeds;
using LlvmLibcListenTest = LIBC_NAMESPACE::testing::ErrnoCheckingTest;
using LIBC_NAMESPACE::cpp::scope_exit;

TEST_F(LlvmLibcListenTest, ListenLocalSocket) {

  const char *FILENAME = "listen_file.test";
  auto SOCK_PATH = libc_make_test_file_path(FILENAME);

  int sock = LIBC_NAMESPACE::socket(AF_UNIX, SOCK_STREAM, 0);
  ASSERT_GE(sock, 0);
  ASSERT_ERRNO_SUCCESS();
  scope_exit close_sock(
      [&] { ASSERT_THAT(LIBC_NAMESPACE::close(sock), Succeeds(0)); });

  struct sockaddr_un my_addr;
  ASSERT_TRUE(LIBC_NAMESPACE::testing::make_sockaddr_un(SOCK_PATH, my_addr));

  ASSERT_THAT(
      LIBC_NAMESPACE::bind(sock, reinterpret_cast<struct sockaddr *>(&my_addr),
                           sizeof(struct sockaddr_un)),
      Succeeds(0));
  scope_exit remove_sock_path(
      [&] { ASSERT_THAT(LIBC_NAMESPACE::remove(SOCK_PATH), Succeeds(0)); });

  ASSERT_THAT(LIBC_NAMESPACE::listen(sock, 5), Succeeds(0));
}

TEST_F(LlvmLibcListenTest, ListenFails) {
  ASSERT_THAT(LIBC_NAMESPACE::listen(-1, 5), Fails(EBADF));
}
