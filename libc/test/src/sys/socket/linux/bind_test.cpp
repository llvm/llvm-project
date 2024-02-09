//===-- Unittests for bind ------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/sys/socket/bind.h"
#include "src/sys/socket/socket.h"

#include "src/stdio/remove.h"
#include "src/unistd/close.h"

#include "src/errno/libc_errno.h"
#include "test/UnitTest/Test.h"

#include <sys/socket.h> // For AF_UNIX and SOCK_DGRAM

TEST(LlvmLibcSocketTest, BindLocalSocket) {

  const char *FILENAME = "bind_file.test";
  auto SOCK_PATH = libc_make_test_file_path(FILENAME);

  int sock = LIBC_NAMESPACE::socket(AF_UNIX, SOCK_DGRAM, 0);
  ASSERT_GE(sock, 0);
  ASSERT_ERRNO_SUCCESS();

  struct sockaddr_un my_addr;

  my_addr.sun_family = AF_UNIX;
  unsigned int i = 0;
  for (;
       SOCK_PATH[i] != '\0' && (i < sizeof(sockaddr_un) - sizeof(sa_family_t));
       ++i)
    my_addr.sun_path[i] = SOCK_PATH[i];
  my_addr.sun_path[i] = '\0';

  // It's important that the path fits in the struct, if it doesn't then we
  // can't try to bind to the file.
  ASSERT_LT(
      i, static_cast<unsigned int>(sizeof(sockaddr_un) - sizeof(sa_family_t)));

  int result =
      LIBC_NAMESPACE::bind(sock, reinterpret_cast<struct sockaddr *>(&my_addr),
                           sizeof(struct sockaddr_un));

  ASSERT_EQ(result, 0);
  ASSERT_ERRNO_SUCCESS();

  LIBC_NAMESPACE::close(sock);

  LIBC_NAMESPACE::remove(SOCK_PATH);
}
