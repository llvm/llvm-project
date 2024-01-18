//===-- Unittests for socket ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/sys/socket/socket.h"

#include "src/unistd/close.h"

#include "src/errno/libc_errno.h"
#include "test/UnitTest/Test.h"

#include <sys/socket.h> // For AF_LOCAL and SOCK_DGRAM

TEST(LlvmLibcSocketTest, LocalSocket) {
  int sock = LIBC_NAMESPACE::socket(AF_LOCAL, SOCK_DGRAM, 0);
  ASSERT_GE(sock, 0);
  ASSERT_EQ(libc_errno, 0);

  LIBC_NAMESPACE::close(sock);
}
