//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unittests for netinet/tcp.h bringing in sys/socket.h
///
//===----------------------------------------------------------------------===//

#include "include/netinet/tcp.h"
#include "test/UnitTest/Test.h"

#ifndef _LLVM_LIBC_SYS_SOCKET_H
#error "<netinet/tcp.h> does not bring in <sys/socket.h>"
#endif

TEST(LlvmLibcNetinetTcpSocketTest, CarriesTheHeaders) {
  struct sockaddr address;
  address.sa_family = AF_INET;
  EXPECT_EQ(address.sa_family, sa_family_t(AF_INET));
}
