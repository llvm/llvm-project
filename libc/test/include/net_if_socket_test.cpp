//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unittests for net/if.h bringing in sys/socket.h
///
//===----------------------------------------------------------------------===//

#include "include/net/if.h"
#include "test/UnitTest/Test.h"

#ifndef _LLVM_LIBC_SYS_SOCKET_H
#error "<net/if.h> does not bring in <sys/socket.h>"
#endif

TEST(LlvmLibcNetIfSocketTest, CarriesTheHeaders) {
  EXPECT_NE(SOCK_DGRAM, SOCK_STREAM);
  EXPECT_NE(AF_INET, AF_UNIX);
}
