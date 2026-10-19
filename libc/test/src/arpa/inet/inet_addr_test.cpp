//===-- Unittests for inet_addr -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/arpa/inet/htonl.h"
#include "src/arpa/inet/inet_addr.h"
#include "test/UnitTest/Test.h"

namespace LIBC_NAMESPACE_DECL {

TEST(LlvmLibcInetAddr, ValidTest) {
  EXPECT_EQ(htonl(0x7f010204), inet_addr("127.1.2.4"));
  EXPECT_EQ(htonl(0x7f010004), inet_addr("127.1.4"));
  EXPECT_EQ(htonl(0x00000000), inet_addr("0"));
  EXPECT_EQ(htonl(0x00000000), inet_addr("00"));
  EXPECT_EQ(htonl(0x00000000), inet_addr("0x0"));
}

TEST(LlvmLibcInetAddr, InvalidTest) {
  EXPECT_EQ(htonl(0xffffffff), inet_addr(""));
  EXPECT_EQ(htonl(0xffffffff), inet_addr("x"));
  EXPECT_EQ(htonl(0xffffffff), inet_addr("0b0"));
  EXPECT_EQ(htonl(0xffffffff), inet_addr("0B0"));
}

} // namespace LIBC_NAMESPACE_DECL
