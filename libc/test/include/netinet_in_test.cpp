//===-- Unittests for netinet/in macro ------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "include/llvm-libc-macros/netinet-in-macros.h"
#include "test/UnitTest/Test.h"

TEST(LlvmLibcNetinetInTest, IPPROTOMacro) {
  EXPECT_EQ(IPPROTO_IP, 0);
  EXPECT_EQ(IPPROTO_ICMP, 1);
  EXPECT_EQ(IPPROTO_TCP, 6);
  EXPECT_EQ(IPPROTO_UDP, 17);
  EXPECT_EQ(IPPROTO_IPV6, 41);
  EXPECT_EQ(IPPROTO_RAW, 255);
}
