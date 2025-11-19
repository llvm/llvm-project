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

TEST(LlvmLibcNetinetInTest, IPV6Macro) {
  EXPECT_EQ(IPV6_UNICAST_HOPS, 16);
  EXPECT_EQ(IPV6_MULTICAST_IF, 17);
  EXPECT_EQ(IPV6_MULTICAST_HOPS, 18);
  EXPECT_EQ(IPV6_MULTICAST_LOOP, 19);
  EXPECT_EQ(IPV6_JOIN_GROUP, 20);
  EXPECT_EQ(IPV6_LEAVE_GROUP, 21);
  EXPECT_EQ(IPV6_V6ONLY, 26);
}
