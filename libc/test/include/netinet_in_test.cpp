//===-- Unittests for netinet/in macro ------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "include/llvm-libc-macros/netinet-in-macros.h"
#include "test/UnitTest/Test.h"

TEST(LlvmLibcNetinetInTest, IN6Macro) {
  char buff[16] = {};

  EXPECT_TRUE(IN6_IS_ADDR_UNSPECIFIED(buff));
  for (int i = 0; i < 16; ++i) {
    buff[i] = 1;
    EXPECT_FALSE(IN6_IS_ADDR_UNSPECIFIED(buff));
    buff[i] = 0;
  }

  EXPECT_FALSE(IN6_IS_ADDR_LOOPBACK(buff));
  buff[15] = 1;
  EXPECT_TRUE(IN6_IS_ADDR_LOOPBACK(buff));
  buff[15] = 0;

  EXPECT_FALSE(IN6_IS_ADDR_MULTICAST(buff));
  buff[0] = 0xff;
  EXPECT_TRUE(IN6_IS_ADDR_MULTICAST(buff));
  buff[0] = 0;

  buff[0] = 0xfe;
  buff[1] = 0x80;
  EXPECT_TRUE(IN6_IS_ADDR_LINKLOCAL(buff));
  buff[0] = 0xff;
  buff[1] = 0x80;
  EXPECT_FALSE(IN6_IS_ADDR_LINKLOCAL(buff));

  buff[0] = 0xfe;
  buff[1] = 0xc0;
  EXPECT_TRUE(IN6_IS_ADDR_SITELOCAL(buff));
  buff[0] = 0xff;
  buff[1] = 0x80;
  EXPECT_FALSE(IN6_IS_ADDR_SITELOCAL(buff));

  buff[0] = 0xff;
  buff[1] = 0x1;
  EXPECT_TRUE(IN6_IS_ADDR_MC_NODELOCAL(buff));
  buff[1] = 0x2;
  EXPECT_TRUE(IN6_IS_ADDR_MC_LINKLOCAL(buff));
  buff[1] = 0x5;
  EXPECT_TRUE(IN6_IS_ADDR_MC_SITELOCAL(buff));
  buff[1] = 0x8;
  EXPECT_TRUE(IN6_IS_ADDR_MC_ORGLOCAL(buff));
  buff[1] = 0xe;
  EXPECT_TRUE(IN6_IS_ADDR_MC_GLOBAL(buff));
  buff[1] = 0;
  buff[0] = 0;

  EXPECT_FALSE(IN6_IS_ADDR_V4MAPPED(buff));
  buff[10] = 0xff;
  buff[11] = 0xff;
  EXPECT_TRUE(IN6_IS_ADDR_V4MAPPED(buff));
  buff[10] = 0;
  buff[11] = 0;

  for (int i = 12; i < 16; ++i) {
    buff[i] ^= 42;
    EXPECT_TRUE(IN6_IS_ADDR_V4COMPAT(buff));
    buff[i] ^= 42;
  }
  for (int i = 0; i < 12; ++i) {
    buff[i] ^= 42;
    EXPECT_FALSE(IN6_IS_ADDR_V4COMPAT(buff));
    buff[i] ^= 42;
  }
}
