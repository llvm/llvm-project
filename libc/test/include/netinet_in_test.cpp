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
  struct in6_addr addr = {};
  const struct in6_addr *const_addr = &addr;

  EXPECT_TRUE(IN6_IS_ADDR_UNSPECIFIED(&addr));
  EXPECT_TRUE(IN6_IS_ADDR_UNSPECIFIED(const_addr));
  for (int i = 0; i < 16; ++i) {
    addr.s6_addr[i] = 1;
    EXPECT_FALSE(IN6_IS_ADDR_UNSPECIFIED(&addr));
    addr.s6_addr[i] = 0;
  }

  EXPECT_FALSE(IN6_IS_ADDR_LOOPBACK(&addr));
  addr.s6_addr[15] = 1;
  EXPECT_TRUE(IN6_IS_ADDR_LOOPBACK(&addr));
  addr.s6_addr[15] = 0;

  EXPECT_FALSE(IN6_IS_ADDR_MULTICAST(&addr));
  addr.s6_addr[0] = 0xff;
  EXPECT_TRUE(IN6_IS_ADDR_MULTICAST(&addr));
  addr.s6_addr[0] = 0;

  addr.s6_addr[0] = 0xfe;
  addr.s6_addr[1] = 0x80;
  EXPECT_TRUE(IN6_IS_ADDR_LINKLOCAL(&addr));
  addr.s6_addr[0] = 0xff;
  addr.s6_addr[1] = 0x80;
  EXPECT_FALSE(IN6_IS_ADDR_LINKLOCAL(&addr));

  addr.s6_addr[0] = 0xfe;
  addr.s6_addr[1] = 0xc0;
  EXPECT_TRUE(IN6_IS_ADDR_SITELOCAL(&addr));
  addr.s6_addr[0] = 0xff;
  addr.s6_addr[1] = 0x80;
  EXPECT_FALSE(IN6_IS_ADDR_SITELOCAL(&addr));

  addr.s6_addr[0] = 0xff;
  addr.s6_addr[1] = 0x1;
  EXPECT_TRUE(IN6_IS_ADDR_MC_NODELOCAL(&addr));
  addr.s6_addr[1] = 0x2;
  EXPECT_TRUE(IN6_IS_ADDR_MC_LINKLOCAL(&addr));
  addr.s6_addr[1] = 0x5;
  EXPECT_TRUE(IN6_IS_ADDR_MC_SITELOCAL(&addr));
  addr.s6_addr[1] = 0x8;
  EXPECT_TRUE(IN6_IS_ADDR_MC_ORGLOCAL(&addr));
  addr.s6_addr[1] = 0xe;
  EXPECT_TRUE(IN6_IS_ADDR_MC_GLOBAL(&addr));
  addr.s6_addr[1] = 0;
  addr.s6_addr[0] = 0;

  EXPECT_FALSE(IN6_IS_ADDR_V4MAPPED(&addr));
  addr.s6_addr[10] = 0xff;
  addr.s6_addr[11] = 0xff;
  EXPECT_TRUE(IN6_IS_ADDR_V4MAPPED(&addr));
  addr.s6_addr[10] = 0;
  addr.s6_addr[11] = 0;

  for (int i = 12; i < 16; ++i) {
    addr.s6_addr[i] ^= 42;
    EXPECT_TRUE(IN6_IS_ADDR_V4COMPAT(&addr));
    addr.s6_addr[i] ^= 42;
  }
  for (int i = 0; i < 12; ++i) {
    addr.s6_addr[i] ^= 42;
    EXPECT_FALSE(IN6_IS_ADDR_V4COMPAT(&addr));
    addr.s6_addr[i] ^= 42;
  }
}
