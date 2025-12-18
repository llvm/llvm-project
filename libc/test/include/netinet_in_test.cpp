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
}
