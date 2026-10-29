//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unittests for ether_ntoa_r.
///
//===----------------------------------------------------------------------===//

#include "hdr/types/struct_ether_addr.h"
#include "src/netinet/ether/ether_ntoa_r.h"
#include "test/UnitTest/Test.h"

TEST(LlvmLibcEtherNtoaRTest, BasicConversion) {
  char buf[18];

  struct ether_addr zeroes = {{0x00, 0x00, 0x00, 0x00, 0x00, 0x00}};
  EXPECT_EQ(buf, LIBC_NAMESPACE::ether_ntoa_r(&zeroes, buf));
  EXPECT_STREQ("0:0:0:0:0:0", buf);

  struct ether_addr broadcast = {{0xff, 0xff, 0xff, 0xff, 0xff, 0xff}};
  EXPECT_EQ(buf, LIBC_NAMESPACE::ether_ntoa_r(&broadcast, buf));
  EXPECT_STREQ("ff:ff:ff:ff:ff:ff", buf);

  struct ether_addr mixed1 = {{0x01, 0x23, 0x45, 0x67, 0x89, 0xab}};
  EXPECT_EQ(buf, LIBC_NAMESPACE::ether_ntoa_r(&mixed1, buf));
  EXPECT_STREQ("1:23:45:67:89:ab", buf);

  struct ether_addr mixed2 = {{0xcd, 0xef, 0x0a, 0x10, 0x00, 0x9f}};
  EXPECT_EQ(buf, LIBC_NAMESPACE::ether_ntoa_r(&mixed2, buf));
  EXPECT_STREQ("cd:ef:a:10:0:9f", buf);
}
