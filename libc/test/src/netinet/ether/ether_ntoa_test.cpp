//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unittests for ether_ntoa.
///
//===----------------------------------------------------------------------===//

#include "hdr/types/struct_ether_addr.h"
#include "src/netinet/ether/ether_ntoa.h"
#include "test/UnitTest/Test.h"

TEST(LlvmLibcEtherNtoaTest, BasicConversion) {
  struct ether_addr zeroes = {{0x00, 0x00, 0x00, 0x00, 0x00, 0x00}};
  EXPECT_STREQ("0:0:0:0:0:0", LIBC_NAMESPACE::ether_ntoa(&zeroes));

  struct ether_addr broadcast = {{0xff, 0xff, 0xff, 0xff, 0xff, 0xff}};
  EXPECT_STREQ("ff:ff:ff:ff:ff:ff", LIBC_NAMESPACE::ether_ntoa(&broadcast));

  struct ether_addr mixed1 = {{0x01, 0x23, 0x45, 0x67, 0x89, 0xab}};
  EXPECT_STREQ("1:23:45:67:89:ab", LIBC_NAMESPACE::ether_ntoa(&mixed1));

  struct ether_addr mixed2 = {{0xcd, 0xef, 0x0a, 0x10, 0x00, 0x9f}};
  EXPECT_STREQ("cd:ef:a:10:0:9f", LIBC_NAMESPACE::ether_ntoa(&mixed2));
}

TEST(LlvmLibcEtherNtoaTest, BufferReuseAndOverwrite) {
  struct ether_addr addr1 = {{0x11, 0x22, 0x33, 0x44, 0x55, 0x66}};
  char *ptr1 = LIBC_NAMESPACE::ether_ntoa(&addr1);
  EXPECT_STREQ("11:22:33:44:55:66", ptr1);

  struct ether_addr addr2 = {{0x01, 0x02, 0x03, 0x04, 0x05, 0x06}};
  char *ptr2 = LIBC_NAMESPACE::ether_ntoa(&addr2);
  EXPECT_EQ(ptr1, ptr2);
  EXPECT_STREQ("1:2:3:4:5:6", ptr1);
  EXPECT_STREQ("1:2:3:4:5:6", ptr2);
}
