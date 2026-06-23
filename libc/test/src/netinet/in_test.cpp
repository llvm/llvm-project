//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unittests for netinet/in.h
///
//===----------------------------------------------------------------------===//

#include "src/arpa/inet/htonl.h"
#include "src/arpa/inet/htons.h"
#include "src/string/memcmp.h"
#include "test/UnitTest/Test.h"

#include "hdr/netinet_in_macros.h"
#include "hdr/types/struct_in6_addr.h"
#include "hdr/types/struct_sockaddr_in6.h"
#include "src/netinet/in6addr_any.h"
#include "src/netinet/in6addr_loopback.h"

TEST(LlvmLibcNetinetInTest, In6AddrLayout) {
  EXPECT_EQ(sizeof(struct in6_addr), static_cast<size_t>(16));

  struct in6_addr addr = {};

  // Using parenthesis to suppress htons-as-a-macro in overlay mode.
  addr.s6_addr16[0] = (LIBC_NAMESPACE::htons)(0x1122);
  addr.s6_addr16[1] = (LIBC_NAMESPACE::htons)(0x3344);
  addr.s6_addr16[2] = (LIBC_NAMESPACE::htons)(0x5566);
  addr.s6_addr16[3] = (LIBC_NAMESPACE::htons)(0x7788);
  addr.s6_addr16[4] = (LIBC_NAMESPACE::htons)(0x99aa);
  addr.s6_addr16[5] = (LIBC_NAMESPACE::htons)(0xbbcc);
  addr.s6_addr16[6] = (LIBC_NAMESPACE::htons)(0xddee);
  addr.s6_addr16[7] = (LIBC_NAMESPACE::htons)(0xff00);

  EXPECT_EQ(
      LIBC_NAMESPACE::memcmp(
          &addr,
          "\x11\x22\x33\x44\x55\x66\x77\x88\x99\xaa\xbb\xcc\xdd\xee\xff\x00",
          16),
      0);

  addr.s6_addr32[0] = (LIBC_NAMESPACE::htonl)(0x12345678);
  addr.s6_addr32[1] = (LIBC_NAMESPACE::htonl)(0x9abcdef0);
  addr.s6_addr32[2] = (LIBC_NAMESPACE::htonl)(0x12345678);
  addr.s6_addr32[3] = (LIBC_NAMESPACE::htonl)(0x9abcdef0);

  EXPECT_EQ(
      LIBC_NAMESPACE::memcmp(
          &addr,
          "\x12\x34\x56\x78\x9a\xbc\xde\xf0\x12\x34\x56\x78\x9a\xbc\xde\xf0",
          16),
      0);
}

TEST(LlvmLibcNetinetInTest, IN6AddrInitMacros) {
  struct in6_addr any = IN6ADDR_ANY_INIT;
  const uint8_t ANY_CONTENT[16] = {0};
  EXPECT_EQ(LIBC_NAMESPACE::memcmp(&any, ANY_CONTENT, 16), 0);
  EXPECT_TRUE(IN6_IS_ADDR_UNSPECIFIED(&any));

  struct in6_addr loopback = IN6ADDR_LOOPBACK_INIT;
  const uint8_t LOOPBACK_CONTENT[16] = {0, 0, 0, 0, 0, 0, 0, 0,
                                        0, 0, 0, 0, 0, 0, 0, 1};
  EXPECT_EQ(LIBC_NAMESPACE::memcmp(&loopback, LOOPBACK_CONTENT, 16), 0);
  EXPECT_TRUE(IN6_IS_ADDR_LOOPBACK(&loopback));
}

TEST(LlvmLibcNetinetInTest, IN6AddrConstants) {
  const uint8_t ANY_CONTENT[16] = {0};
  EXPECT_EQ(
      LIBC_NAMESPACE::memcmp(&LIBC_NAMESPACE::in6addr_any, ANY_CONTENT, 16), 0);

  const uint8_t LOOPBACK_CONTENT[16] = {0, 0, 0, 0, 0, 0, 0, 0,
                                        0, 0, 0, 0, 0, 0, 0, 1};
  EXPECT_EQ(LIBC_NAMESPACE::memcmp(&LIBC_NAMESPACE::in6addr_loopback,
                                   LOOPBACK_CONTENT, 16),
            0);
}

TEST(LlvmLibcNetinetInTest, SockaddrIn6Layout) {
  EXPECT_EQ(offsetof(struct sockaddr_in6, sin6_family), static_cast<size_t>(0));
  EXPECT_EQ(offsetof(struct sockaddr_in6, sin6_port), static_cast<size_t>(2));
  EXPECT_EQ(offsetof(struct sockaddr_in6, sin6_flowinfo),
            static_cast<size_t>(4));
  EXPECT_EQ(offsetof(struct sockaddr_in6, sin6_addr), static_cast<size_t>(8));
  EXPECT_EQ(offsetof(struct sockaddr_in6, sin6_scope_id),
            static_cast<size_t>(24));
  EXPECT_EQ(sizeof(struct sockaddr_in6), static_cast<size_t>(28));
}
