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

#include "hdr/types/struct_in6_addr.h"

TEST(LlvmLibcNetinetInTest, In6AddrLayout) {
  EXPECT_EQ(sizeof(struct in6_addr), size_t(16));

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
