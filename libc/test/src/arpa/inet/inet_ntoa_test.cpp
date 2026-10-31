//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unittests for inet_ntoa.
///
//===----------------------------------------------------------------------===//

#include "hdr/types/struct_in_addr.h"
#include "src/__support/endian_internal.h"
#include "src/arpa/inet/inet_ntoa.h"
#include "test/UnitTest/Test.h"

static struct in_addr make_addr(uint8_t a, uint8_t b, uint8_t c, uint8_t d) {
  struct in_addr addr;
  addr.s_addr = LIBC_NAMESPACE::Endian::to_big_endian(
      static_cast<uint32_t>(a) << 24 | static_cast<uint32_t>(b) << 16 |
      static_cast<uint32_t>(c) << 8 | static_cast<uint32_t>(d));
  return addr;
}

TEST(LlvmLibcInetNtoaTest, BasicConversion) {
  EXPECT_STREQ("127.0.0.1", LIBC_NAMESPACE::inet_ntoa(make_addr(127, 0, 0, 1)));
  EXPECT_STREQ("0.0.0.0", LIBC_NAMESPACE::inet_ntoa(make_addr(0, 0, 0, 0)));
  EXPECT_STREQ("255.255.255.255",
               LIBC_NAMESPACE::inet_ntoa(make_addr(255, 255, 255, 255)));
  EXPECT_STREQ("192.168.1.100",
               LIBC_NAMESPACE::inet_ntoa(make_addr(192, 168, 1, 100)));
  EXPECT_STREQ("10.0.0.1", LIBC_NAMESPACE::inet_ntoa(make_addr(10, 0, 0, 1)));
}

TEST(LlvmLibcInetNtoaTest, BufferReuseAndOverwrite) {
  char *ptr1 = LIBC_NAMESPACE::inet_ntoa(make_addr(1, 2, 3, 4));
  EXPECT_STREQ("1.2.3.4", ptr1);

  char *ptr2 = LIBC_NAMESPACE::inet_ntoa(make_addr(5, 6, 7, 8));
  // Our implementation returns pointer to the same static buffer
  EXPECT_EQ(ptr1, ptr2);
  EXPECT_STREQ("5.6.7.8", ptr1);
  EXPECT_STREQ("5.6.7.8", ptr2);
}
