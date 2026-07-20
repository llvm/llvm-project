//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unittests for inet_ntop.
///
//===----------------------------------------------------------------------===//

#include "hdr/errno_macros.h"
#include "hdr/sys_socket_macros.h"
#include "hdr/types/struct_in6_addr.h"
#include "hdr/types/struct_in_addr.h"
#include "src/__support/endian_internal.h"
#include "src/arpa/inet/inet_ntop.h"
#include "test/UnitTest/ErrnoCheckingTest.h"
#include "test/UnitTest/Test.h"

using LlvmLibcInetNtopTest = LIBC_NAMESPACE::testing::ErrnoCheckingTest;

TEST_F(LlvmLibcInetNtopTest, InvalidFamily) {
  char buf[64];
  struct in_addr addr = {0};
  ASSERT_EQ(static_cast<const char *>(nullptr),
            LIBC_NAMESPACE::inet_ntop(AF_INET + AF_INET6 + 1, &addr, buf,
                                      sizeof(buf)));
  ASSERT_ERRNO_EQ(EAFNOSUPPORT);
}

static void *ipv4(uint8_t a, uint8_t b, uint8_t c, uint8_t d) {
  static struct in_addr addr;
  addr.s_addr = LIBC_NAMESPACE::Endian::to_big_endian(
      static_cast<uint32_t>(a) << 24 | static_cast<uint32_t>(b) << 16 |
      static_cast<uint32_t>(c) << 8 | static_cast<uint32_t>(d));
  return &addr;
}

static void *ipv6(uint16_t a, uint16_t b, uint16_t c, uint16_t d, uint16_t e,
                  uint16_t f, uint16_t g, uint16_t h) {
  static struct in6_addr addr;
  addr.s6_addr16[0] = LIBC_NAMESPACE::Endian::to_big_endian(a);
  addr.s6_addr16[1] = LIBC_NAMESPACE::Endian::to_big_endian(b);
  addr.s6_addr16[2] = LIBC_NAMESPACE::Endian::to_big_endian(c);
  addr.s6_addr16[3] = LIBC_NAMESPACE::Endian::to_big_endian(d);
  addr.s6_addr16[4] = LIBC_NAMESPACE::Endian::to_big_endian(e);
  addr.s6_addr16[5] = LIBC_NAMESPACE::Endian::to_big_endian(f);
  addr.s6_addr16[6] = LIBC_NAMESPACE::Endian::to_big_endian(g);
  addr.s6_addr16[7] = LIBC_NAMESPACE::Endian::to_big_endian(h);
  return &addr;
}

TEST_F(LlvmLibcInetNtopTest, IPv4Tests) {
  char buf[16];

  EXPECT_STREQ("127.0.0.1", LIBC_NAMESPACE::inet_ntop(
                                AF_INET, ipv4(127, 0, 0, 1), buf, sizeof(buf)));
  ASSERT_ERRNO_SUCCESS();

  // Test buffer too small (needs 10 bytes including null terminator)
  EXPECT_EQ(static_cast<const char *>(nullptr),
            LIBC_NAMESPACE::inet_ntop(AF_INET, ipv4(127, 0, 0, 1), buf, 9));
  ASSERT_ERRNO_EQ(ENOSPC);

  EXPECT_STREQ("127.0.0.1",
               LIBC_NAMESPACE::inet_ntop(AF_INET, ipv4(127, 0, 0, 1), buf, 10));
  ASSERT_ERRNO_SUCCESS();

  EXPECT_STREQ("255.255.255.255",
               LIBC_NAMESPACE::inet_ntop(AF_INET, ipv4(255, 255, 255, 255), buf,
                                         sizeof(buf)));
  ASSERT_ERRNO_SUCCESS();

  // Test buffer too small (needs 16 bytes)
  EXPECT_EQ(
      static_cast<const char *>(nullptr),
      LIBC_NAMESPACE::inet_ntop(AF_INET, ipv4(255, 255, 255, 255), buf, 15));
  ASSERT_ERRNO_EQ(ENOSPC);

  // Boundary conditions for the number of digits.
  EXPECT_STREQ("0.0.0.0",
               LIBC_NAMESPACE::inet_ntop(AF_INET, ipv4(0, 0, 0, 0), buf, 10));
  ASSERT_ERRNO_SUCCESS();

  EXPECT_STREQ("0.0.0.1",
               LIBC_NAMESPACE::inet_ntop(AF_INET, ipv4(0, 0, 0, 1), buf, 10));
  ASSERT_ERRNO_SUCCESS();

  EXPECT_STREQ("0.0.0.9",
               LIBC_NAMESPACE::inet_ntop(AF_INET, ipv4(0, 0, 0, 9), buf, 10));
  ASSERT_ERRNO_SUCCESS();

  EXPECT_STREQ("0.0.0.10",
               LIBC_NAMESPACE::inet_ntop(AF_INET, ipv4(0, 0, 0, 10), buf, 11));
  ASSERT_ERRNO_SUCCESS();

  EXPECT_STREQ("0.0.0.11",
               LIBC_NAMESPACE::inet_ntop(AF_INET, ipv4(0, 0, 0, 11), buf, 11));
  ASSERT_ERRNO_SUCCESS();

  EXPECT_STREQ("0.0.0.99",
               LIBC_NAMESPACE::inet_ntop(AF_INET, ipv4(0, 0, 0, 99), buf, 11));
  ASSERT_ERRNO_SUCCESS();

  EXPECT_STREQ("0.0.0.100",
               LIBC_NAMESPACE::inet_ntop(AF_INET, ipv4(0, 0, 0, 100), buf, 12));
  ASSERT_ERRNO_SUCCESS();
}

TEST_F(LlvmLibcInetNtopTest, IPv6Tests) {
  char buf[64];

  // No compression
  EXPECT_STREQ("1:2:3:4:5:6:7:8",
               LIBC_NAMESPACE::inet_ntop(AF_INET6, ipv6(1, 2, 3, 4, 5, 6, 7, 8),
                                         buf, sizeof(buf)));
  ASSERT_ERRNO_SUCCESS();

  // Compression, fully compressed.
  EXPECT_STREQ(static_cast<const char *>(nullptr),
               LIBC_NAMESPACE::inet_ntop(AF_INET6, ipv6(0, 0, 0, 0, 0, 0, 0, 0),
                                         buf, 2));
  ASSERT_ERRNO_EQ(ENOSPC);

  EXPECT_STREQ("::", LIBC_NAMESPACE::inet_ntop(
                         AF_INET6, ipv6(0, 0, 0, 0, 0, 0, 0, 0), buf, 3));
  ASSERT_ERRNO_SUCCESS();

  // Leading block of zeroes.
  EXPECT_STREQ("::1", LIBC_NAMESPACE::inet_ntop(
                          AF_INET6, ipv6(0, 0, 0, 0, 0, 0, 0, 1), buf, 4));
  ASSERT_ERRNO_SUCCESS();

  EXPECT_EQ(static_cast<const char *>(nullptr),
            LIBC_NAMESPACE::inet_ntop(AF_INET6, ipv6(0, 0, 0, 0, 0, 0, 0, 1),
                                      buf, 3));
  ASSERT_ERRNO_EQ(ENOSPC);

  EXPECT_STREQ("::1:2:3", LIBC_NAMESPACE::inet_ntop(
                              AF_INET6, ipv6(0, 0, 0, 0, 0, 1, 2, 3), buf, 8));
  ASSERT_ERRNO_SUCCESS();

  // Trailing block of zeroes.
  EXPECT_STREQ("1::", LIBC_NAMESPACE::inet_ntop(
                          AF_INET6, ipv6(1, 0, 0, 0, 0, 0, 0, 0), buf, 4));
  ASSERT_ERRNO_SUCCESS();

  EXPECT_STREQ("1:2::", LIBC_NAMESPACE::inet_ntop(
                            AF_INET6, ipv6(1, 2, 0, 0, 0, 0, 0, 0), buf, 6));
  ASSERT_ERRNO_SUCCESS();

  // Middle of string.
  EXPECT_STREQ("2001:db8::1",
               LIBC_NAMESPACE::inet_ntop(AF_INET6,
                                         ipv6(0x2001, 0x0db8, 0, 0, 0, 0, 0, 1),
                                         buf, sizeof(buf)));
  ASSERT_ERRNO_SUCCESS();

  EXPECT_STREQ("2001:db8:0:1::1",
               LIBC_NAMESPACE::inet_ntop(AF_INET6,
                                         ipv6(0x2001, 0x0db8, 0, 1, 0, 0, 0, 1),
                                         buf, sizeof(buf)));
  ASSERT_ERRNO_SUCCESS();

  // Longer block wins.
  EXPECT_STREQ("2001:0:0:1::1", LIBC_NAMESPACE::inet_ntop(
                                    AF_INET6, ipv6(0x2001, 0, 0, 1, 0, 0, 0, 1),
                                    buf, sizeof(buf)));
  ASSERT_ERRNO_SUCCESS();

  EXPECT_STREQ("0:0:1:1::",
               LIBC_NAMESPACE::inet_ntop(AF_INET6, ipv6(0, 0, 1, 1, 0, 0, 0, 0),
                                         buf, sizeof(buf)));
  ASSERT_ERRNO_SUCCESS();

  // In case of ties, the first block wins.
  EXPECT_STREQ("::1:0:0:1:0:0",
               LIBC_NAMESPACE::inet_ntop(AF_INET6, ipv6(0, 0, 1, 0, 0, 1, 0, 0),
                                         buf, sizeof(buf)));
  ASSERT_ERRNO_SUCCESS();

  // A single zero is not compressed, no matter it's position.
  EXPECT_STREQ("0:1:1:1:1:1:1:1",
               LIBC_NAMESPACE::inet_ntop(AF_INET6, ipv6(0, 1, 1, 1, 1, 1, 1, 1),
                                         buf, sizeof(buf)));
  ASSERT_ERRNO_SUCCESS();

  EXPECT_STREQ("1:0:1:1:1:1:1:1",
               LIBC_NAMESPACE::inet_ntop(AF_INET6, ipv6(1, 0, 1, 1, 1, 1, 1, 1),
                                         buf, sizeof(buf)));
  ASSERT_ERRNO_SUCCESS();

  EXPECT_STREQ("1:1:1:1:1:1:1:0",
               LIBC_NAMESPACE::inet_ntop(AF_INET6, ipv6(1, 1, 1, 1, 1, 1, 1, 0),
                                         buf, sizeof(buf)));
  ASSERT_ERRNO_SUCCESS();

  // V4 mapped addresses.
  EXPECT_STREQ("::ffff:192.168.0.1",
               LIBC_NAMESPACE::inet_ntop(
                   AF_INET6, ipv6(0, 0, 0, 0, 0, 0xffff, 0xc0a8, 1), buf, 19));
  ASSERT_ERRNO_SUCCESS();

  EXPECT_STREQ("::192.168.0.1", LIBC_NAMESPACE::inet_ntop(
                                    AF_INET6, ipv6(0, 0, 0, 0, 0, 0, 0xc0a8, 1),
                                    buf, sizeof(buf)));
  ASSERT_ERRNO_SUCCESS();

  // If only the last component is set, we don't v4-map.
  EXPECT_STREQ("::dead", LIBC_NAMESPACE::inet_ntop(
                             AF_INET6, ipv6(0, 0, 0, 0, 0, 0, 0, 0xdead), buf,
                             sizeof(buf)));
  ASSERT_ERRNO_SUCCESS();

  // But we do for the penultimate.
  EXPECT_STREQ("::10.13.0.0", LIBC_NAMESPACE::inet_ntop(
                                  AF_INET6, ipv6(0, 0, 0, 0, 0, 0, 0x0a0d, 0),
                                  buf, sizeof(buf)));
  ASSERT_ERRNO_SUCCESS();

  // Edge cases for numbers of digits.
  EXPECT_STREQ("0:1::", LIBC_NAMESPACE::inet_ntop(
                            AF_INET6, ipv6(0, 1, 0, 0, 0, 0, 0, 0), buf, 6));
  ASSERT_ERRNO_SUCCESS();

  EXPECT_STREQ("1:1::", LIBC_NAMESPACE::inet_ntop(
                            AF_INET6, ipv6(1, 1, 0, 0, 0, 0, 0, 0), buf, 6));
  ASSERT_ERRNO_SUCCESS();

  EXPECT_STREQ("2:1::", LIBC_NAMESPACE::inet_ntop(
                            AF_INET6, ipv6(2, 1, 0, 0, 0, 0, 0, 0), buf, 6));
  ASSERT_ERRNO_SUCCESS();

  EXPECT_STREQ("4:1::", LIBC_NAMESPACE::inet_ntop(
                            AF_INET6, ipv6(4, 1, 0, 0, 0, 0, 0, 0), buf, 6));
  ASSERT_ERRNO_SUCCESS();

  EXPECT_STREQ("8:1::", LIBC_NAMESPACE::inet_ntop(
                            AF_INET6, ipv6(8, 1, 0, 0, 0, 0, 0, 0), buf, 6));
  ASSERT_ERRNO_SUCCESS();

  EXPECT_STREQ("10:1::",
               LIBC_NAMESPACE::inet_ntop(
                   AF_INET6, ipv6(0x10, 1, 0, 0, 0, 0, 0, 0), buf, 7));
  ASSERT_ERRNO_SUCCESS();

  EXPECT_STREQ("ff:1::",
               LIBC_NAMESPACE::inet_ntop(
                   AF_INET6, ipv6(0xff, 1, 0, 0, 0, 0, 0, 0), buf, 7));
  ASSERT_ERRNO_SUCCESS();

  EXPECT_STREQ("100:1::",
               LIBC_NAMESPACE::inet_ntop(
                   AF_INET6, ipv6(0x100, 1, 0, 0, 0, 0, 0, 0), buf, 8));
  ASSERT_ERRNO_SUCCESS();

  EXPECT_STREQ("fff:1::",
               LIBC_NAMESPACE::inet_ntop(
                   AF_INET6, ipv6(0xfff, 1, 0, 0, 0, 0, 0, 0), buf, 8));
  ASSERT_ERRNO_SUCCESS();

  EXPECT_STREQ("1000:1::",
               LIBC_NAMESPACE::inet_ntop(
                   AF_INET6, ipv6(0x1000, 1, 0, 0, 0, 0, 0, 0), buf, 9));
  ASSERT_ERRNO_SUCCESS();

  // Maximum length address.
  EXPECT_STREQ("ffff:ffff:ffff:ffff:ffff:ffff:ffff:ffff",
               LIBC_NAMESPACE::inet_ntop(AF_INET6,
                                         ipv6(0xffff, 0xffff, 0xffff, 0xffff,
                                              0xffff, 0xffff, 0xffff, 0xffff),
                                         buf, 40));
  ASSERT_ERRNO_SUCCESS();

  EXPECT_EQ(static_cast<const char *>(nullptr),
            LIBC_NAMESPACE::inet_ntop(AF_INET6,
                                      ipv6(0xffff, 0xffff, 0xffff, 0xffff,
                                           0xffff, 0xffff, 0xffff, 0xffff),
                                      buf, 39));
  ASSERT_ERRNO_EQ(ENOSPC);

  // IPv4-mapped max length
  EXPECT_STREQ(
      "::ffff:255.255.255.255",
      LIBC_NAMESPACE::inet_ntop(
          AF_INET6, ipv6(0, 0, 0, 0, 0, 0xffff, 0xffff, 0xffff), buf, 23));
  ASSERT_ERRNO_SUCCESS();

  EXPECT_EQ(
      static_cast<const char *>(nullptr),
      LIBC_NAMESPACE::inet_ntop(
          AF_INET6, ipv6(0, 0, 0, 0, 0, 0xffff, 0xffff, 0xffff), buf, 22));
  ASSERT_ERRNO_EQ(ENOSPC);
}
