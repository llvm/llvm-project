//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unittests for inet_pton.
///
//===----------------------------------------------------------------------===//

#include "hdr/errno_macros.h"
#include "hdr/sys_socket_macros.h"
#include "hdr/types/struct_in6_addr.h"
#include "hdr/types/struct_in_addr.h"
#include "src/__support/endian_internal.h"
#include "src/__support/libc_errno.h"
#include "src/arpa/inet/inet_pton.h"
#include "test/UnitTest/ErrnoCheckingTest.h"
#include "test/UnitTest/Test.h"

using LlvmLibcInetPtonTest = LIBC_NAMESPACE::testing::ErrnoCheckingTest;

static uint32_t ipv4_bits(uint8_t a, uint8_t b, uint8_t c, uint8_t d) {
  return LIBC_NAMESPACE::Endian::to_big_endian(
      static_cast<uint32_t>(a) << 24 | static_cast<uint32_t>(b) << 16 |
      static_cast<uint32_t>(c) << 8 | static_cast<uint32_t>(d));
}

TEST_F(LlvmLibcInetPtonTest, ValidIPv4Addresses) {
  struct in_addr addr;

  EXPECT_EQ(1, LIBC_NAMESPACE::inet_pton(AF_INET, "127.0.0.1", &addr));
  EXPECT_EQ(addr.s_addr, ipv4_bits(127, 0, 0, 1));
  ASSERT_ERRNO_SUCCESS();

  EXPECT_EQ(1, LIBC_NAMESPACE::inet_pton(AF_INET, "192.168.1.254", &addr));
  EXPECT_EQ(addr.s_addr, ipv4_bits(192, 168, 1, 254));
  ASSERT_ERRNO_SUCCESS();

  EXPECT_EQ(1, LIBC_NAMESPACE::inet_pton(AF_INET, "0.0.0.0", &addr));
  EXPECT_EQ(addr.s_addr, ipv4_bits(0, 0, 0, 0));
  ASSERT_ERRNO_SUCCESS();

  EXPECT_EQ(1, LIBC_NAMESPACE::inet_pton(AF_INET, "255.255.255.255", &addr));
  EXPECT_EQ(addr.s_addr, ipv4_bits(255, 255, 255, 255));
  ASSERT_ERRNO_SUCCESS();
}

TEST_F(LlvmLibcInetPtonTest, InvalidOctetValues) {
  struct in_addr addr;

  EXPECT_EQ(0, LIBC_NAMESPACE::inet_pton(AF_INET, "256.0.0.1", &addr));
  EXPECT_EQ(0, LIBC_NAMESPACE::inet_pton(AF_INET, "192.168.1.300", &addr));
  EXPECT_EQ(0, LIBC_NAMESPACE::inet_pton(AF_INET, "192.168.-1.1", &addr));
  ASSERT_ERRNO_SUCCESS();
}

TEST_F(LlvmLibcInetPtonTest, InvalidFormats) {
  struct in_addr addr;

  EXPECT_EQ(0, LIBC_NAMESPACE::inet_pton(AF_INET, "127.0.0", &addr));
  EXPECT_EQ(0, LIBC_NAMESPACE::inet_pton(AF_INET, "10.0.0.0.1", &addr));
  EXPECT_EQ(0, LIBC_NAMESPACE::inet_pton(AF_INET, ".1.2.3.4", &addr));
  EXPECT_EQ(0, LIBC_NAMESPACE::inet_pton(AF_INET, "1.2.3.4.", &addr));
  EXPECT_EQ(0, LIBC_NAMESPACE::inet_pton(AF_INET, "192..168.1.1", &addr));
  EXPECT_EQ(0, LIBC_NAMESPACE::inet_pton(AF_INET, "192.168.1.1a", &addr));
  EXPECT_EQ(0, LIBC_NAMESPACE::inet_pton(AF_INET, "abc.def.ghi.jkl", &addr));
  EXPECT_EQ(0, LIBC_NAMESPACE::inet_pton(AF_INET, "127.0.0.1 ", &addr));
  EXPECT_EQ(0, LIBC_NAMESPACE::inet_pton(AF_INET, "", &addr));
  EXPECT_EQ(0, LIBC_NAMESPACE::inet_pton(AF_INET, ".", &addr));
  EXPECT_EQ(0, LIBC_NAMESPACE::inet_pton(AF_INET, "0", &addr));
  EXPECT_EQ(0, LIBC_NAMESPACE::inet_pton(AF_INET, "1", &addr));
  EXPECT_EQ(0, LIBC_NAMESPACE::inet_pton(AF_INET, ".1", &addr));
  EXPECT_EQ(0, LIBC_NAMESPACE::inet_pton(AF_INET, "1.1", &addr));
  EXPECT_EQ(0, LIBC_NAMESPACE::inet_pton(AF_INET, "1.1.", &addr));
  EXPECT_EQ(0, LIBC_NAMESPACE::inet_pton(AF_INET, "1.1.1", &addr));
  EXPECT_EQ(0, LIBC_NAMESPACE::inet_pton(AF_INET, "1.1.1.", &addr));
  ASSERT_ERRNO_SUCCESS();
}

TEST_F(LlvmLibcInetPtonTest, StrictPosixLeadingZeros) {
  struct in_addr addr;

  // inet_pton must reject octal-style leading zeros
  EXPECT_EQ(0, LIBC_NAMESPACE::inet_pton(AF_INET, "192.168.01.1", &addr));
  EXPECT_EQ(0, LIBC_NAMESPACE::inet_pton(AF_INET, "010.0.0.1", &addr));
  EXPECT_EQ(0, LIBC_NAMESPACE::inet_pton(AF_INET, "00.0.0.0", &addr));
  ASSERT_ERRNO_SUCCESS();
}

static bool check_ipv6(const struct in6_addr &addr, uint16_t a, uint16_t b,
                       uint16_t c, uint16_t d, uint16_t e, uint16_t f,
                       uint16_t g, uint16_t h) {
  uint16_t expected[8] = {a, b, c, d, e, f, g, h};
  for (size_t i = 0; i < 8; ++i) {
    uint16_t actual = static_cast<uint16_t>(
        (static_cast<uint16_t>(addr.s6_addr[2 * i]) << 8) |
        static_cast<uint16_t>(addr.s6_addr[2 * i + 1]));
    if (actual != expected[i])
      return false;
  }
  return true;
}

TEST_F(LlvmLibcInetPtonTest, ValidIPv6Addresses) {
  struct in6_addr addr;

  // Unspecified address
  EXPECT_EQ(1, LIBC_NAMESPACE::inet_pton(AF_INET6, "::", &addr));
  EXPECT_TRUE(check_ipv6(addr, 0, 0, 0, 0, 0, 0, 0, 0));
  ASSERT_ERRNO_SUCCESS();

  // Loopback address
  EXPECT_EQ(1, LIBC_NAMESPACE::inet_pton(AF_INET6, "::1", &addr));
  EXPECT_TRUE(check_ipv6(addr, 0, 0, 0, 0, 0, 0, 0, 1));
  ASSERT_ERRNO_SUCCESS();

  // Full 8 groups without compression
  EXPECT_EQ(1, LIBC_NAMESPACE::inet_pton(
                   AF_INET6, "2001:0db8:85a3:0000:0000:8a2e:0370:7334", &addr));
  EXPECT_TRUE(
      check_ipv6(addr, 0x2001, 0x0db8, 0x85a3, 0, 0, 0x8a2e, 0x0370, 0x7334));
  ASSERT_ERRNO_SUCCESS();

  // Full 8 groups with varying digit lengths
  EXPECT_EQ(1, LIBC_NAMESPACE::inet_pton(
                   AF_INET6, "2001:db8:85a3:0:0:8a2e:370:7334", &addr));
  EXPECT_TRUE(
      check_ipv6(addr, 0x2001, 0x0db8, 0x85a3, 0, 0, 0x8a2e, 0x0370, 0x7334));
  ASSERT_ERRNO_SUCCESS();

  // Compression at beginning
  EXPECT_EQ(1, LIBC_NAMESPACE::inet_pton(AF_INET6, "::1234", &addr));
  EXPECT_TRUE(check_ipv6(addr, 0, 0, 0, 0, 0, 0, 0, 0x1234));
  ASSERT_ERRNO_SUCCESS();

  // Compression at end
  EXPECT_EQ(1, LIBC_NAMESPACE::inet_pton(AF_INET6, "fe80::", &addr));
  EXPECT_TRUE(check_ipv6(addr, 0xfe80, 0, 0, 0, 0, 0, 0, 0));
  ASSERT_ERRNO_SUCCESS();

  // Compression in middle
  EXPECT_EQ(1, LIBC_NAMESPACE::inet_pton(AF_INET6, "2001:db8::1", &addr));
  EXPECT_TRUE(check_ipv6(addr, 0x2001, 0x0db8, 0, 0, 0, 0, 0, 1));
  ASSERT_ERRNO_SUCCESS();

  // Multiple words before and after compression
  EXPECT_EQ(1, LIBC_NAMESPACE::inet_pton(
                   AF_INET6, "2001:db8:85a3::8a2e:370:7334", &addr));
  EXPECT_TRUE(
      check_ipv6(addr, 0x2001, 0x0db8, 0x85a3, 0, 0, 0x8a2e, 0x0370, 0x7334));
  ASSERT_ERRNO_SUCCESS();

  // Case insensitivity
  EXPECT_EQ(1, LIBC_NAMESPACE::inet_pton(AF_INET6, "2001:Db8::AbCd", &addr));
  EXPECT_TRUE(check_ipv6(addr, 0x2001, 0x0db8, 0, 0, 0, 0, 0, 0xabcd));
  ASSERT_ERRNO_SUCCESS();
}

TEST_F(LlvmLibcInetPtonTest, ValidIPv4MappedIPv6Addresses) {
  struct in6_addr addr;

  // Standard IPv4-mapped address ::ffff:192.168.1.1
  EXPECT_EQ(1,
            LIBC_NAMESPACE::inet_pton(AF_INET6, "::ffff:192.168.1.1", &addr));
  EXPECT_TRUE(check_ipv6(addr, 0, 0, 0, 0, 0, 0xffff, 0xc0a8, 0x0101));
  ASSERT_ERRNO_SUCCESS();

  // IPv4-compatible address ::192.168.1.1
  EXPECT_EQ(1, LIBC_NAMESPACE::inet_pton(AF_INET6, "::192.168.1.1", &addr));
  EXPECT_TRUE(check_ipv6(addr, 0, 0, 0, 0, 0, 0, 0xc0a8, 0x0101));
  ASSERT_ERRNO_SUCCESS();

  // Mixed prefix with compression: 2001:db8::192.168.1.1
  EXPECT_EQ(
      1, LIBC_NAMESPACE::inet_pton(AF_INET6, "2001:db8::192.168.1.1", &addr));
  EXPECT_TRUE(check_ipv6(addr, 0x2001, 0x0db8, 0, 0, 0, 0, 0xc0a8, 0x0101));
  ASSERT_ERRNO_SUCCESS();

  // Full 6 words without compression followed by IPv4
  EXPECT_EQ(
      1, LIBC_NAMESPACE::inet_pton(AF_INET6, "0:0:0:0:0:0:192.168.1.1", &addr));
  EXPECT_TRUE(check_ipv6(addr, 0, 0, 0, 0, 0, 0, 0xc0a8, 0x0101));
  ASSERT_ERRNO_SUCCESS();
}

TEST_F(LlvmLibcInetPtonTest, InvalidIPv6Formats) {
  struct in6_addr addr;

  // Empty string
  EXPECT_EQ(0, LIBC_NAMESPACE::inet_pton(AF_INET6, "", &addr));

  // Colon misuse
  EXPECT_EQ(0, LIBC_NAMESPACE::inet_pton(AF_INET6, ":", &addr));
  EXPECT_EQ(0, LIBC_NAMESPACE::inet_pton(AF_INET6, ":::", &addr));
  EXPECT_EQ(0, LIBC_NAMESPACE::inet_pton(AF_INET6, ":1:2:3:4:5:6:7:8", &addr));
  EXPECT_EQ(0, LIBC_NAMESPACE::inet_pton(AF_INET6, "1:2:3:4:5:6:7:8:", &addr));
  EXPECT_EQ(0, LIBC_NAMESPACE::inet_pton(AF_INET6, "1::2:", &addr));
  EXPECT_EQ(0, LIBC_NAMESPACE::inet_pton(AF_INET6, "1::2::3", &addr));

  // Word count errors
  EXPECT_EQ(0, LIBC_NAMESPACE::inet_pton(AF_INET6, "1:2:3:4:5:6:7:8:9", &addr));
  EXPECT_EQ(0, LIBC_NAMESPACE::inet_pton(AF_INET6, "1:2:3:4::5:6:7:8", &addr));
  EXPECT_EQ(0, LIBC_NAMESPACE::inet_pton(AF_INET6, "1:2:3:4:5:6:7", &addr));

  // Field overflow (> 4 hex digits)
  EXPECT_EQ(0, LIBC_NAMESPACE::inet_pton(AF_INET6, "12345::1", &addr));
  EXPECT_EQ(0, LIBC_NAMESPACE::inet_pton(AF_INET6, "00000::1", &addr));

  // Non-hex characters
  EXPECT_EQ(0, LIBC_NAMESPACE::inet_pton(AF_INET6, "2001:xyz::1", &addr));
  EXPECT_EQ(0, LIBC_NAMESPACE::inet_pton(AF_INET6, "fe80::1%eth0", &addr));

  // Plain IPv4 without ::
  EXPECT_EQ(0, LIBC_NAMESPACE::inet_pton(AF_INET6, "192.168.1.1", &addr));

  // Embedded IPv4 errors
  EXPECT_EQ(0, LIBC_NAMESPACE::inet_pton(AF_INET6, "::ffff:192.168.1", &addr));
  EXPECT_EQ(0,
            LIBC_NAMESPACE::inet_pton(AF_INET6, "::ffff:192.168.1.256", &addr));
  EXPECT_EQ(0,
            LIBC_NAMESPACE::inet_pton(AF_INET6, "::ffff:192.168.1.1:", &addr));
  EXPECT_EQ(0,
            LIBC_NAMESPACE::inet_pton(AF_INET6, "::ffff:192.168.01.1", &addr));
  EXPECT_EQ(0, LIBC_NAMESPACE::inet_pton(AF_INET6, "1:2:3:4:5:6:7:192.168.1.1",
                                         &addr));
  EXPECT_EQ(0, LIBC_NAMESPACE::inet_pton(AF_INET6, "1:2:3:4:5:6::192.168.1.1",
                                         &addr));

  // Whitespace
  EXPECT_EQ(0, LIBC_NAMESPACE::inet_pton(AF_INET6, " ::1", &addr));
  EXPECT_EQ(0, LIBC_NAMESPACE::inet_pton(AF_INET6, "::1 ", &addr));
  ASSERT_ERRNO_SUCCESS();
}

TEST_F(LlvmLibcInetPtonTest, InvalidAddressFamily) {
  struct in_addr addr;

  EXPECT_EQ(-1, LIBC_NAMESPACE::inet_pton(AF_INET + AF_INET6 + 1, "127.0.0.1",
                                          &addr));
  ASSERT_ERRNO_EQ(EAFNOSUPPORT);

  EXPECT_EQ(-1, LIBC_NAMESPACE::inet_pton(12345, "127.0.0.1", &addr));
  ASSERT_ERRNO_EQ(EAFNOSUPPORT);
}
