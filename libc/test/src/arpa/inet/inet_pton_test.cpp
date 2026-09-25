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

TEST_F(LlvmLibcInetPtonTest, InvalidAddressFamily) {
  struct in_addr addr;

  EXPECT_EQ(-1, LIBC_NAMESPACE::inet_pton(AF_INET + AF_INET6 + 1, "127.0.0.1",
                                          &addr));
  ASSERT_ERRNO_EQ(EAFNOSUPPORT);

  EXPECT_EQ(-1, LIBC_NAMESPACE::inet_pton(12345, "127.0.0.1", &addr));
  ASSERT_ERRNO_EQ(EAFNOSUPPORT);
}
