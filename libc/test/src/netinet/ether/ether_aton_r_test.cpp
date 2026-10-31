//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unittests for ether_aton_r.
///
//===----------------------------------------------------------------------===//

#include "hdr/stdint_proxy.h"
#include "hdr/types/struct_ether_addr.h"
#include "src/netinet/ether/ether_aton_r.h"
#include "test/UnitTest/Test.h"

namespace {

// Packs the address into an integer to get readable test failure messages.
uint64_t to_u64(const struct ether_addr &addr) {
  uint64_t result = 0;
  for (uint8_t octet : addr.ether_addr_octet)
    result = result << 8 | octet;
  return result;
}

} // namespace

TEST(LlvmLibcEtherAtonRTest, ValidAddresses) {
  struct ether_addr addr;

  EXPECT_EQ(&addr, LIBC_NAMESPACE::ether_aton_r("00:00:00:00:00:00", &addr));
  EXPECT_EQ(uint64_t{0x000000000000}, to_u64(addr));

  EXPECT_EQ(&addr, LIBC_NAMESPACE::ether_aton_r("ff:ff:ff:ff:ff:ff", &addr));
  EXPECT_EQ(uint64_t{0xffffffffffff}, to_u64(addr));

  EXPECT_EQ(&addr, LIBC_NAMESPACE::ether_aton_r("01:23:45:67:89:ab", &addr));
  EXPECT_EQ(uint64_t{0x0123456789ab}, to_u64(addr));

  // Uppercase and mixed-case digits.
  EXPECT_EQ(&addr, LIBC_NAMESPACE::ether_aton_r("CD:EF:aB:Cd:eF:00", &addr));
  EXPECT_EQ(uint64_t{0xcdefabcdef00}, to_u64(addr));

  // Single-digit octets, as produced by ether_ntoa.
  EXPECT_EQ(&addr, LIBC_NAMESPACE::ether_aton_r("0:0:0:0:0:0", &addr));
  EXPECT_EQ(uint64_t{0x000000000000}, to_u64(addr));

  EXPECT_EQ(&addr, LIBC_NAMESPACE::ether_aton_r("1:2:3:4:5:6", &addr));
  EXPECT_EQ(uint64_t{0x010203040506}, to_u64(addr));

  EXPECT_EQ(&addr, LIBC_NAMESPACE::ether_aton_r("cd:ef:a:10:0:9f", &addr));
  EXPECT_EQ(uint64_t{0xcdef0a10009f}, to_u64(addr));
}

TEST(LlvmLibcEtherAtonRTest, InvalidAddresses) {
  struct ether_addr addr;

  const char *invalid[] = {
      "",
      ":",
      "01:23:45:67:89",
      "01:23:45:67:89:",
      "a",
      "1:2:3:4:5",
      "1:2:3:4:5:",
      "01:23:45:67:89:ab:",
      "01:23:45:67:89:ab:cd",
      "001:23:45:67:89:ab",
      "01:23:45:67:89:abc",
      "01:23:45:67::ab",
      ":01:23:45:67:89:ab",
      " 01:23:45:67:89:ab",
      "01 :23:45:67:89:ab",
      "01: 23:45:67:89:ab",
      "01-23-45-67-89-ab",
      "01.23.45.67.89.ab",
      "0123456789ab",
      "0g:23:45:67:89:ab",
      "01:23:45:67:89:ag",
      "0x1:23:45:67:89:ab",
      "0X1:23:45:67:89:ab",
      "+1:23:45:67:89:ab",
      "-1:23:45:67:89:ab",
      "01:23:45:67:89: ab",
      "01:23:45:67:89:+b",
      "01:23:45:67:89:-b",
      "01:23:45:67:89:0x",
      "01:23:45:67:89:abfoo",
      "01:23:45:67:89:ab ",
      "1:2:3:4:5:6 ",
      "1:2:3:4:5:6\tfoo",
      "1:2:3:4:5:6\n",
  };
  for (const char *str : invalid)
    EXPECT_EQ(static_cast<struct ether_addr *>(nullptr),
              LIBC_NAMESPACE::ether_aton_r(str, &addr));
}

TEST(LlvmLibcEtherAtonRTest, OutputUnchangedOnFailure) {
  struct ether_addr addr = {{0x11, 0x22, 0x33, 0x44, 0x55, 0x66}};
  EXPECT_EQ(static_cast<struct ether_addr *>(nullptr),
            LIBC_NAMESPACE::ether_aton_r("01:23:45:67:89:xy", &addr));
  EXPECT_EQ(uint64_t{0x112233445566}, to_u64(addr));
}
