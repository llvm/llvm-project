//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unittests for ether_aton.
///
//===----------------------------------------------------------------------===//

#include "hdr/stdint_proxy.h"
#include "hdr/types/struct_ether_addr.h"
#include "src/netinet/ether/ether_aton.h"
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

TEST(LlvmLibcEtherAtonTest, BasicConversion) {
  struct ether_addr *addr = LIBC_NAMESPACE::ether_aton("01:23:45:67:89:ab");
  ASSERT_NE(static_cast<struct ether_addr *>(nullptr), addr);
  EXPECT_EQ(uint64_t{0x0123456789ab}, to_u64(*addr));

  addr = LIBC_NAMESPACE::ether_aton("cd:ef:a:10:0:9F");
  ASSERT_NE(static_cast<struct ether_addr *>(nullptr), addr);
  EXPECT_EQ(uint64_t{0xcdef0a10009f}, to_u64(*addr));

  EXPECT_EQ(static_cast<struct ether_addr *>(nullptr),
            LIBC_NAMESPACE::ether_aton("01:23:45:67:89"));
  EXPECT_EQ(static_cast<struct ether_addr *>(nullptr),
            LIBC_NAMESPACE::ether_aton("01:23:45:67:89:abc"));
}

TEST(LlvmLibcEtherAtonTest, BufferReuseAndOverwrite) {
  struct ether_addr *ptr1 = LIBC_NAMESPACE::ether_aton("11:22:33:44:55:66");
  ASSERT_NE(static_cast<struct ether_addr *>(nullptr), ptr1);
  EXPECT_EQ(uint64_t{0x112233445566}, to_u64(*ptr1));

  struct ether_addr *ptr2 = LIBC_NAMESPACE::ether_aton("1:2:3:4:5:6");
  EXPECT_EQ(ptr1, ptr2);
  EXPECT_EQ(uint64_t{0x010203040506}, to_u64(*ptr1));

  // A failed conversion leaves the buffer untouched.
  EXPECT_EQ(static_cast<struct ether_addr *>(nullptr),
            LIBC_NAMESPACE::ether_aton("a:b:c:d:e:g"));
  EXPECT_EQ(uint64_t{0x010203040506}, to_u64(*ptr1));
}
