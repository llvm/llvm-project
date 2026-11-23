//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unittests for netinet/ip.
///
//===----------------------------------------------------------------------===//

#include "hdr/types/struct_iphdr.h"
#include "src/__support/net/byte_order.h"
#include "src/string/memcmp.h"
#include "test/UnitTest/Test.h"

TEST(LlvmLibcNetinetIpTest, StructIphdrLayout) {
  ASSERT_EQ(sizeof(struct iphdr), static_cast<size_t>(20));

  struct iphdr header;
  header.version = 4;
  header.ihl = 5;
  header.tos = 0x10;
  header.tot_len = LIBC_NAMESPACE::net::hton<uint16_t>(0x0028);
  header.id = LIBC_NAMESPACE::net::hton<uint16_t>(0x1234);
  header.frag_off = LIBC_NAMESPACE::net::hton<uint16_t>(0x4000);
  header.ttl = 64;
  header.protocol = 6;
  header.check = LIBC_NAMESPACE::net::hton<uint16_t>(0xabcd);
  header.saddr = LIBC_NAMESPACE::net::hton<uint32_t>(0x7f000001);
  header.daddr = LIBC_NAMESPACE::net::hton<uint32_t>(0x08080808);

  EXPECT_EQ(LIBC_NAMESPACE::memcmp(&header,
                                   "\x45\x10\x00\x28"
                                   "\x12\x34\x40\x00"
                                   "\x40\x06\xab\xcd"
                                   "\x7f\x00\x00\x01"
                                   "\x08\x08\x08\x08",
                                   20),
            0);
}
