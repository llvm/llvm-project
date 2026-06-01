//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unittests for netinet/udp.
///
//===----------------------------------------------------------------------===//

#include "hdr/types/struct_udphdr.h"
#include "src/arpa/inet/htons.h"
#include "src/string/memcmp.h"
#include "test/UnitTest/Test.h"

TEST(LlvmLibcNetinetUdpTest, StructUdphdrLayout) {
  EXPECT_EQ(sizeof(struct udphdr), size_t(8));

  struct udphdr header;

  // 1. Set fields using BSD-style names and read via Linux-style names
  header.uh_sport = LIBC_NAMESPACE::htons(0x1234);
  header.uh_dport = LIBC_NAMESPACE::htons(0x5678);
  header.uh_ulen = LIBC_NAMESPACE::htons(0x0010);
  header.uh_sum = LIBC_NAMESPACE::htons(0xABCD);

  EXPECT_EQ(
      LIBC_NAMESPACE::memcmp(&header, "\x12\x34\x56\x78\x00\x10\xAB\xCD", 8),
      0);

  // 2. Set fields using Linux-style names and read via BSD-style names
  header.source = LIBC_NAMESPACE::htons(0x4321);
  header.dest = LIBC_NAMESPACE::htons(0x8765);
  header.len = LIBC_NAMESPACE::htons(0x2000);
  header.check = LIBC_NAMESPACE::htons(0xDCBA);

  EXPECT_EQ(
      LIBC_NAMESPACE::memcmp(&header, "\x43\x21\x87\x65\x20\x00\xDC\xBA", 8),
      0);
}
