//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unittests for the network byte order helpers.
///
//===----------------------------------------------------------------------===//

#include "hdr/stdint_proxy.h"
#include "src/__support/endian_internal.h"
#include "src/__support/macros/config.h"
#include "src/__support/net/byte_order.h"
#include "test/UnitTest/Test.h"

namespace LIBC_NAMESPACE_DECL {
namespace net {

static_assert(ntoh(hton<uint8_t>(0x12)) == 0x12);
static_assert(hton(ntoh<uint8_t>(0x12)) == 0x12);
static_assert(ntoh(hton<uint16_t>(0x0123)) == 0x0123);
static_assert(hton(ntoh<uint16_t>(0x0123)) == 0x0123);
static_assert(ntoh(hton<uint32_t>(0x01234567)) == 0x01234567);
static_assert(hton(ntoh<uint32_t>(0x01234567)) == 0x01234567);
static_assert(ntoh(hton<uint64_t>(0x0123456789abcdef)) == 0x0123456789abcdef);
static_assert(hton(ntoh<uint64_t>(0x0123456789abcdef)) == 0x0123456789abcdef);

template <typename T> void check_conversion(T original, T swapped) {
#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
  EXPECT_EQ(hton(original), swapped);
  EXPECT_EQ(ntoh(swapped), original);
#endif
#if __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
  EXPECT_EQ(hton(original), original);
  EXPECT_EQ(ntoh(original), original);
#endif
  EXPECT_EQ(ntoh(hton(original)), original);
  EXPECT_EQ(hton(ntoh(original)), original);
}

TEST(LlvmLibcNetByteOrderTest, Conversion) {
  check_conversion<uint8_t>(0x12, 0x12);
  check_conversion<uint16_t>(0x2301, 0x0123);
  check_conversion<uint32_t>(0x67452301, 0x01234567);
  check_conversion<uint64_t>(0xefcdab8967452301, 0x0123456789abcdef);
}

} // namespace net
} // namespace LIBC_NAMESPACE_DECL
