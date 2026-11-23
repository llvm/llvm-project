//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unittests for the network hex parsing helpers.
///
//===----------------------------------------------------------------------===//

#include "hdr/stdint_proxy.h"
#include "src/__support/CPP/optional.h"
#include "src/__support/CPP/string_view.h"
#include "src/__support/macros/config.h"
#include "src/__support/net/hex.h"
#include "test/UnitTest/Test.h"

namespace LIBC_NAMESPACE_DECL {
namespace net {

TEST(LlvmLibcNetHexTest, HexDigitValue) {
  EXPECT_EQ(hex_digit_value('0'), 0);
  EXPECT_EQ(hex_digit_value('9'), 9);
  EXPECT_EQ(hex_digit_value('a'), 10);
  EXPECT_EQ(hex_digit_value('F'), 15);
  EXPECT_EQ(hex_digit_value('g'), -1);
  EXPECT_EQ(hex_digit_value('G'), -1);
  EXPECT_EQ(hex_digit_value(':'), -1);
  EXPECT_EQ(hex_digit_value('\0'), -1);
}

TEST(LlvmLibcNetHexTest, ConsumeHex) {
  cpp::string_view src = "aB:cd";
  cpp::optional<uint8_t> val = consume_hex<uint8_t>(src);
  ASSERT_TRUE(val.has_value());
  EXPECT_EQ(*val, uint8_t{0xab});
  EXPECT_EQ(src, cpp::string_view(":cd"));

  src = "f";
  val = consume_hex<uint8_t>(src);
  ASSERT_TRUE(val.has_value());
  EXPECT_EQ(*val, uint8_t{0xf});
  EXPECT_TRUE(src.empty());

  src = "12345";
  cpp::optional<uint16_t> val16 = consume_hex<uint16_t>(src);
  ASSERT_TRUE(val16.has_value());
  EXPECT_EQ(*val16, uint16_t{0x1234});
  EXPECT_EQ(src, cpp::string_view("5"));

  src = "0ffff";
  val16 = consume_hex<uint16_t>(src);
  ASSERT_TRUE(val16.has_value());
  EXPECT_EQ(*val16, uint16_t{0x0fff});
  EXPECT_EQ(src, cpp::string_view("f"));
}

TEST(LlvmLibcNetHexTest, ConsumeHexInvalid) {
  const cpp::string_view INVALID[] = {"", ":1", "g", " 1", "+1", "-1"};
  for (cpp::string_view str : INVALID) {
    cpp::string_view src = str;
    EXPECT_FALSE(consume_hex<uint16_t>(src).has_value());
    EXPECT_EQ(src, str);
  }

  // A base prefix is not recognized: only the leading zero is consumed.
  cpp::string_view src = "0x12";
  cpp::optional<uint16_t> val = consume_hex<uint16_t>(src);
  ASSERT_TRUE(val.has_value());
  EXPECT_EQ(*val, uint16_t{0});
  EXPECT_EQ(src, cpp::string_view("x12"));
}

} // namespace net
} // namespace LIBC_NAMESPACE_DECL
