//===-- Unittests for IntegerToString -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/CPP/string_view.h"
#include "src/__support/UInt.h"
#include "src/__support/UInt128.h"
#include "src/__support/integer_to_string.h"

#include "test/UnitTest/Test.h"

#include "limits.h"

using __llvm_libc::IntegerToString;
using __llvm_libc::cpp::string_view;

TEST(LlvmLibcIntegerToStringTest, UINT8) {
  char buf[IntegerToString::dec_bufsize<uint8_t>()];
  EXPECT_EQ(*IntegerToString::dec(uint8_t(0), buf), string_view("0"));
  EXPECT_EQ(*IntegerToString::dec(uint8_t(1), buf), string_view("1"));
  EXPECT_EQ(*IntegerToString::dec(uint8_t(12), buf), string_view("12"));
  EXPECT_EQ(*IntegerToString::dec(uint8_t(123), buf), string_view("123"));
  EXPECT_EQ(*IntegerToString::dec(uint8_t(UINT8_MAX), buf), string_view("255"));
  EXPECT_EQ(*IntegerToString::dec(uint8_t(-1), buf), string_view("255"));
}

TEST(LlvmLibcIntegerToStringTest, INT8) {
  char buf[IntegerToString::dec_bufsize<int8_t>()];
  EXPECT_EQ(*IntegerToString::dec(int8_t(0), buf), string_view("0"));
  EXPECT_EQ(*IntegerToString::dec(int8_t(1), buf), string_view("1"));
  EXPECT_EQ(*IntegerToString::dec(int8_t(12), buf), string_view("12"));
  EXPECT_EQ(*IntegerToString::dec(int8_t(123), buf), string_view("123"));
  EXPECT_EQ(*IntegerToString::dec(int8_t(-12), buf), string_view("-12"));
  EXPECT_EQ(*IntegerToString::dec(int8_t(-123), buf), string_view("-123"));
  EXPECT_EQ(*IntegerToString::dec(int8_t(INT8_MAX), buf), string_view("127"));
  EXPECT_EQ(*IntegerToString::dec(int8_t(INT8_MIN), buf), string_view("-128"));
}

TEST(LlvmLibcIntegerToStringTest, UINT16) {
  char buf[IntegerToString::dec_bufsize<uint16_t>()];
  EXPECT_EQ(*IntegerToString::dec(uint16_t(0), buf), string_view("0"));
  EXPECT_EQ(*IntegerToString::dec(uint16_t(1), buf), string_view("1"));
  EXPECT_EQ(*IntegerToString::dec(uint16_t(12), buf), string_view("12"));
  EXPECT_EQ(*IntegerToString::dec(uint16_t(123), buf), string_view("123"));
  EXPECT_EQ(*IntegerToString::dec(uint16_t(1234), buf), string_view("1234"));
  EXPECT_EQ(*IntegerToString::dec(uint16_t(12345), buf), string_view("12345"));
  EXPECT_EQ(*IntegerToString::dec(uint16_t(UINT16_MAX), buf),
            string_view("65535"));
  EXPECT_EQ(*IntegerToString::dec(uint16_t(-1), buf), string_view("65535"));
}

TEST(LlvmLibcIntegerToStringTest, INT16) {
  char buf[IntegerToString::dec_bufsize<int16_t>()];
  EXPECT_EQ(*IntegerToString::dec(int16_t(0), buf), string_view("0"));
  EXPECT_EQ(*IntegerToString::dec(int16_t(1), buf), string_view("1"));
  EXPECT_EQ(*IntegerToString::dec(int16_t(12), buf), string_view("12"));
  EXPECT_EQ(*IntegerToString::dec(int16_t(123), buf), string_view("123"));
  EXPECT_EQ(*IntegerToString::dec(int16_t(1234), buf), string_view("1234"));
  EXPECT_EQ(*IntegerToString::dec(int16_t(12345), buf), string_view("12345"));
  EXPECT_EQ(*IntegerToString::dec(int16_t(-1), buf), string_view("-1"));
  EXPECT_EQ(*IntegerToString::dec(int16_t(-12), buf), string_view("-12"));
  EXPECT_EQ(*IntegerToString::dec(int16_t(-123), buf), string_view("-123"));
  EXPECT_EQ(*IntegerToString::dec(int16_t(-1234), buf), string_view("-1234"));
  EXPECT_EQ(*IntegerToString::dec(int16_t(-12345), buf), string_view("-12345"));
  EXPECT_EQ(*IntegerToString::dec(int16_t(INT16_MAX), buf),
            string_view("32767"));
  EXPECT_EQ(*IntegerToString::dec(int16_t(INT16_MIN), buf),
            string_view("-32768"));
}

TEST(LlvmLibcIntegerToStringTest, UINT32) {
  char buf[IntegerToString::dec_bufsize<uint32_t>()];
  EXPECT_EQ(*IntegerToString::dec(uint32_t(0), buf), string_view("0"));
  EXPECT_EQ(*IntegerToString::dec(uint32_t(1), buf), string_view("1"));
  EXPECT_EQ(*IntegerToString::dec(uint32_t(12), buf), string_view("12"));
  EXPECT_EQ(*IntegerToString::dec(uint32_t(123), buf), string_view("123"));
  EXPECT_EQ(*IntegerToString::dec(uint32_t(1234), buf), string_view("1234"));
  EXPECT_EQ(*IntegerToString::dec(uint32_t(12345), buf), string_view("12345"));
  EXPECT_EQ(*IntegerToString::dec(uint32_t(123456), buf), string_view("123456"));
  EXPECT_EQ(*IntegerToString::dec(uint32_t(1234567), buf),
            string_view("1234567"));
  EXPECT_EQ(*IntegerToString::dec(uint32_t(12345678), buf),
            string_view("12345678"));
  EXPECT_EQ(*IntegerToString::dec(uint32_t(123456789), buf),
            string_view("123456789"));
  EXPECT_EQ(*IntegerToString::dec(uint32_t(1234567890), buf),
            string_view("1234567890"));
  EXPECT_EQ(*IntegerToString::dec(uint32_t(UINT32_MAX), buf),
            string_view("4294967295"));
  EXPECT_EQ(*IntegerToString::dec(uint32_t(-1), buf), string_view("4294967295"));
}

TEST(LlvmLibcIntegerToStringTest, INT32) {
  char buf[IntegerToString::dec_bufsize<int32_t>()];
  EXPECT_EQ(*IntegerToString::dec(int32_t(0), buf), string_view("0"));
  EXPECT_EQ(*IntegerToString::dec(int32_t(1), buf), string_view("1"));
  EXPECT_EQ(*IntegerToString::dec(int32_t(12), buf), string_view("12"));
  EXPECT_EQ(*IntegerToString::dec(int32_t(123), buf), string_view("123"));
  EXPECT_EQ(*IntegerToString::dec(int32_t(1234), buf), string_view("1234"));
  EXPECT_EQ(*IntegerToString::dec(int32_t(12345), buf), string_view("12345"));
  EXPECT_EQ(*IntegerToString::dec(int32_t(123456), buf), string_view("123456"));
  EXPECT_EQ(*IntegerToString::dec(int32_t(1234567), buf),
            string_view("1234567"));
  EXPECT_EQ(*IntegerToString::dec(int32_t(12345678), buf),
            string_view("12345678"));
  EXPECT_EQ(*IntegerToString::dec(int32_t(123456789), buf),
            string_view("123456789"));
  EXPECT_EQ(*IntegerToString::dec(int32_t(1234567890), buf),
            string_view("1234567890"));
  EXPECT_EQ(*IntegerToString::dec(int32_t(-1), buf), string_view("-1"));
  EXPECT_EQ(*IntegerToString::dec(int32_t(-12), buf), string_view("-12"));
  EXPECT_EQ(*IntegerToString::dec(int32_t(-123), buf), string_view("-123"));
  EXPECT_EQ(*IntegerToString::dec(int32_t(-1234), buf), string_view("-1234"));
  EXPECT_EQ(*IntegerToString::dec(int32_t(-12345), buf), string_view("-12345"));
  EXPECT_EQ(*IntegerToString::dec(int32_t(-123456), buf),
            string_view("-123456"));
  EXPECT_EQ(*IntegerToString::dec(int32_t(-1234567), buf),
            string_view("-1234567"));
  EXPECT_EQ(*IntegerToString::dec(int32_t(-12345678), buf),
            string_view("-12345678"));
  EXPECT_EQ(*IntegerToString::dec(int32_t(-123456789), buf),
            string_view("-123456789"));
  EXPECT_EQ(*IntegerToString::dec(int32_t(-1234567890), buf),
            string_view("-1234567890"));
  EXPECT_EQ(*IntegerToString::dec(int32_t(INT32_MAX), buf),
            string_view("2147483647"));
  EXPECT_EQ(*IntegerToString::dec(int32_t(INT32_MIN), buf),
            string_view("-2147483648"));
}

TEST(LlvmLibcIntegerToStringTest, UINT64) {
  char buf[IntegerToString::dec_bufsize<uint64_t>()];
  EXPECT_EQ(*IntegerToString::dec(uint64_t(0), buf), string_view("0"));
  EXPECT_EQ(*IntegerToString::dec(uint64_t(1), buf), string_view("1"));
  EXPECT_EQ(*IntegerToString::dec(uint64_t(12), buf), string_view("12"));
  EXPECT_EQ(*IntegerToString::dec(uint64_t(123), buf), string_view("123"));
  EXPECT_EQ(*IntegerToString::dec(uint64_t(1234), buf), string_view("1234"));
  EXPECT_EQ(*IntegerToString::dec(uint64_t(12345), buf), string_view("12345"));
  EXPECT_EQ(*IntegerToString::dec(uint64_t(123456), buf), string_view("123456"));
  EXPECT_EQ(*IntegerToString::dec(uint64_t(1234567), buf),
            string_view("1234567"));
  EXPECT_EQ(*IntegerToString::dec(uint64_t(12345678), buf),
            string_view("12345678"));
  EXPECT_EQ(*IntegerToString::dec(uint64_t(123456789), buf),
            string_view("123456789"));
  EXPECT_EQ(*IntegerToString::dec(uint64_t(1234567890), buf),
            string_view("1234567890"));
  EXPECT_EQ(*IntegerToString::dec(uint64_t(1234567890123456789), buf),
            string_view("1234567890123456789"));
  EXPECT_EQ(*IntegerToString::dec(uint64_t(UINT64_MAX), buf),
            string_view("18446744073709551615"));
  EXPECT_EQ(*IntegerToString::dec(uint64_t(-1), buf),
            string_view("18446744073709551615"));
}

TEST(LlvmLibcIntegerToStringTest, INT64) {
  char buf[IntegerToString::dec_bufsize<int64_t>()];
  EXPECT_EQ(*IntegerToString::dec(int64_t(0), buf), string_view("0"));
  EXPECT_EQ(*IntegerToString::dec(int64_t(1), buf), string_view("1"));
  EXPECT_EQ(*IntegerToString::dec(int64_t(12), buf), string_view("12"));
  EXPECT_EQ(*IntegerToString::dec(int64_t(123), buf), string_view("123"));
  EXPECT_EQ(*IntegerToString::dec(int64_t(1234), buf), string_view("1234"));
  EXPECT_EQ(*IntegerToString::dec(int64_t(12345), buf), string_view("12345"));
  EXPECT_EQ(*IntegerToString::dec(int64_t(123456), buf), string_view("123456"));
  EXPECT_EQ(*IntegerToString::dec(int64_t(1234567), buf),
            string_view("1234567"));
  EXPECT_EQ(*IntegerToString::dec(int64_t(12345678), buf),
            string_view("12345678"));
  EXPECT_EQ(*IntegerToString::dec(int64_t(123456789), buf),
            string_view("123456789"));
  EXPECT_EQ(*IntegerToString::dec(int64_t(1234567890), buf),
            string_view("1234567890"));
  EXPECT_EQ(*IntegerToString::dec(int64_t(1234567890123456789), buf),
            string_view("1234567890123456789"));
  EXPECT_EQ(*IntegerToString::dec(int64_t(-1), buf), string_view("-1"));
  EXPECT_EQ(*IntegerToString::dec(int64_t(-12), buf), string_view("-12"));
  EXPECT_EQ(*IntegerToString::dec(int64_t(-123), buf), string_view("-123"));
  EXPECT_EQ(*IntegerToString::dec(int64_t(-1234), buf), string_view("-1234"));
  EXPECT_EQ(*IntegerToString::dec(int64_t(-12345), buf), string_view("-12345"));
  EXPECT_EQ(*IntegerToString::dec(int64_t(-123456), buf),
            string_view("-123456"));
  EXPECT_EQ(*IntegerToString::dec(int64_t(-1234567), buf),
            string_view("-1234567"));
  EXPECT_EQ(*IntegerToString::dec(int64_t(-12345678), buf),
            string_view("-12345678"));
  EXPECT_EQ(*IntegerToString::dec(int64_t(-123456789), buf),
            string_view("-123456789"));
  EXPECT_EQ(*IntegerToString::dec(int64_t(-1234567890), buf),
            string_view("-1234567890"));
  EXPECT_EQ(*IntegerToString::dec(int64_t(-1234567890123456789), buf),
            string_view("-1234567890123456789"));
  EXPECT_EQ(*IntegerToString::dec(int64_t(INT64_MAX), buf),
            string_view("9223372036854775807"));
  EXPECT_EQ(*IntegerToString::dec(int64_t(INT64_MIN), buf),
            string_view("-9223372036854775808"));
}

TEST(LlvmLibcIntegerToStringTest, UINT64_Base_8) {
  char buf[IntegerToString::oct_bufsize<uint64_t>()];
  EXPECT_EQ((*IntegerToString::oct(uint64_t(0), buf)), string_view("0"));
  EXPECT_EQ((*IntegerToString::oct(uint64_t(012345), buf)),
            string_view("12345"));
  EXPECT_EQ((*IntegerToString::oct(uint64_t(0123456701234567012345), buf)),
            string_view("123456701234567012345"));
  EXPECT_EQ((*IntegerToString::oct(uint64_t(01777777777777777777777), buf)),
            string_view("1777777777777777777777"));
}

TEST(LlvmLibcIntegerToStringTest, UINT64_Base_16) {
  char buf[IntegerToString::hex_bufsize<uint64_t>()];
  EXPECT_EQ(*IntegerToString::hex(uint64_t(0), buf), string_view("0"));
  EXPECT_EQ(*IntegerToString::hex(uint64_t(0x12345), buf), string_view("12345"));
  EXPECT_EQ((*IntegerToString::hex(uint64_t(0x123456789abcdef), buf)),
            string_view("123456789abcdef"));
  EXPECT_EQ(*IntegerToString::hex(uint64_t(0x123456789abcdef), buf, false),
            string_view("123456789ABCDEF"));
  EXPECT_EQ(*IntegerToString::hex(uint64_t(0xffffffffffffffff), buf),
            string_view("ffffffffffffffff"));
}

TEST(LlvmLibcIntegerToStringTest, UINT64_Base_2) {
  char buf[IntegerToString::bin_bufsize<uint64_t>()];
  EXPECT_EQ(*IntegerToString::bin(uint64_t(0), buf), string_view("0"));
  EXPECT_EQ(*IntegerToString::bin(uint64_t(0xf0c), buf),
            string_view("111100001100"));
  EXPECT_EQ(*IntegerToString::bin(uint64_t(0x123abc), buf),
            string_view("100100011101010111100"));
  EXPECT_EQ(
      *IntegerToString::bin(uint64_t(0xffffffffffffffff), buf),
      string_view(
          "1111111111111111111111111111111111111111111111111111111111111111"));
}

TEST(LlvmLibcIntegerToStringTest, UINT64_Base_36) {
  char buf[IntegerToString::bufsize<36, uint64_t>()];
  EXPECT_EQ(*IntegerToString::convert<36>(uint64_t(0), buf), string_view("0"));
  EXPECT_EQ(*IntegerToString::convert<36>(uint64_t(12345), buf),
            string_view("9ix"));
  EXPECT_EQ(*IntegerToString::convert<36>(uint64_t(1047601316295595), buf),
            string_view("abcdefghij"));
  EXPECT_EQ(*IntegerToString::convert<36>(uint64_t(2092218013456445), buf),
            string_view("klmnopqrst"));
  EXPECT_EQ(*IntegerToString::convert<36>(uint64_t(1867590395), buf, false),
            string_view("UVWXYZ"));
  EXPECT_EQ(*IntegerToString::convert<36>(uint64_t(0xffffffffffffffff), buf),
            string_view("3w5e11264sgsf"));
}

TEST(LlvmLibcIntegerToStringTest, UINT128_Base_16) {
  char buf[IntegerToString::hex_bufsize<UInt128>()];
  EXPECT_EQ(*IntegerToString::hex(static_cast<UInt128>(0), buf),
            string_view("00000000000000000000000000000000"));
  EXPECT_EQ(*IntegerToString::hex(static_cast<UInt128>(0x12345), buf),
            string_view("00000000000000000000000000012345"));
  EXPECT_EQ((*IntegerToString::hex(static_cast<UInt128>(0x1234) << 112, buf)),
            string_view("12340000000000000000000000000000"));
  EXPECT_EQ((*IntegerToString::hex(static_cast<UInt128>(0x1234) << 48, buf)),
            string_view("00000000000000001234000000000000"));
  EXPECT_EQ((*IntegerToString::hex(static_cast<UInt128>(0x1234) << 52, buf)),
            string_view("00000000000000012340000000000000"));
}

TEST(LlvmLibcIntegerToStringTest, UINT256_Base_16) {
  using UInt256 = __llvm_libc::cpp::UInt<256>;
  char buf[IntegerToString::hex_bufsize<UInt256>()];
  EXPECT_EQ(
      *IntegerToString::hex(static_cast<UInt256>(0), buf),
      string_view(
          "0000000000000000000000000000000000000000000000000000000000000000"));
  EXPECT_EQ(
      *IntegerToString::hex(static_cast<UInt256>(0x12345), buf),
      string_view(
          "0000000000000000000000000000000000000000000000000000000000012345"));
  EXPECT_EQ(
      (*IntegerToString::hex(static_cast<UInt256>(0x1234) << 112, buf)),
      string_view(
          "0000000000000000000000000000000012340000000000000000000000000000"));
  EXPECT_EQ(
      (*IntegerToString::hex(static_cast<UInt256>(0x1234) << 116, buf)),
      string_view(
          "0000000000000000000000000000000123400000000000000000000000000000"));
  EXPECT_EQ(
      (*IntegerToString::hex(static_cast<UInt256>(0x1234) << 240, buf)),
      string_view(
          "1234000000000000000000000000000000000000000000000000000000000000"));
}
