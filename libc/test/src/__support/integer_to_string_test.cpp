//===-- Unittests for integer_to_string -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/CPP/StringView.h"
#include "src/__support/integer_to_string.h"

#include "utils/UnitTest/Test.h"

#include "limits.h"

using __llvm_libc::integer_to_string;
using __llvm_libc::IntegerToString;
using __llvm_libc::cpp::StringView;

TEST(LlvmLibcIntegerToStringTest, UINT8) {
  EXPECT_EQ(integer_to_string(uint8_t(0)).str(), (StringView("0")));
  EXPECT_EQ(integer_to_string(uint8_t(1)).str(), (StringView("1")));
  EXPECT_EQ(integer_to_string(uint8_t(12)).str(), (StringView("12")));
  EXPECT_EQ(integer_to_string(uint8_t(123)).str(), (StringView("123")));
  EXPECT_EQ(integer_to_string(uint8_t(UINT8_MAX)).str(), (StringView("255")));
  EXPECT_EQ(integer_to_string(uint8_t(-1)).str(), (StringView("255")));
}

TEST(LlvmLibcIntegerToStringTest, INT8) {
  EXPECT_EQ(integer_to_string(int8_t(0)).str(), (StringView("0")));
  EXPECT_EQ(integer_to_string(int8_t(1)).str(), (StringView("1")));
  EXPECT_EQ(integer_to_string(int8_t(12)).str(), (StringView("12")));
  EXPECT_EQ(integer_to_string(int8_t(123)).str(), (StringView("123")));
  EXPECT_EQ(integer_to_string(int8_t(-12)).str(), (StringView("-12")));
  EXPECT_EQ(integer_to_string(int8_t(-123)).str(), (StringView("-123")));
  EXPECT_EQ(integer_to_string(int8_t(INT8_MAX)).str(), (StringView("127")));
  EXPECT_EQ(integer_to_string(int8_t(INT8_MIN)).str(), (StringView("-128")));
}

TEST(LlvmLibcIntegerToStringTest, UINT16) {
  EXPECT_EQ(integer_to_string(uint16_t(0)).str(), (StringView("0")));
  EXPECT_EQ(integer_to_string(uint16_t(1)).str(), (StringView("1")));
  EXPECT_EQ(integer_to_string(uint16_t(12)).str(), (StringView("12")));
  EXPECT_EQ(integer_to_string(uint16_t(123)).str(), (StringView("123")));
  EXPECT_EQ(integer_to_string(uint16_t(1234)).str(), (StringView("1234")));
  EXPECT_EQ(integer_to_string(uint16_t(12345)).str(), (StringView("12345")));
  EXPECT_EQ(integer_to_string(uint16_t(UINT16_MAX)).str(),
            (StringView("65535")));
  EXPECT_EQ(integer_to_string(uint16_t(-1)).str(), (StringView("65535")));
}

TEST(LlvmLibcIntegerToStringTest, INT16) {
  EXPECT_EQ(integer_to_string(int16_t(0)).str(), (StringView("0")));
  EXPECT_EQ(integer_to_string(int16_t(1)).str(), (StringView("1")));
  EXPECT_EQ(integer_to_string(int16_t(12)).str(), (StringView("12")));
  EXPECT_EQ(integer_to_string(int16_t(123)).str(), (StringView("123")));
  EXPECT_EQ(integer_to_string(int16_t(1234)).str(), (StringView("1234")));
  EXPECT_EQ(integer_to_string(int16_t(12345)).str(), (StringView("12345")));
  EXPECT_EQ(integer_to_string(int16_t(-1)).str(), (StringView("-1")));
  EXPECT_EQ(integer_to_string(int16_t(-12)).str(), (StringView("-12")));
  EXPECT_EQ(integer_to_string(int16_t(-123)).str(), (StringView("-123")));
  EXPECT_EQ(integer_to_string(int16_t(-1234)).str(), (StringView("-1234")));
  EXPECT_EQ(integer_to_string(int16_t(-12345)).str(), (StringView("-12345")));
  EXPECT_EQ(integer_to_string(int16_t(INT16_MAX)).str(), (StringView("32767")));
  EXPECT_EQ(integer_to_string(int16_t(INT16_MIN)).str(),
            (StringView("-32768")));
}

TEST(LlvmLibcIntegerToStringTest, UINT32) {
  EXPECT_EQ(integer_to_string(uint32_t(0)).str(), (StringView("0")));
  EXPECT_EQ(integer_to_string(uint32_t(1)).str(), (StringView("1")));
  EXPECT_EQ(integer_to_string(uint32_t(12)).str(), (StringView("12")));
  EXPECT_EQ(integer_to_string(uint32_t(123)).str(), (StringView("123")));
  EXPECT_EQ(integer_to_string(uint32_t(1234)).str(), (StringView("1234")));
  EXPECT_EQ(integer_to_string(uint32_t(12345)).str(), (StringView("12345")));
  EXPECT_EQ(integer_to_string(uint32_t(123456)).str(), (StringView("123456")));
  EXPECT_EQ(integer_to_string(uint32_t(1234567)).str(),
            (StringView("1234567")));
  EXPECT_EQ(integer_to_string(uint32_t(12345678)).str(),
            (StringView("12345678")));
  EXPECT_EQ(integer_to_string(uint32_t(123456789)).str(),
            (StringView("123456789")));
  EXPECT_EQ(integer_to_string(uint32_t(1234567890)).str(),
            (StringView("1234567890")));
  EXPECT_EQ(integer_to_string(uint32_t(UINT32_MAX)).str(),
            (StringView("4294967295")));
  EXPECT_EQ(integer_to_string(uint32_t(-1)).str(), (StringView("4294967295")));
}

TEST(LlvmLibcIntegerToStringTest, INT32) {
  EXPECT_EQ(integer_to_string(int32_t(0)).str(), (StringView("0")));
  EXPECT_EQ(integer_to_string(int32_t(1)).str(), (StringView("1")));
  EXPECT_EQ(integer_to_string(int32_t(12)).str(), (StringView("12")));
  EXPECT_EQ(integer_to_string(int32_t(123)).str(), (StringView("123")));
  EXPECT_EQ(integer_to_string(int32_t(1234)).str(), (StringView("1234")));
  EXPECT_EQ(integer_to_string(int32_t(12345)).str(), (StringView("12345")));
  EXPECT_EQ(integer_to_string(int32_t(123456)).str(), (StringView("123456")));
  EXPECT_EQ(integer_to_string(int32_t(1234567)).str(), (StringView("1234567")));
  EXPECT_EQ(integer_to_string(int32_t(12345678)).str(),
            (StringView("12345678")));
  EXPECT_EQ(integer_to_string(int32_t(123456789)).str(),
            (StringView("123456789")));
  EXPECT_EQ(integer_to_string(int32_t(1234567890)).str(),
            (StringView("1234567890")));
  EXPECT_EQ(integer_to_string(int32_t(-1)).str(), (StringView("-1")));
  EXPECT_EQ(integer_to_string(int32_t(-12)).str(), (StringView("-12")));
  EXPECT_EQ(integer_to_string(int32_t(-123)).str(), (StringView("-123")));
  EXPECT_EQ(integer_to_string(int32_t(-1234)).str(), (StringView("-1234")));
  EXPECT_EQ(integer_to_string(int32_t(-12345)).str(), (StringView("-12345")));
  EXPECT_EQ(integer_to_string(int32_t(-123456)).str(), (StringView("-123456")));
  EXPECT_EQ(integer_to_string(int32_t(-1234567)).str(),
            (StringView("-1234567")));
  EXPECT_EQ(integer_to_string(int32_t(-12345678)).str(),
            (StringView("-12345678")));
  EXPECT_EQ(integer_to_string(int32_t(-123456789)).str(),
            (StringView("-123456789")));
  EXPECT_EQ(integer_to_string(int32_t(-1234567890)).str(),
            (StringView("-1234567890")));
  EXPECT_EQ(integer_to_string(int32_t(INT32_MAX)).str(),
            (StringView("2147483647")));
  EXPECT_EQ(integer_to_string(int32_t(INT32_MIN)).str(),
            (StringView("-2147483648")));
}

TEST(LlvmLibcIntegerToStringTest, UINT64) {
  EXPECT_EQ(integer_to_string(uint64_t(0)).str(), (StringView("0")));
  EXPECT_EQ(integer_to_string(uint64_t(1)).str(), (StringView("1")));
  EXPECT_EQ(integer_to_string(uint64_t(12)).str(), (StringView("12")));
  EXPECT_EQ(integer_to_string(uint64_t(123)).str(), (StringView("123")));
  EXPECT_EQ(integer_to_string(uint64_t(1234)).str(), (StringView("1234")));
  EXPECT_EQ(integer_to_string(uint64_t(12345)).str(), (StringView("12345")));
  EXPECT_EQ(integer_to_string(uint64_t(123456)).str(), (StringView("123456")));
  EXPECT_EQ(integer_to_string(uint64_t(1234567)).str(),
            (StringView("1234567")));
  EXPECT_EQ(integer_to_string(uint64_t(12345678)).str(),
            (StringView("12345678")));
  EXPECT_EQ(integer_to_string(uint64_t(123456789)).str(),
            (StringView("123456789")));
  EXPECT_EQ(integer_to_string(uint64_t(1234567890)).str(),
            (StringView("1234567890")));
  EXPECT_EQ(integer_to_string(uint64_t(1234567890123456789)).str(),
            (StringView("1234567890123456789")));
  EXPECT_EQ(integer_to_string(uint64_t(UINT64_MAX)).str(),
            (StringView("18446744073709551615")));
  EXPECT_EQ(integer_to_string(uint64_t(-1)).str(),
            (StringView("18446744073709551615")));
}

TEST(LlvmLibcIntegerToStringTest, INT64) {
  EXPECT_EQ(integer_to_string(int64_t(0)).str(), (StringView("0")));
  EXPECT_EQ(integer_to_string(int64_t(1)).str(), (StringView("1")));
  EXPECT_EQ(integer_to_string(int64_t(12)).str(), (StringView("12")));
  EXPECT_EQ(integer_to_string(int64_t(123)).str(), (StringView("123")));
  EXPECT_EQ(integer_to_string(int64_t(1234)).str(), (StringView("1234")));
  EXPECT_EQ(integer_to_string(int64_t(12345)).str(), (StringView("12345")));
  EXPECT_EQ(integer_to_string(int64_t(123456)).str(), (StringView("123456")));
  EXPECT_EQ(integer_to_string(int64_t(1234567)).str(), (StringView("1234567")));
  EXPECT_EQ(integer_to_string(int64_t(12345678)).str(),
            (StringView("12345678")));
  EXPECT_EQ(integer_to_string(int64_t(123456789)).str(),
            (StringView("123456789")));
  EXPECT_EQ(integer_to_string(int64_t(1234567890)).str(),
            (StringView("1234567890")));
  EXPECT_EQ(integer_to_string(int64_t(1234567890123456789)).str(),
            (StringView("1234567890123456789")));
  EXPECT_EQ(integer_to_string(int64_t(-1)).str(), (StringView("-1")));
  EXPECT_EQ(integer_to_string(int64_t(-12)).str(), (StringView("-12")));
  EXPECT_EQ(integer_to_string(int64_t(-123)).str(), (StringView("-123")));
  EXPECT_EQ(integer_to_string(int64_t(-1234)).str(), (StringView("-1234")));
  EXPECT_EQ(integer_to_string(int64_t(-12345)).str(), (StringView("-12345")));
  EXPECT_EQ(integer_to_string(int64_t(-123456)).str(), (StringView("-123456")));
  EXPECT_EQ(integer_to_string(int64_t(-1234567)).str(),
            (StringView("-1234567")));
  EXPECT_EQ(integer_to_string(int64_t(-12345678)).str(),
            (StringView("-12345678")));
  EXPECT_EQ(integer_to_string(int64_t(-123456789)).str(),
            (StringView("-123456789")));
  EXPECT_EQ(integer_to_string(int64_t(-1234567890)).str(),
            (StringView("-1234567890")));
  EXPECT_EQ(integer_to_string(int64_t(-1234567890123456789)).str(),
            (StringView("-1234567890123456789")));
  EXPECT_EQ(integer_to_string(int64_t(INT64_MAX)).str(),
            (StringView("9223372036854775807")));
  EXPECT_EQ(integer_to_string(int64_t(INT64_MIN)).str(),
            (StringView("-9223372036854775808")));
}

TEST(LlvmLibcIntegerToStringTest, UINT64_Base_10) {
  EXPECT_EQ((IntegerToString<uint64_t, 10>(int64_t(0)).str()), StringView("0"));
  EXPECT_EQ((IntegerToString<uint64_t, 10>(int64_t(1234567890123456789)).str()),
            StringView("1234567890123456789"));
}

TEST(LlvmLibcIntegerToStringTest, UINT64_Base_8) {
  EXPECT_EQ((IntegerToString<uint64_t, 8>(int64_t(0)).str()), StringView("0"));
  EXPECT_EQ((IntegerToString<uint64_t, 8>(int64_t(012345)).str()),
            StringView("12345"));
  EXPECT_EQ(
      (IntegerToString<uint64_t, 8>(int64_t(0123456701234567012345)).str()),
      StringView("123456701234567012345"));
  EXPECT_EQ(
      (IntegerToString<uint64_t, 8>(int64_t(01777777777777777777777)).str()),
      StringView("1777777777777777777777"));
}

TEST(LlvmLibcIntegerToStringTest, UINT64_Base_16) {
  EXPECT_EQ((IntegerToString<uint64_t, 16>(int64_t(0)).str()), StringView("0"));
  EXPECT_EQ((IntegerToString<uint64_t, 16>(int64_t(0x12345)).str()),
            StringView("12345"));
  EXPECT_EQ((IntegerToString<uint64_t, 16>(int64_t(0x123456789abcdef)).str()),
            StringView("123456789abcdef"));
  EXPECT_EQ(
      (IntegerToString<uint64_t, 16>(int64_t(0x123456789abcdef), false).str()),
      StringView("123456789ABCDEF"));
  EXPECT_EQ((IntegerToString<uint64_t, 16>(int64_t(0xffffffffffffffff)).str()),
            StringView("ffffffffffffffff"));
}

TEST(LlvmLibcIntegerToStringTest, UINT64_Base_2) {
  EXPECT_EQ((IntegerToString<uint64_t, 2>(int64_t(0)).str()), StringView("0"));
  EXPECT_EQ((IntegerToString<uint64_t, 2>(int64_t(0xf0c)).str()),
            StringView("111100001100"));
  EXPECT_EQ((IntegerToString<uint64_t, 2>(int64_t(0x123abc)).str()),
            StringView("100100011101010111100"));
  EXPECT_EQ(
      (IntegerToString<uint64_t, 2>(int64_t(0xffffffffffffffff)).str()),
      StringView(
          "1111111111111111111111111111111111111111111111111111111111111111"));
}

TEST(LlvmLibcIntegerToStringTest, UINT64_Base_36) {
  EXPECT_EQ((IntegerToString<uint64_t, 36>(int64_t(0)).str()), StringView("0"));
  EXPECT_EQ((IntegerToString<uint64_t, 36>(int64_t(12345)).str()),
            StringView("9ix"));
  EXPECT_EQ((IntegerToString<uint64_t, 36>(int64_t(1047601316295595)).str()),
            StringView("abcdefghij"));
  EXPECT_EQ((IntegerToString<uint64_t, 36>(int64_t(2092218013456445)).str()),
            StringView("klmnopqrst"));
  EXPECT_EQ((IntegerToString<uint64_t, 36>(int64_t(1867590395), false).str()),
            StringView("UVWXYZ"));
  EXPECT_EQ((IntegerToString<uint64_t, 36>(int64_t(0xffffffffffffffff)).str()),
            StringView("3w5e11264sgsf"));
}
