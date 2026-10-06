//===-- Unittests for IntegerToString -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/CPP/limits.h"
#include "src/__support/CPP/span.h"
#include "src/__support/CPP/string_view.h"
#include "src/__support/big_int.h"
#include "src/__support/integer_literals.h"
#include "src/__support/integer_to_string.h"
#include "src/__support/macros/properties/types.h"
#include "src/__support/uint128.h"

#include "test/UnitTest/CharLiteralUtils.h"
#include "test/UnitTest/Test.h"

using LIBC_NAMESPACE::BigInt;
using LIBC_NAMESPACE::IntegerToString;
using LIBC_NAMESPACE::cpp::basic_string_view;
using LIBC_NAMESPACE::cpp::span;
using LIBC_NAMESPACE::radix::Bin;
using LIBC_NAMESPACE::radix::Custom;
using LIBC_NAMESPACE::radix::Dec;
using LIBC_NAMESPACE::radix::Hex;
using LIBC_NAMESPACE::radix::Oct;
using LIBC_NAMESPACE::operator""_u128;
using LIBC_NAMESPACE::operator""_u256;

#define EXPECT(type, value, string_value)                                      \
  {                                                                            \
    const type buffer(value);                                                  \
    decltype(buffer.view()) expected = string_value;                           \
    EXPECT_EQ(buffer.view(), expected);                                        \
  }

#if defined(LIBC_TYPES_WCHAR_T_IS_UTF32)
using TestCharTypes = LIBC_NAMESPACE::testing::TypeList<char, wchar_t>;
#else
using TestCharTypes = LIBC_NAMESPACE::testing::TypeList<char>;
#endif

TYPED_TEST(LlvmLibcIntegerToStringTest, UINT8, TestCharTypes) {
  using CharT = ParamType;
  using type = IntegerToString<uint8_t, CharT>;
  EXPECT(type, 0, ENCODED(CharT, "0"));
  EXPECT(type, 1, ENCODED(CharT, "1"));
  EXPECT(type, 12, ENCODED(CharT, "12"));
  EXPECT(type, 123, ENCODED(CharT, "123"));
  EXPECT(type, UINT8_MAX, ENCODED(CharT, "255"));
  EXPECT(type, static_cast<uint8_t>(-1), ENCODED(CharT, "255"));
}

TYPED_TEST(LlvmLibcIntegerToStringTest, INT8, TestCharTypes) {
  using CharT = ParamType;
  using type = IntegerToString<int8_t, CharT>;
  EXPECT(type, 0, ENCODED(CharT, "0"));
  EXPECT(type, 1, ENCODED(CharT, "1"));
  EXPECT(type, 12, ENCODED(CharT, "12"));
  EXPECT(type, 123, ENCODED(CharT, "123"));
  EXPECT(type, -12, ENCODED(CharT, "-12"));
  EXPECT(type, -123, ENCODED(CharT, "-123"));
  EXPECT(type, INT8_MAX, ENCODED(CharT, "127"));
  EXPECT(type, INT8_MIN, ENCODED(CharT, "-128"));
}

TYPED_TEST(LlvmLibcIntegerToStringTest, UINT16, TestCharTypes) {
  using CharT = ParamType;
  using type = IntegerToString<uint16_t, CharT>;
  EXPECT(type, 0, ENCODED(CharT, "0"));
  EXPECT(type, 1, ENCODED(CharT, "1"));
  EXPECT(type, 12, ENCODED(CharT, "12"));
  EXPECT(type, 123, ENCODED(CharT, "123"));
  EXPECT(type, 1234, ENCODED(CharT, "1234"));
  EXPECT(type, 12345, ENCODED(CharT, "12345"));
  EXPECT(type, UINT16_MAX, ENCODED(CharT, "65535"));
  EXPECT(type, static_cast<uint16_t>(-1), ENCODED(CharT, "65535"));
}

TYPED_TEST(LlvmLibcIntegerToStringTest, INT16, TestCharTypes) {
  using CharT = ParamType;
  using type = IntegerToString<int16_t, CharT>;
  EXPECT(type, 0, ENCODED(CharT, "0"));
  EXPECT(type, 1, ENCODED(CharT, "1"));
  EXPECT(type, 12, ENCODED(CharT, "12"));
  EXPECT(type, 123, ENCODED(CharT, "123"));
  EXPECT(type, 1234, ENCODED(CharT, "1234"));
  EXPECT(type, 12345, ENCODED(CharT, "12345"));
  EXPECT(type, -1, ENCODED(CharT, "-1"));
  EXPECT(type, -12, ENCODED(CharT, "-12"));
  EXPECT(type, -123, ENCODED(CharT, "-123"));
  EXPECT(type, -1234, ENCODED(CharT, "-1234"));
  EXPECT(type, -12345, ENCODED(CharT, "-12345"));
  EXPECT(type, INT16_MAX, ENCODED(CharT, "32767"));
  EXPECT(type, INT16_MIN, ENCODED(CharT, "-32768"));
}

TYPED_TEST(LlvmLibcIntegerToStringTest, UINT32, TestCharTypes) {
  using CharT = ParamType;
  using type = IntegerToString<uint32_t, CharT>;
  EXPECT(type, 0, ENCODED(CharT, "0"));
  EXPECT(type, 1, ENCODED(CharT, "1"));
  EXPECT(type, 12, ENCODED(CharT, "12"));
  EXPECT(type, 123, ENCODED(CharT, "123"));
  EXPECT(type, 1234, ENCODED(CharT, "1234"));
  EXPECT(type, 12345, ENCODED(CharT, "12345"));
  EXPECT(type, 123456, ENCODED(CharT, "123456"));
  EXPECT(type, 1234567, ENCODED(CharT, "1234567"));
  EXPECT(type, 12345678, ENCODED(CharT, "12345678"));
  EXPECT(type, 123456789, ENCODED(CharT, "123456789"));
  EXPECT(type, 1234567890, ENCODED(CharT, "1234567890"));
  EXPECT(type, UINT32_MAX, ENCODED(CharT, "4294967295"));
  EXPECT(type, static_cast<uint32_t>(-1), ENCODED(CharT, "4294967295"));
}

TYPED_TEST(LlvmLibcIntegerToStringTest, INT32, TestCharTypes) {
  using CharT = ParamType;
  using type = IntegerToString<int32_t, CharT>;
  EXPECT(type, 0, ENCODED(CharT, "0"));
  EXPECT(type, 1, ENCODED(CharT, "1"));
  EXPECT(type, 12, ENCODED(CharT, "12"));
  EXPECT(type, 123, ENCODED(CharT, "123"));
  EXPECT(type, 1234, ENCODED(CharT, "1234"));
  EXPECT(type, 12345, ENCODED(CharT, "12345"));
  EXPECT(type, 123456, ENCODED(CharT, "123456"));
  EXPECT(type, 1234567, ENCODED(CharT, "1234567"));
  EXPECT(type, 12345678, ENCODED(CharT, "12345678"));
  EXPECT(type, 123456789, ENCODED(CharT, "123456789"));
  EXPECT(type, 1234567890, ENCODED(CharT, "1234567890"));
  EXPECT(type, -1, ENCODED(CharT, "-1"));
  EXPECT(type, -12, ENCODED(CharT, "-12"));
  EXPECT(type, -123, ENCODED(CharT, "-123"));
  EXPECT(type, -1234, ENCODED(CharT, "-1234"));
  EXPECT(type, -12345, ENCODED(CharT, "-12345"));
  EXPECT(type, -123456, ENCODED(CharT, "-123456"));
  EXPECT(type, -1234567, ENCODED(CharT, "-1234567"));
  EXPECT(type, -12345678, ENCODED(CharT, "-12345678"));
  EXPECT(type, -123456789, ENCODED(CharT, "-123456789"));
  EXPECT(type, -1234567890, ENCODED(CharT, "-1234567890"));
  EXPECT(type, INT32_MAX, ENCODED(CharT, "2147483647"));
  EXPECT(type, INT32_MIN, ENCODED(CharT, "-2147483648"));
}

TYPED_TEST(LlvmLibcIntegerToStringTest, UINT64, TestCharTypes) {
  using CharT = ParamType;
  using type = IntegerToString<uint64_t, CharT>;
  EXPECT(type, 0, ENCODED(CharT, "0"));
  EXPECT(type, 1, ENCODED(CharT, "1"));
  EXPECT(type, 12, ENCODED(CharT, "12"));
  EXPECT(type, 123, ENCODED(CharT, "123"));
  EXPECT(type, 1234, ENCODED(CharT, "1234"));
  EXPECT(type, 12345, ENCODED(CharT, "12345"));
  EXPECT(type, 123456, ENCODED(CharT, "123456"));
  EXPECT(type, 1234567, ENCODED(CharT, "1234567"));
  EXPECT(type, 12345678, ENCODED(CharT, "12345678"));
  EXPECT(type, 123456789, ENCODED(CharT, "123456789"));
  EXPECT(type, 1234567890, ENCODED(CharT, "1234567890"));
  EXPECT(type, 1234567890123456789, ENCODED(CharT, "1234567890123456789"));
  EXPECT(type, UINT64_MAX, ENCODED(CharT, "18446744073709551615"));
  EXPECT(type, static_cast<uint64_t>(-1),
         ENCODED(CharT, "18446744073709551615"));
}

TYPED_TEST(LlvmLibcIntegerToStringTest, INT64, TestCharTypes) {
  using CharT = ParamType;
  using type = IntegerToString<int64_t, CharT>;
  EXPECT(type, 0, ENCODED(CharT, "0"));
  EXPECT(type, 1, ENCODED(CharT, "1"));
  EXPECT(type, 12, ENCODED(CharT, "12"));
  EXPECT(type, 123, ENCODED(CharT, "123"));
  EXPECT(type, 1234, ENCODED(CharT, "1234"));
  EXPECT(type, 12345, ENCODED(CharT, "12345"));
  EXPECT(type, 123456, ENCODED(CharT, "123456"));
  EXPECT(type, 1234567, ENCODED(CharT, "1234567"));
  EXPECT(type, 12345678, ENCODED(CharT, "12345678"));
  EXPECT(type, 123456789, ENCODED(CharT, "123456789"));
  EXPECT(type, 1234567890, ENCODED(CharT, "1234567890"));
  EXPECT(type, 1234567890123456789, ENCODED(CharT, "1234567890123456789"));
  EXPECT(type, -1, ENCODED(CharT, "-1"));
  EXPECT(type, -12, ENCODED(CharT, "-12"));
  EXPECT(type, -123, ENCODED(CharT, "-123"));
  EXPECT(type, -1234, ENCODED(CharT, "-1234"));
  EXPECT(type, -12345, ENCODED(CharT, "-12345"));
  EXPECT(type, -123456, ENCODED(CharT, "-123456"));
  EXPECT(type, -1234567, ENCODED(CharT, "-1234567"));
  EXPECT(type, -12345678, ENCODED(CharT, "-12345678"));
  EXPECT(type, -123456789, ENCODED(CharT, "-123456789"));
  EXPECT(type, -1234567890, ENCODED(CharT, "-1234567890"));
  EXPECT(type, -1234567890123456789, ENCODED(CharT, "-1234567890123456789"));
  EXPECT(type, INT64_MAX, ENCODED(CharT, "9223372036854775807"));
  EXPECT(type, INT64_MIN, ENCODED(CharT, "-9223372036854775808"));
}

TYPED_TEST(LlvmLibcIntegerToStringTest, UINT64_Base_8, TestCharTypes) {
  using CharT = ParamType;
  using type = IntegerToString<int64_t, CharT, Oct>;
  EXPECT(type, 0, ENCODED(CharT, "0"));
  EXPECT(type, 012345, ENCODED(CharT, "12345"));
  EXPECT(type, 0123456701234567012345, ENCODED(CharT, "123456701234567012345"));
  EXPECT(type, static_cast<int64_t>(01777777777777777777777),
         ENCODED(CharT, "1777777777777777777777"));
}

TYPED_TEST(LlvmLibcIntegerToStringTest, UINT64_Base_16, TestCharTypes) {
  using CharT = ParamType;
  using type = IntegerToString<uint64_t, CharT, Hex>;
  EXPECT(type, 0, ENCODED(CharT, "0"));
  EXPECT(type, 0x12345, ENCODED(CharT, "12345"));
  EXPECT(type, 0x123456789abcdef, ENCODED(CharT, "123456789abcdef"));
  EXPECT(type, 0xffffffffffffffff, ENCODED(CharT, "ffffffffffffffff"));
  using TYPE = IntegerToString<uint64_t, CharT, Hex::Uppercase>;
  EXPECT(TYPE, 0x123456789abcdef, ENCODED(CharT, "123456789ABCDEF"));
}

TYPED_TEST(LlvmLibcIntegerToStringTest, UINT64_Base_2, TestCharTypes) {
  using CharT = ParamType;
  using type = IntegerToString<uint64_t, CharT, Bin>;
  EXPECT(type, 0, ENCODED(CharT, "0"));
  EXPECT(type, 0b111100001100, ENCODED(CharT, "111100001100"));
  EXPECT(type, 0b100100011101010111100,
         ENCODED(CharT, "100100011101010111100"));
  EXPECT(
      type, 0xffffffffffffffff,
      ENCODED(
          CharT,
          "1111111111111111111111111111111111111111111111111111111111111111"));
}

TYPED_TEST(LlvmLibcIntegerToStringTest, UINT128_Base_16, TestCharTypes) {
  using CharT = ParamType;
  using type = IntegerToString<UInt128, CharT, Hex::WithWidth<32>>;
  EXPECT(type, 0, ENCODED(CharT, "00000000000000000000000000000000"));
  EXPECT(type, 0x12345, ENCODED(CharT, "00000000000000000000000000012345"));
  EXPECT(type, 0x12340000'00000000'00000000'00000000_u128,
         ENCODED(CharT, "12340000000000000000000000000000"));
  EXPECT(type, 0x00000000'00000000'12340000'00000000_u128,
         ENCODED(CharT, "00000000000000001234000000000000"));
  EXPECT(type, 0x00000000'00000001'23400000'00000000_u128,
         ENCODED(CharT, "00000000000000012340000000000000"));
}

TYPED_TEST(LlvmLibcIntegerToStringTest, UINT64_Base_36, TestCharTypes) {
  using CharT = ParamType;
  using type = IntegerToString<uint64_t, CharT, Custom<36>>;
  EXPECT(type, 0, ENCODED(CharT, "0"));
  EXPECT(type, 12345, ENCODED(CharT, "9ix"));
  EXPECT(type, 1047601316295595, ENCODED(CharT, "abcdefghij"));
  EXPECT(type, 2092218013456445, ENCODED(CharT, "klmnopqrst"));
  EXPECT(type, 0xffffffffffffffff, ENCODED(CharT, "3w5e11264sgsf"));

  using TYPE = IntegerToString<uint64_t, CharT, Custom<36>::Uppercase>;
  EXPECT(TYPE, 1867590395, ENCODED(CharT, "UVWXYZ"));
}

TYPED_TEST(LlvmLibcIntegerToStringTest, UINT256_Base_16, TestCharTypes) {
  using CharT = ParamType;
  using UInt256 = LIBC_NAMESPACE::UInt<256>;
  using type = IntegerToString<UInt256, CharT, Hex::WithWidth<64>>;
  EXPECT(
      type,
      0x0000000000000000000000000000000000000000000000000000000000000000_u256,
      ENCODED(
          CharT,
          "0000000000000000000000000000000000000000000000000000000000000000"));
  EXPECT(
      type,
      0x0000000000000000000000000000000000000000000000000000000000012345_u256,
      ENCODED(
          CharT,
          "0000000000000000000000000000000000000000000000000000000000012345"));
  EXPECT(
      type,
      0x0000000000000000000000000000000012340000000000000000000000000000_u256,
      ENCODED(
          CharT,
          "0000000000000000000000000000000012340000000000000000000000000000"));
  EXPECT(
      type,
      0x0000000000000000000000000000000123400000000000000000000000000000_u256,
      ENCODED(
          CharT,
          "0000000000000000000000000000000123400000000000000000000000000000"));
  EXPECT(
      type,
      0x1234000000000000000000000000000000000000000000000000000000000000_u256,
      ENCODED(
          CharT,
          "1234000000000000000000000000000000000000000000000000000000000000"));
}

TYPED_TEST(LlvmLibcIntegerToStringTest, NegativeInterpretedAsPositive,
           TestCharTypes) {
  using CharT = ParamType;
  using BIN = IntegerToString<int8_t, CharT, Bin>;
  using OCT = IntegerToString<int8_t, CharT, Oct>;
  using DEC = IntegerToString<int8_t, CharT, Dec>;
  using HEX = IntegerToString<int8_t, CharT, Hex>;
  EXPECT(BIN, -1, ENCODED(CharT, "11111111"));
  EXPECT(OCT, -1, ENCODED(CharT, "377"));
  EXPECT(DEC, -1, ENCODED(CharT, "-1")); // Only DEC format negative values
  EXPECT(HEX, -1, ENCODED(CharT, "ff"));
}

TYPED_TEST(LlvmLibcIntegerToStringTest, Width, TestCharTypes) {
  using CharT = ParamType;
  using BIN = IntegerToString<uint8_t, CharT, Bin::WithWidth<4>>;
  using OCT = IntegerToString<uint8_t, CharT, Oct::WithWidth<4>>;
  using DEC = IntegerToString<uint8_t, CharT, Dec::WithWidth<4>>;
  using HEX = IntegerToString<uint8_t, CharT, Hex::WithWidth<4>>;
  EXPECT(BIN, 1, ENCODED(CharT, "0001"));
  EXPECT(HEX, 1, ENCODED(CharT, "0001"));
  EXPECT(OCT, 1, ENCODED(CharT, "0001"));
  EXPECT(DEC, 1, ENCODED(CharT, "0001"));
}

TYPED_TEST(LlvmLibcIntegerToStringTest, Prefix, TestCharTypes) {
  using CharT = ParamType;
  // WithPrefix is not supported for Decimal
  using BIN = IntegerToString<uint8_t, CharT, Bin::WithPrefix>;
  using OCT = IntegerToString<uint8_t, CharT, Oct::WithPrefix>;
  using HEX = IntegerToString<uint8_t, CharT, Hex::WithPrefix>;
  EXPECT(BIN, 1, ENCODED(CharT, "0b1"));
  EXPECT(HEX, 1, ENCODED(CharT, "0x1"));
  EXPECT(OCT, 1, ENCODED(CharT, "01"));
  EXPECT(OCT, 0, ENCODED(CharT, "0")); // Zero is not prefixed for octal
}

TYPED_TEST(LlvmLibcIntegerToStringTest, Uppercase, TestCharTypes) {
  using CharT = ParamType;
  using HEX = IntegerToString<uint64_t, CharT, Hex::Uppercase>;
  EXPECT(HEX, 0xDEADC0DE, ENCODED(CharT, "DEADC0DE"));
}

TYPED_TEST(LlvmLibcIntegerToStringTest, Sign, TestCharTypes) {
  using CharT = ParamType;
  // WithSign only compiles with DEC
  using DEC = IntegerToString<int8_t, CharT, Dec::WithSign>;
  EXPECT(DEC, -1, ENCODED(CharT, "-1"));
  EXPECT(DEC, 0, ENCODED(CharT, "+0"));
  EXPECT(DEC, 1, ENCODED(CharT, "+1"));
}

TYPED_TEST(LlvmLibcIntegerToStringTest, BigInt_Base_10, TestCharTypes) {
  using CharT = ParamType;
  uint64_t int256_max_w64[4] = {
      0xFFFFFFFFFFFFFFFF,
      0xFFFFFFFFFFFFFFFF,
      0xFFFFFFFFFFFFFFFF,
      0x7FFFFFFFFFFFFFFF,
  };
  uint64_t int256_min_w64[4] = {
      0,
      0,
      0,
      0x8000000000000000,
  };
  uint32_t int256_max_w32[8] = {
      0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF,
      0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0x7FFFFFFF,
  };
  uint32_t int256_min_w32[8] = {
      0, 0, 0, 0, 0, 0, 0, 0x80000000,
  };
  uint16_t int256_max_w16[16] = {
      0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF,
      0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0x7FFF,
  };
  uint16_t int256_min_w16[16] = {
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0x8000,
  };

  using unsigned_type_w64 =
      IntegerToString<BigInt<256, false, uint64_t>, CharT, Dec>;
  EXPECT(unsigned_type_w64, 0, ENCODED(CharT, "0"));
  EXPECT(unsigned_type_w64, 1, ENCODED(CharT, "1"));
  EXPECT(unsigned_type_w64, -1,
         ENCODED(CharT, "115792089237316195423570985008687907853269984665640564"
                        "039457584007913129639935"));
  EXPECT(unsigned_type_w64, int256_max_w64,
         ENCODED(CharT, "578960446186580977117854925043439539266349923328202820"
                        "19728792003956564819967"));
  EXPECT(unsigned_type_w64, int256_min_w64,
         ENCODED(CharT, "578960446186580977117854925043439539266349923328202820"
                        "19728792003956564819968"));

  using unsigned_type_w32 =
      IntegerToString<BigInt<256, false, uint32_t>, CharT, Dec>;
  EXPECT(unsigned_type_w32, 0, ENCODED(CharT, "0"));
  EXPECT(unsigned_type_w32, 1, ENCODED(CharT, "1"));
  EXPECT(unsigned_type_w32, -1,
         ENCODED(CharT, "115792089237316195423570985008687907853269984665640564"
                        "039457584007913129639935"));
  EXPECT(unsigned_type_w32, int256_max_w32,
         ENCODED(CharT, "578960446186580977117854925043439539266349923328202820"
                        "19728792003956564819967"));
  EXPECT(unsigned_type_w32, int256_min_w32,
         ENCODED(CharT, "578960446186580977117854925043439539266349923328202820"
                        "19728792003956564819968"));

  using unsigned_type_w16 =
      IntegerToString<BigInt<256, false, uint16_t>, CharT, Dec>;
  EXPECT(unsigned_type_w16, 0, ENCODED(CharT, "0"));
  EXPECT(unsigned_type_w16, 1, ENCODED(CharT, "1"));
  EXPECT(unsigned_type_w16, -1,
         ENCODED(CharT, "115792089237316195423570985008687907853269984665640564"
                        "039457584007913129639935"));
  EXPECT(unsigned_type_w16, int256_max_w16,
         ENCODED(CharT, "578960446186580977117854925043439539266349923328202820"
                        "19728792003956564819967"));
  EXPECT(unsigned_type_w16, int256_min_w16,
         ENCODED(CharT, "578960446186580977117854925043439539266349923328202820"
                        "19728792003956564819968"));

  using signed_type_w64 =
      IntegerToString<BigInt<256, true, uint64_t>, CharT, Dec>;
  EXPECT(signed_type_w64, 0, ENCODED(CharT, "0"));
  EXPECT(signed_type_w64, 1, ENCODED(CharT, "1"));
  EXPECT(signed_type_w64, -1, ENCODED(CharT, "-1"));
  EXPECT(signed_type_w64, int256_max_w64,
         ENCODED(CharT, "578960446186580977117854925043439539266349923328202820"
                        "19728792003956564819967"));
  EXPECT(signed_type_w64, int256_min_w64,
         ENCODED(CharT, "-57896044618658097711785492504343953926634992332820282"
                        "019728792003956564819968"));

  using signed_type_w32 =
      IntegerToString<BigInt<256, true, uint32_t>, CharT, Dec>;
  EXPECT(signed_type_w32, 0, ENCODED(CharT, "0"));
  EXPECT(signed_type_w32, 1, ENCODED(CharT, "1"));
  EXPECT(signed_type_w32, -1, ENCODED(CharT, "-1"));
  EXPECT(signed_type_w32, int256_max_w32,
         ENCODED(CharT, "578960446186580977117854925043439539266349923328202820"
                        "19728792003956564819967"));
  EXPECT(signed_type_w32, int256_min_w32,
         ENCODED(CharT, "-57896044618658097711785492504343953926634992332820282"
                        "019728792003956564819968"));

  using signed_type_w16 =
      IntegerToString<BigInt<256, true, uint16_t>, CharT, Dec>;
  EXPECT(signed_type_w16, 0, ENCODED(CharT, "0"));
  EXPECT(signed_type_w16, 1, ENCODED(CharT, "1"));
  EXPECT(signed_type_w16, -1, ENCODED(CharT, "-1"));
  EXPECT(signed_type_w16, int256_max_w16,
         ENCODED(CharT, "578960446186580977117854925043439539266349923328202820"
                        "19728792003956564819967"));
  EXPECT(signed_type_w16, int256_min_w16,
         ENCODED(CharT, "-57896044618658097711785492504343953926634992332820282"
                        "019728792003956564819968"));
}

TYPED_TEST(LlvmLibcIntegerToStringTest, BufferOverrun, TestCharTypes) {
  using CharT = ParamType;
  { // Writing '0' in an empty buffer requiring zero digits : works
    const auto view = IntegerToString<int, CharT, Dec::WithWidth<0>>::format_to(
        span<CharT>(), 0);
    ASSERT_TRUE(view.has_value());
    ASSERT_EQ(*view, basic_string_view<CharT>());
  }
  CharT buffer[1];
  { // Writing '1' in a buffer of one char : works
    const auto view = IntegerToString<int, CharT>::format_to(buffer, 1);
    ASSERT_TRUE(view.has_value());
    ASSERT_EQ(*view, basic_string_view<CharT>(ENCODED(CharT, "1")));
  }
  { // Writing '11' in a buffer of one char : fails
    const auto view = IntegerToString<int, CharT>::format_to(buffer, 11);
    ASSERT_FALSE(view.has_value());
  }
}
