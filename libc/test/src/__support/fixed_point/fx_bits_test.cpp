//===-- Unittests for the FXBits class ------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "include/llvm-libc-macros/stdfix-macros.h"

#include "src/__support/fixed_point/fx_bits.h"
#include "src/__support/integer_literals.h"
#include "test/UnitTest/Test.h"

using LIBC_NAMESPACE::fixed_point::FXBits;
using LIBC_NAMESPACE::fixed_point::FXRep;

using LIBC_NAMESPACE::operator""_u8;
using LIBC_NAMESPACE::operator""_u16;
using LIBC_NAMESPACE::operator""_u32;
using LIBC_NAMESPACE::operator""_u64;

class LlvmLibcFxBitsTest : public LIBC_NAMESPACE::testing::Test {
public:
  template <typename T> void testBitwiseOps() {
    EXPECT_EQ(LIBC_NAMESPACE::fixed_point::bit_and(T(0.75), T(0.375)), T(0.25));
    EXPECT_EQ(LIBC_NAMESPACE::fixed_point::bit_or(T(0.75), T(0.375)), T(0.875));
    using StorageType = typename FXRep<T>::StorageType;
    StorageType a = LIBC_NAMESPACE::cpp::bit_cast<StorageType>(T(0.75));
    a = ~a;
    EXPECT_EQ(LIBC_NAMESPACE::fixed_point::bit_not(T(0.75)),
              FXBits<T>(a).get_val());
  }
};

// -------------------------------- SHORT TESTS --------------------------------

TEST_F(LlvmLibcFxBitsTest, FXBits_UnsignedShortFract) {
  auto bits_var = FXBits<unsigned short fract>(0b00000000_u8);

  EXPECT_EQ(bits_var.get_sign(), false);
  EXPECT_EQ(bits_var.get_integral(), 0x00_u8);
  EXPECT_EQ(bits_var.get_fraction(), 0x00_u8);

  // Since an unsigned fract has no sign or integral components, setting either
  // should have no effect.

  bits_var.set_sign(true);

  EXPECT_EQ(bits_var.get_sign(), false);
  EXPECT_EQ(bits_var.get_integral(), 0x00_u8);
  EXPECT_EQ(bits_var.get_fraction(), 0x00_u8);

  bits_var.set_integral(0xab);

  EXPECT_EQ(bits_var.get_sign(), false);
  EXPECT_EQ(bits_var.get_integral(), 0x00_u8);
  EXPECT_EQ(bits_var.get_fraction(), 0x00_u8);

  // but setting the fraction should work

  bits_var.set_fraction(0xcd);

  EXPECT_EQ(bits_var.get_sign(), false);
  EXPECT_EQ(bits_var.get_integral(), 0x00_u8);
  EXPECT_EQ(bits_var.get_fraction(), 0xcd_u8);

  // Bitwise ops
  testBitwiseOps<unsigned short fract>();
}

TEST_F(LlvmLibcFxBitsTest, FXBits_UnsignedShortAccum) {
  auto bits_var = FXBits<unsigned short accum>(0b00000000'00000000_u16);

  EXPECT_EQ(bits_var.get_sign(), false);
  EXPECT_EQ(bits_var.get_integral(), 0x0000_u16);
  EXPECT_EQ(bits_var.get_fraction(), 0x0000_u16);

  bits_var.set_sign(true); // 0 sign bits used

  EXPECT_EQ(bits_var.get_sign(), false);
  EXPECT_EQ(bits_var.get_integral(), 0x0000_u16);
  EXPECT_EQ(bits_var.get_fraction(), 0x0000_u16);

  bits_var.set_integral(0xabcd); // 8 integral bits used

  EXPECT_EQ(bits_var.get_sign(), false);
  EXPECT_EQ(bits_var.get_integral(), 0x00cd_u16);
  EXPECT_EQ(bits_var.get_fraction(), 0x0000_u16);

  bits_var.set_fraction(0x21fe); // 8 fractional bits used

  EXPECT_EQ(bits_var.get_sign(), false);
  EXPECT_EQ(bits_var.get_integral(), 0x00cd_u16);
  EXPECT_EQ(bits_var.get_fraction(), 0x00fe_u16);

  // Bitwise ops
  testBitwiseOps<unsigned short accum>();
}

TEST_F(LlvmLibcFxBitsTest, FXBits_ShortFract) {
  auto bits_var = FXBits<short fract>(0b0'0000000_u8);

  EXPECT_EQ(bits_var.get_sign(), false);
  EXPECT_EQ(bits_var.get_integral(), 0x00_u8);
  EXPECT_EQ(bits_var.get_fraction(), 0x00_u8);

  bits_var.set_sign(true); // 1 sign bit used

  EXPECT_EQ(bits_var.get_sign(), true);
  EXPECT_EQ(bits_var.get_integral(), 0x00_u8);
  EXPECT_EQ(bits_var.get_fraction(), 0x00_u8);

  bits_var.set_integral(0xab); // 0 integral bits used

  EXPECT_EQ(bits_var.get_sign(), true);
  EXPECT_EQ(bits_var.get_integral(), 0x00_u8);
  EXPECT_EQ(bits_var.get_fraction(), 0x00_u8);

  bits_var.set_fraction(0xcd); // 7 fractional bits used

  EXPECT_EQ(bits_var.get_sign(), true);
  EXPECT_EQ(bits_var.get_integral(), 0x00_u8);
  EXPECT_EQ(bits_var.get_fraction(), 0x4d_u8);

  // Bitwise ops
  testBitwiseOps<short fract>();
}

TEST_F(LlvmLibcFxBitsTest, FXBits_ShortAccum) {
  auto bits_var = FXBits<short accum>(0b0'00000000'0000000_u16);

  EXPECT_EQ(bits_var.get_sign(), false);
  EXPECT_EQ(bits_var.get_integral(), 0x0000_u16);
  EXPECT_EQ(bits_var.get_fraction(), 0x0000_u16);

  bits_var.set_sign(true); // 1 sign bit used

  EXPECT_EQ(bits_var.get_sign(), true);
  EXPECT_EQ(bits_var.get_integral(), 0x0000_u16);
  EXPECT_EQ(bits_var.get_fraction(), 0x0000_u16);

  bits_var.set_integral(0xabcd); // 8 integral bits used

  EXPECT_EQ(bits_var.get_sign(), true);
  EXPECT_EQ(bits_var.get_integral(), 0x00cd_u16);
  EXPECT_EQ(bits_var.get_fraction(), 0x0000_u16);

  bits_var.set_fraction(0x21fe); // 7 fractional bits used

  EXPECT_EQ(bits_var.get_sign(), true);
  EXPECT_EQ(bits_var.get_integral(), 0x00cd_u16);
  EXPECT_EQ(bits_var.get_fraction(), 0x007e_u16);

  // Bitwise ops
  testBitwiseOps<short accum>();
}

// -------------------------------- NORMAL TESTS -------------------------------

TEST_F(LlvmLibcFxBitsTest, FXBits_UnsignedFract) {
  auto bits_var = FXBits<unsigned fract>(0b0000000000000000_u16);

  EXPECT_EQ(bits_var.get_sign(), false);
  EXPECT_EQ(bits_var.get_integral(), 0x0000_u16);
  EXPECT_EQ(bits_var.get_fraction(), 0x0000_u16);

  bits_var.set_sign(true); // 0 sign bits used

  EXPECT_EQ(bits_var.get_sign(), false);
  EXPECT_EQ(bits_var.get_integral(), 0x0000_u16);
  EXPECT_EQ(bits_var.get_fraction(), 0x0000_u16);

  bits_var.set_integral(0xabcd); // 0 integral bits used

  EXPECT_EQ(bits_var.get_sign(), false);
  EXPECT_EQ(bits_var.get_integral(), 0x0000_u16);
  EXPECT_EQ(bits_var.get_fraction(), 0x0000_u16);

  bits_var.set_fraction(0xef12); // 16 fractional bits used

  EXPECT_EQ(bits_var.get_sign(), false);
  EXPECT_EQ(bits_var.get_integral(), 0x0000_u16);
  EXPECT_EQ(bits_var.get_fraction(), 0xef12_u16);

  // Bitwise ops
  testBitwiseOps<unsigned fract>();
}

TEST_F(LlvmLibcFxBitsTest, FXBits_UnsignedAccum) {
  auto bits_var =
      FXBits<unsigned accum>(0b0000000000000000'0000000000000000_u32);

  EXPECT_EQ(bits_var.get_sign(), false);
  EXPECT_EQ(bits_var.get_integral(), 0x00000000_u32);
  EXPECT_EQ(bits_var.get_fraction(), 0x00000000_u32);

  bits_var.set_sign(true); // 0 sign bits used

  EXPECT_EQ(bits_var.get_sign(), false);
  EXPECT_EQ(bits_var.get_integral(), 0x00000000_u32);
  EXPECT_EQ(bits_var.get_fraction(), 0x00000000_u32);

  bits_var.set_integral(0xabcd); // 16 integral bits used

  EXPECT_EQ(bits_var.get_sign(), false);
  EXPECT_EQ(bits_var.get_integral(), 0x0000abcd_u32);
  EXPECT_EQ(bits_var.get_fraction(), 0x00000000_u32);

  bits_var.set_fraction(0xef12); // 16 fractional bits used

  EXPECT_EQ(bits_var.get_sign(), false);
  EXPECT_EQ(bits_var.get_integral(), 0x0000abcd_u32);
  EXPECT_EQ(bits_var.get_fraction(), 0x0000ef12_u32);

  // Bitwise ops
  testBitwiseOps<unsigned accum>();
}

TEST_F(LlvmLibcFxBitsTest, FXBits_Fract) {
  auto bits_var = FXBits<fract>(0b0'000000000000000_u16);

  EXPECT_EQ(bits_var.get_sign(), false);
  EXPECT_EQ(bits_var.get_integral(), 0x0000_u16);
  EXPECT_EQ(bits_var.get_fraction(), 0x0000_u16);

  bits_var.set_sign(true); // 1 sign bit used

  EXPECT_EQ(bits_var.get_sign(), true);
  EXPECT_EQ(bits_var.get_integral(), 0x0000_u16);
  EXPECT_EQ(bits_var.get_fraction(), 0x0000_u16);

  bits_var.set_integral(0xabcd); // 0 integral bits used

  EXPECT_EQ(bits_var.get_sign(), true);
  EXPECT_EQ(bits_var.get_integral(), 0x0000_u16);
  EXPECT_EQ(bits_var.get_fraction(), 0x0000_u16);

  bits_var.set_fraction(0xef12); // 15 fractional bits used

  EXPECT_EQ(bits_var.get_sign(), true);
  EXPECT_EQ(bits_var.get_integral(), 0x0000_u16);
  EXPECT_EQ(bits_var.get_fraction(), 0x6f12_u16);

  // Bitwise ops
  testBitwiseOps<fract>();
}

TEST_F(LlvmLibcFxBitsTest, FXBits_Accum) {
  auto bits_var = FXBits<accum>(0b0'0000000000000000'000000000000000_u32);

  EXPECT_EQ(bits_var.get_sign(), false);
  EXPECT_EQ(bits_var.get_integral(), 0x00000000_u32);
  EXPECT_EQ(bits_var.get_fraction(), 0x00000000_u32);

  bits_var.set_sign(true); // 1 sign bit used

  EXPECT_EQ(bits_var.get_sign(), true);
  EXPECT_EQ(bits_var.get_integral(), 0x00000000_u32);
  EXPECT_EQ(bits_var.get_fraction(), 0x00000000_u32);

  bits_var.set_integral(0xabcd); // 16 integral bits used

  EXPECT_EQ(bits_var.get_sign(), true);
  EXPECT_EQ(bits_var.get_integral(), 0x0000abcd_u32);
  EXPECT_EQ(bits_var.get_fraction(), 0x00000000_u32);

  bits_var.set_fraction(0xef12); // 15 fractional bits used

  EXPECT_EQ(bits_var.get_sign(), true);
  EXPECT_EQ(bits_var.get_integral(), 0x0000abcd_u32);
  EXPECT_EQ(bits_var.get_fraction(), 0x00006f12_u32);

  // Bitwise ops
  testBitwiseOps<accum>();
}

// --------------------------------- LONG TESTS --------------------------------

TEST_F(LlvmLibcFxBitsTest, FXBits_UnsignedLongFract) {
  auto bits_var =
      FXBits<unsigned long fract>(0b00000000000000000000000000000000_u32);

  EXPECT_EQ(bits_var.get_sign(), false);
  EXPECT_EQ(bits_var.get_integral(), 0x00000000_u32);
  EXPECT_EQ(bits_var.get_fraction(), 0x00000000_u32);

  bits_var.set_sign(true); // 0 sign bits used

  EXPECT_EQ(bits_var.get_sign(), false);
  EXPECT_EQ(bits_var.get_integral(), 0x00000000_u32);
  EXPECT_EQ(bits_var.get_fraction(), 0x00000000_u32);

  bits_var.set_integral(0xabcdef12); // 0 integral bits used

  EXPECT_EQ(bits_var.get_sign(), false);
  EXPECT_EQ(bits_var.get_integral(), 0x00000000_u32);
  EXPECT_EQ(bits_var.get_fraction(), 0x00000000_u32);

  bits_var.set_fraction(0xfedcba98); // 32 integral bits used

  EXPECT_EQ(bits_var.get_sign(), false);
  EXPECT_EQ(bits_var.get_integral(), 0x00000000_u32);
  EXPECT_EQ(bits_var.get_fraction(), 0xfedcba98_u32);

  // Bitwise ops
  testBitwiseOps<unsigned long fract>();
}

TEST_F(LlvmLibcFxBitsTest, FXBits_UnsignedLongAccum) {
  auto bits_var = FXBits<unsigned long accum>(
      0b00000000000000000000000000000000'00000000000000000000000000000000_u64);

  EXPECT_EQ(bits_var.get_sign(), false);
  EXPECT_EQ(bits_var.get_integral(), 0x0000000000000000_u64);
  EXPECT_EQ(bits_var.get_fraction(), 0x0000000000000000_u64);

  bits_var.set_sign(true); // 0 sign bits used

  EXPECT_EQ(bits_var.get_sign(), false);
  EXPECT_EQ(bits_var.get_integral(), 0x0000000000000000_u64);
  EXPECT_EQ(bits_var.get_fraction(), 0x0000000000000000_u64);

  bits_var.set_integral(0xabcdef12); // 32 integral bits used

  EXPECT_EQ(bits_var.get_sign(), false);
  EXPECT_EQ(bits_var.get_integral(), 0x00000000abcdef12_u64);
  EXPECT_EQ(bits_var.get_fraction(), 0x0000000000000000_u64);

  bits_var.set_fraction(0xfedcba98); // 32 fractional bits used

  EXPECT_EQ(bits_var.get_sign(), false);
  EXPECT_EQ(bits_var.get_integral(), 0x00000000abcdef12_u64);
  EXPECT_EQ(bits_var.get_fraction(), 0x00000000fedcba98_u64);

  // Bitwise ops
  testBitwiseOps<unsigned long accum>();
}

TEST_F(LlvmLibcFxBitsTest, FXBits_LongFract) {
  auto bits_var = FXBits<long fract>(0b0'0000000000000000000000000000000_u32);

  EXPECT_EQ(bits_var.get_sign(), false);
  EXPECT_EQ(bits_var.get_integral(), 0x00000000_u32);
  EXPECT_EQ(bits_var.get_fraction(), 0x00000000_u32);

  bits_var.set_sign(true); // 1 sign bit used

  EXPECT_EQ(bits_var.get_sign(), true);
  EXPECT_EQ(bits_var.get_integral(), 0x00000000_u32);
  EXPECT_EQ(bits_var.get_fraction(), 0x00000000_u32);

  bits_var.set_integral(0xabcdef12); // 0 integral bits used

  EXPECT_EQ(bits_var.get_sign(), true);
  EXPECT_EQ(bits_var.get_integral(), 0x00000000_u32);
  EXPECT_EQ(bits_var.get_fraction(), 0x00000000_u32);

  bits_var.set_fraction(0xfedcba98); // 31 fractional bits used

  EXPECT_EQ(bits_var.get_sign(), true);
  EXPECT_EQ(bits_var.get_integral(), 0x00000000_u32);
  EXPECT_EQ(bits_var.get_fraction(), 0x7edcba98_u32);

  // Bitwise ops
  testBitwiseOps<long fract>();
}

TEST_F(LlvmLibcFxBitsTest, FXBits_LongAccum) {
  auto bits_var = FXBits<long accum>(
      0b0'00000000000000000000000000000000'0000000000000000000000000000000_u64);

  EXPECT_EQ(bits_var.get_sign(), false);
  EXPECT_EQ(bits_var.get_integral(), 0x0000000000000000_u64);
  EXPECT_EQ(bits_var.get_fraction(), 0x0000000000000000_u64);

  bits_var.set_sign(true); // 1 sign bit used

  EXPECT_EQ(bits_var.get_sign(), true);
  EXPECT_EQ(bits_var.get_integral(), 0x0000000000000000_u64);
  EXPECT_EQ(bits_var.get_fraction(), 0x0000000000000000_u64);

  bits_var.set_integral(0xabcdef12); // 32 integral bits used

  EXPECT_EQ(bits_var.get_sign(), true);
  EXPECT_EQ(bits_var.get_integral(), 0x00000000abcdef12_u64);
  EXPECT_EQ(bits_var.get_fraction(), 0x0000000000000000_u64);

  bits_var.set_fraction(0xfedcba98); // 31 fractional bits used

  EXPECT_EQ(bits_var.get_sign(), true);
  EXPECT_EQ(bits_var.get_integral(), 0x00000000abcdef12_u64);
  EXPECT_EQ(bits_var.get_fraction(), 0x000000007edcba98_u64);

  // Bitwise ops
  testBitwiseOps<long accum>();
}
