//===-- Unittests for the FXBits class ------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "include/llvm-libc-macros/stdfix-macros.h"

#include "src/__support/fixed_point/fx_bits.h"
// #include "src/__support/FPUtil/fx_bits_str.h"
#include "src/__support/integer_literals.h"
#include "test/UnitTest/Test.h"

using LIBC_NAMESPACE::fixed_point::FXBits;
using LIBC_NAMESPACE::fixed_point::FXRep;

using LIBC_NAMESPACE::operator""_u8;
using LIBC_NAMESPACE::operator""_u16;
using LIBC_NAMESPACE::operator""_u32;
using LIBC_NAMESPACE::operator""_u64;
using LIBC_NAMESPACE::operator""_u128;

TEST(LlvmLibcFxBitsTest, FXBits_UnsignedShortFract) {
  auto bits_var = FXBits<unsigned short fract>(0x00_u8);

  EXPECT_EQ(bits_var.get_sign(), 0x00_u8);
  EXPECT_EQ(bits_var.get_integral(), 0x00_u8);
  EXPECT_EQ(bits_var.get_fraction(), 0x00_u8);

  // Since an unsigned fract has no sign or integral components, setting either
  // should have no effect.

  bits_var.set_sign(1);

  EXPECT_EQ(bits_var.get_sign(), 0x00_u8);
  EXPECT_EQ(bits_var.get_integral(), 0x00_u8);
  EXPECT_EQ(bits_var.get_fraction(), 0x00_u8);

  bits_var.set_integral(0xab);

  EXPECT_EQ(bits_var.get_sign(), 0x00_u8);
  EXPECT_EQ(bits_var.get_integral(), 0x00_u8);
  EXPECT_EQ(bits_var.get_fraction(), 0x00_u8);

  // but setting the fraction should work

  bits_var.set_fraction(0xcd);

  EXPECT_EQ(bits_var.get_sign(), 0x00_u8);
  EXPECT_EQ(bits_var.get_integral(), 0x00_u8);
  EXPECT_EQ(bits_var.get_fraction(), 0xcd_u8);
}

TEST(LlvmLibcFxBitsTest, FXBits_ShortAccum) {
  auto bits_var = FXBits<short accum>(0b0'00000000'0000000_u16);

  EXPECT_EQ(bits_var.get_sign(), 0x0000_u16);
  EXPECT_EQ(bits_var.get_integral(), 0x0000_u16);
  EXPECT_EQ(bits_var.get_fraction(), 0x0000_u16);

  bits_var.set_sign(0xffff); // one sign bit used

  EXPECT_EQ(bits_var.get_sign(), 0x0001_u16);
  EXPECT_EQ(bits_var.get_integral(), 0x0000_u16);
  EXPECT_EQ(bits_var.get_fraction(), 0x0000_u16);

  bits_var.set_integral(0xabcd_u16); // 8 integral bits used

  EXPECT_EQ(bits_var.get_sign(), 0x0001_u16);
  EXPECT_EQ(bits_var.get_integral(), 0x00cd_u16);
  EXPECT_EQ(bits_var.get_fraction(), 0x0000_u16);

  bits_var.set_fraction(0x21fe_u16); // 7 fract bits used

  EXPECT_EQ(bits_var.get_sign(), 0x0001_u16);
  EXPECT_EQ(bits_var.get_integral(), 0x00cd_u16);
  EXPECT_EQ(bits_var.get_fraction(), 0x007e_u16);
}

// TODO: more types
