//===-- Unittests for the DyadicFloat class -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/FPUtil/FPBits.h"
#include "src/__support/FPUtil/fpbits_str.h"
#include "test/UnitTest/Test.h"

using LIBC_NAMESPACE::fputil::FPBits;

TEST(LlvmLibcFPBitsTest, FloatType) {
  using FloatBits = FPBits<float>;

  EXPECT_STREQ(LIBC_NAMESPACE::str(FloatBits(FloatBits::inf())).c_str(),
               "(+Infinity)");
  EXPECT_STREQ(LIBC_NAMESPACE::str(FloatBits(FloatBits::neg_inf())).c_str(),
               "(-Infinity)");
  EXPECT_STREQ(LIBC_NAMESPACE::str(FloatBits(FloatBits::build_nan(1))).c_str(),
               "(NaN)");

  FloatBits zero(0.0f);
  EXPECT_EQ(zero.get_sign(), false);
  EXPECT_EQ(zero.get_biased_exponent(), static_cast<uint16_t>(0));
  EXPECT_EQ(zero.get_mantissa(), static_cast<uint32_t>(0));
  EXPECT_EQ(zero.uintval(), static_cast<uint32_t>(0x00000000));
  EXPECT_STREQ(LIBC_NAMESPACE::str(zero).c_str(),
               "0x00000000 = (S: 0, E: 0x0000, M: 0x00000000)");

  FloatBits negzero(-0.0f);
  EXPECT_EQ(negzero.get_sign(), true);
  EXPECT_EQ(negzero.get_biased_exponent(), static_cast<uint16_t>(0));
  EXPECT_EQ(negzero.get_mantissa(), static_cast<uint32_t>(0));
  EXPECT_EQ(negzero.uintval(), static_cast<uint32_t>(0x80000000));
  EXPECT_STREQ(LIBC_NAMESPACE::str(negzero).c_str(),
               "0x80000000 = (S: 1, E: 0x0000, M: 0x00000000)");

  FloatBits one(1.0f);
  EXPECT_EQ(one.get_sign(), false);
  EXPECT_EQ(one.get_biased_exponent(), static_cast<uint16_t>(0x7F));
  EXPECT_EQ(one.get_mantissa(), static_cast<uint32_t>(0));
  EXPECT_EQ(one.uintval(), static_cast<uint32_t>(0x3F800000));
  EXPECT_STREQ(LIBC_NAMESPACE::str(one).c_str(),
               "0x3F800000 = (S: 0, E: 0x007F, M: 0x00000000)");

  FloatBits negone(-1.0f);
  EXPECT_EQ(negone.get_sign(), true);
  EXPECT_EQ(negone.get_biased_exponent(), static_cast<uint16_t>(0x7F));
  EXPECT_EQ(negone.get_mantissa(), static_cast<uint32_t>(0));
  EXPECT_EQ(negone.uintval(), static_cast<uint32_t>(0xBF800000));
  EXPECT_STREQ(LIBC_NAMESPACE::str(negone).c_str(),
               "0xBF800000 = (S: 1, E: 0x007F, M: 0x00000000)");

  FloatBits num(1.125f);
  EXPECT_EQ(num.get_sign(), false);
  EXPECT_EQ(num.get_biased_exponent(), static_cast<uint16_t>(0x7F));
  EXPECT_EQ(num.get_mantissa(), static_cast<uint32_t>(0x00100000));
  EXPECT_EQ(num.uintval(), static_cast<uint32_t>(0x3F900000));
  EXPECT_STREQ(LIBC_NAMESPACE::str(num).c_str(),
               "0x3F900000 = (S: 0, E: 0x007F, M: 0x00100000)");

  FloatBits negnum(-1.125f);
  EXPECT_EQ(negnum.get_sign(), true);
  EXPECT_EQ(negnum.get_biased_exponent(), static_cast<uint16_t>(0x7F));
  EXPECT_EQ(negnum.get_mantissa(), static_cast<uint32_t>(0x00100000));
  EXPECT_EQ(negnum.uintval(), static_cast<uint32_t>(0xBF900000));
  EXPECT_STREQ(LIBC_NAMESPACE::str(negnum).c_str(),
               "0xBF900000 = (S: 1, E: 0x007F, M: 0x00100000)");

  FloatBits quiet_nan = FloatBits(FloatBits::build_quiet_nan(1));
  EXPECT_EQ(quiet_nan.is_quiet_nan(), true);
}

TEST(LlvmLibcFPBitsTest, DoubleType) {
  using DoubleBits = FPBits<double>;

  EXPECT_STREQ(LIBC_NAMESPACE::str(DoubleBits(DoubleBits::inf())).c_str(),
               "(+Infinity)");
  EXPECT_STREQ(LIBC_NAMESPACE::str(DoubleBits(DoubleBits::neg_inf())).c_str(),
               "(-Infinity)");
  EXPECT_STREQ(
      LIBC_NAMESPACE::str(DoubleBits(DoubleBits::build_nan(1))).c_str(),
      "(NaN)");

  DoubleBits zero(0.0);
  EXPECT_EQ(zero.get_sign(), false);
  EXPECT_EQ(zero.get_biased_exponent(), static_cast<uint16_t>(0x0000));
  EXPECT_EQ(zero.get_mantissa(), static_cast<uint64_t>(0x0000000000000000));
  EXPECT_EQ(zero.uintval(), static_cast<uint64_t>(0x0000000000000000));
  EXPECT_STREQ(LIBC_NAMESPACE::str(zero).c_str(),
               "0x0000000000000000 = (S: 0, E: 0x0000, M: 0x0000000000000000)");

  DoubleBits negzero(-0.0);
  EXPECT_EQ(negzero.get_sign(), true);
  EXPECT_EQ(negzero.get_biased_exponent(), static_cast<uint16_t>(0x0000));
  EXPECT_EQ(negzero.get_mantissa(), static_cast<uint64_t>(0x0000000000000000));
  EXPECT_EQ(negzero.uintval(), static_cast<uint64_t>(0x8000000000000000));
  EXPECT_STREQ(LIBC_NAMESPACE::str(negzero).c_str(),
               "0x8000000000000000 = (S: 1, E: 0x0000, M: 0x0000000000000000)");

  DoubleBits one(1.0);
  EXPECT_EQ(one.get_sign(), false);
  EXPECT_EQ(one.get_biased_exponent(), static_cast<uint16_t>(0x03FF));
  EXPECT_EQ(one.get_mantissa(), static_cast<uint64_t>(0x0000000000000000));
  EXPECT_EQ(one.uintval(), static_cast<uint64_t>(0x3FF0000000000000));
  EXPECT_STREQ(LIBC_NAMESPACE::str(one).c_str(),
               "0x3FF0000000000000 = (S: 0, E: 0x03FF, M: 0x0000000000000000)");

  DoubleBits negone(-1.0);
  EXPECT_EQ(negone.get_sign(), true);
  EXPECT_EQ(negone.get_biased_exponent(), static_cast<uint16_t>(0x03FF));
  EXPECT_EQ(negone.get_mantissa(), static_cast<uint64_t>(0x0000000000000000));
  EXPECT_EQ(negone.uintval(), static_cast<uint64_t>(0xBFF0000000000000));
  EXPECT_STREQ(LIBC_NAMESPACE::str(negone).c_str(),
               "0xBFF0000000000000 = (S: 1, E: 0x03FF, M: 0x0000000000000000)");

  DoubleBits num(1.125);
  EXPECT_EQ(num.get_sign(), false);
  EXPECT_EQ(num.get_biased_exponent(), static_cast<uint16_t>(0x03FF));
  EXPECT_EQ(num.get_mantissa(), static_cast<uint64_t>(0x0002000000000000));
  EXPECT_EQ(num.uintval(), static_cast<uint64_t>(0x3FF2000000000000));
  EXPECT_STREQ(LIBC_NAMESPACE::str(num).c_str(),
               "0x3FF2000000000000 = (S: 0, E: 0x03FF, M: 0x0002000000000000)");

  DoubleBits negnum(-1.125);
  EXPECT_EQ(negnum.get_sign(), true);
  EXPECT_EQ(negnum.get_biased_exponent(), static_cast<uint16_t>(0x03FF));
  EXPECT_EQ(negnum.get_mantissa(), static_cast<uint64_t>(0x0002000000000000));
  EXPECT_EQ(negnum.uintval(), static_cast<uint64_t>(0xBFF2000000000000));
  EXPECT_STREQ(LIBC_NAMESPACE::str(negnum).c_str(),
               "0xBFF2000000000000 = (S: 1, E: 0x03FF, M: 0x0002000000000000)");

  DoubleBits quiet_nan = DoubleBits(DoubleBits::build_quiet_nan(1));
  EXPECT_EQ(quiet_nan.is_quiet_nan(), true);
}

#ifdef LIBC_TARGET_ARCH_IS_X86
TEST(LlvmLibcFPBitsTest, X86LongDoubleType) {
  using LongDoubleBits = FPBits<long double>;

  if constexpr (sizeof(long double) == sizeof(double))
    return; // The tests for the "double" type cover for this case.

  EXPECT_STREQ(
      LIBC_NAMESPACE::str(LongDoubleBits(LongDoubleBits::inf())).c_str(),
      "(+Infinity)");
  EXPECT_STREQ(
      LIBC_NAMESPACE::str(LongDoubleBits(LongDoubleBits::neg_inf())).c_str(),
      "(-Infinity)");
  EXPECT_STREQ(
      LIBC_NAMESPACE::str(LongDoubleBits(LongDoubleBits::build_nan(1))).c_str(),
      "(NaN)");

  LongDoubleBits zero(0.0l);
  EXPECT_EQ(zero.get_sign(), false);
  EXPECT_EQ(zero.get_biased_exponent(), static_cast<uint16_t>(0x0000));
  EXPECT_EQ(zero.get_mantissa(), static_cast<UInt128>(0x0000000000000000)
                                     << 64);
  EXPECT_EQ(zero.uintval(), static_cast<UInt128>(0x0000000000000000) << 64);
  EXPECT_STREQ(
      LIBC_NAMESPACE::str(zero).c_str(),
      "0x00000000000000000000000000000000 = "
      "(S: 0, E: 0x0000, I: 0, M: 0x00000000000000000000000000000000)");

  LongDoubleBits negzero(-0.0l);
  EXPECT_EQ(negzero.get_sign(), true);
  EXPECT_EQ(negzero.get_biased_exponent(), static_cast<uint16_t>(0x0000));
  EXPECT_EQ(negzero.get_mantissa(), static_cast<UInt128>(0x0000000000000000)
                                        << 64);
  EXPECT_EQ(negzero.uintval(), static_cast<UInt128>(0x1) << 79);
  EXPECT_STREQ(
      LIBC_NAMESPACE::str(negzero).c_str(),
      "0x00000000000080000000000000000000 = "
      "(S: 1, E: 0x0000, I: 0, M: 0x00000000000000000000000000000000)");

  LongDoubleBits one(1.0l);
  EXPECT_EQ(one.get_sign(), false);
  EXPECT_EQ(one.get_biased_exponent(), static_cast<uint16_t>(0x3FFF));
  EXPECT_EQ(one.get_mantissa(), static_cast<UInt128>(0x0000000000000000) << 64);
  EXPECT_EQ(one.uintval(), static_cast<UInt128>(0x3FFF8) << 60);
  EXPECT_STREQ(
      LIBC_NAMESPACE::str(one).c_str(),
      "0x0000000000003FFF8000000000000000 = "
      "(S: 0, E: 0x3FFF, I: 1, M: 0x00000000000000000000000000000000)");

  LongDoubleBits negone(-1.0l);
  EXPECT_EQ(negone.get_sign(), true);
  EXPECT_EQ(negone.get_biased_exponent(), static_cast<uint16_t>(0x3FFF));
  EXPECT_EQ(negone.get_mantissa(), static_cast<UInt128>(0x0000000000000000)
                                       << 64);
  EXPECT_EQ(negone.uintval(), static_cast<UInt128>(0xBFFF8) << 60);
  EXPECT_STREQ(
      LIBC_NAMESPACE::str(negone).c_str(),
      "0x000000000000BFFF8000000000000000 = "
      "(S: 1, E: 0x3FFF, I: 1, M: 0x00000000000000000000000000000000)");

  LongDoubleBits num(1.125l);
  EXPECT_EQ(num.get_sign(), false);
  EXPECT_EQ(num.get_biased_exponent(), static_cast<uint16_t>(0x3FFF));
  EXPECT_EQ(num.get_mantissa(), static_cast<UInt128>(0x1) << 60);
  EXPECT_EQ(num.uintval(), static_cast<UInt128>(0x3FFF9) << 60);
  EXPECT_STREQ(
      LIBC_NAMESPACE::str(num).c_str(),
      "0x0000000000003FFF9000000000000000 = "
      "(S: 0, E: 0x3FFF, I: 1, M: 0x00000000000000001000000000000000)");

  LongDoubleBits negnum(-1.125l);
  EXPECT_EQ(negnum.get_sign(), true);
  EXPECT_EQ(negnum.get_biased_exponent(), static_cast<uint16_t>(0x3FFF));
  EXPECT_EQ(negnum.get_mantissa(), static_cast<UInt128>(0x1) << 60);
  EXPECT_EQ(negnum.uintval(), static_cast<UInt128>(0xBFFF9) << 60);
  EXPECT_STREQ(
      LIBC_NAMESPACE::str(negnum).c_str(),
      "0x000000000000BFFF9000000000000000 = "
      "(S: 1, E: 0x3FFF, I: 1, M: 0x00000000000000001000000000000000)");

  LongDoubleBits quiet_nan = LongDoubleBits(LongDoubleBits::build_quiet_nan(1));
  EXPECT_EQ(quiet_nan.is_quiet_nan(), true);
}
#else
TEST(LlvmLibcFPBitsTest, LongDoubleType) {
#if defined(LIBC_LONG_DOUBLE_IS_FLOAT64)
  return; // The tests for the "double" type cover for this case.
#else
  using LongDoubleBits = FPBits<long double>;

  EXPECT_STREQ(
      LIBC_NAMESPACE::str(LongDoubleBits(LongDoubleBits::inf())).c_str(),
      "(+Infinity)");
  EXPECT_STREQ(
      LIBC_NAMESPACE::str(LongDoubleBits(LongDoubleBits::neg_inf())).c_str(),
      "(-Infinity)");
  EXPECT_STREQ(
      LIBC_NAMESPACE::str(LongDoubleBits(LongDoubleBits::build_nan(1))).c_str(),
      "(NaN)");

  LongDoubleBits zero(0.0l);
  EXPECT_EQ(zero.get_sign(), false);
  EXPECT_EQ(zero.get_biased_exponent(), static_cast<uint16_t>(0x0000));
  EXPECT_EQ(zero.get_mantissa(), static_cast<UInt128>(0x0000000000000000)
                                     << 64);
  EXPECT_EQ(zero.uintval(), static_cast<UInt128>(0x0000000000000000) << 64);
  EXPECT_STREQ(LIBC_NAMESPACE::str(zero).c_str(),
               "0x00000000000000000000000000000000 = "
               "(S: 0, E: 0x0000, M: 0x00000000000000000000000000000000)");

  LongDoubleBits negzero(-0.0l);
  EXPECT_EQ(negzero.get_sign(), true);
  EXPECT_EQ(negzero.get_biased_exponent(), static_cast<uint16_t>(0x0000));
  EXPECT_EQ(negzero.get_mantissa(), static_cast<UInt128>(0x0000000000000000)
                                        << 64);
  EXPECT_EQ(negzero.uintval(), static_cast<UInt128>(0x1) << 127);
  EXPECT_STREQ(LIBC_NAMESPACE::str(negzero).c_str(),
               "0x80000000000000000000000000000000 = "
               "(S: 1, E: 0x0000, M: 0x00000000000000000000000000000000)");

  LongDoubleBits one(1.0l);
  EXPECT_EQ(one.get_sign(), false);
  EXPECT_EQ(one.get_biased_exponent(), static_cast<uint16_t>(0x3FFF));
  EXPECT_EQ(one.get_mantissa(), static_cast<UInt128>(0x0000000000000000) << 64);
  EXPECT_EQ(one.uintval(), static_cast<UInt128>(0x3FFF) << 112);
  EXPECT_STREQ(LIBC_NAMESPACE::str(one).c_str(),
               "0x3FFF0000000000000000000000000000 = "
               "(S: 0, E: 0x3FFF, M: 0x00000000000000000000000000000000)");

  LongDoubleBits negone(-1.0l);
  EXPECT_EQ(negone.get_sign(), true);
  EXPECT_EQ(negone.get_biased_exponent(), static_cast<uint16_t>(0x3FFF));
  EXPECT_EQ(negone.get_mantissa(), static_cast<UInt128>(0x0000000000000000)
                                       << 64);
  EXPECT_EQ(negone.uintval(), static_cast<UInt128>(0xBFFF) << 112);
  EXPECT_STREQ(LIBC_NAMESPACE::str(negone).c_str(),
               "0xBFFF0000000000000000000000000000 = "
               "(S: 1, E: 0x3FFF, M: 0x00000000000000000000000000000000)");

  LongDoubleBits num(1.125l);
  EXPECT_EQ(num.get_sign(), false);
  EXPECT_EQ(num.get_biased_exponent(), static_cast<uint16_t>(0x3FFF));
  EXPECT_EQ(num.get_mantissa(), static_cast<UInt128>(0x2) << 108);
  EXPECT_EQ(num.uintval(), static_cast<UInt128>(0x3FFF2) << 108);
  EXPECT_STREQ(LIBC_NAMESPACE::str(num).c_str(),
               "0x3FFF2000000000000000000000000000 = "
               "(S: 0, E: 0x3FFF, M: 0x00002000000000000000000000000000)");

  LongDoubleBits negnum(-1.125l);
  EXPECT_EQ(negnum.get_sign(), true);
  EXPECT_EQ(negnum.get_biased_exponent(), static_cast<uint16_t>(0x3FFF));
  EXPECT_EQ(negnum.get_mantissa(), static_cast<UInt128>(0x2) << 108);
  EXPECT_EQ(negnum.uintval(), static_cast<UInt128>(0xBFFF2) << 108);
  EXPECT_STREQ(LIBC_NAMESPACE::str(negnum).c_str(),
               "0xBFFF2000000000000000000000000000 = "
               "(S: 1, E: 0x3FFF, M: 0x00002000000000000000000000000000)");

  LongDoubleBits quiet_nan = LongDoubleBits(LongDoubleBits::build_quiet_nan(1));
  EXPECT_EQ(quiet_nan.is_quiet_nan(), true);
#endif
}
#endif

#if defined(LIBC_COMPILER_HAS_FLOAT128)
TEST(LlvmLibcFPBitsTest, Float128Type) {
  using Float128Bits = FPBits<float128>;

  EXPECT_STREQ(LIBC_NAMESPACE::str(Float128Bits(Float128Bits::inf())).c_str(),
               "(+Infinity)");
  EXPECT_STREQ(
      LIBC_NAMESPACE::str(Float128Bits(Float128Bits::neg_inf())).c_str(),
      "(-Infinity)");
  EXPECT_STREQ(
      LIBC_NAMESPACE::str(Float128Bits(Float128Bits::build_nan(1))).c_str(),
      "(NaN)");

  Float128Bits zero(Float128Bits::zero());
  EXPECT_EQ(zero.get_sign(), false);
  EXPECT_EQ(zero.get_biased_exponent(), static_cast<uint16_t>(0x0000));
  EXPECT_EQ(zero.get_mantissa(), static_cast<UInt128>(0x0000000000000000)
                                     << 64);
  EXPECT_EQ(zero.uintval(), static_cast<UInt128>(0x0000000000000000) << 64);
  EXPECT_STREQ(LIBC_NAMESPACE::str(zero).c_str(),
               "0x00000000000000000000000000000000 = "
               "(S: 0, E: 0x0000, M: 0x00000000000000000000000000000000)");

  Float128Bits negzero(Float128Bits::neg_zero());
  EXPECT_EQ(negzero.get_sign(), true);
  EXPECT_EQ(negzero.get_biased_exponent(), static_cast<uint16_t>(0x0000));
  EXPECT_EQ(negzero.get_mantissa(), static_cast<UInt128>(0x0000000000000000)
                                        << 64);
  EXPECT_EQ(negzero.uintval(), static_cast<UInt128>(0x1) << 127);
  EXPECT_STREQ(LIBC_NAMESPACE::str(negzero).c_str(),
               "0x80000000000000000000000000000000 = "
               "(S: 1, E: 0x0000, M: 0x00000000000000000000000000000000)");

  Float128Bits one(float128(1.0));
  EXPECT_EQ(one.get_sign(), false);
  EXPECT_EQ(one.get_biased_exponent(), static_cast<uint16_t>(0x3FFF));
  EXPECT_EQ(one.get_mantissa(), static_cast<UInt128>(0x0000000000000000) << 64);
  EXPECT_EQ(one.uintval(), static_cast<UInt128>(0x3FFF) << 112);
  EXPECT_STREQ(LIBC_NAMESPACE::str(one).c_str(),
               "0x3FFF0000000000000000000000000000 = "
               "(S: 0, E: 0x3FFF, M: 0x00000000000000000000000000000000)");

  Float128Bits negone(float128(-1.0));
  EXPECT_EQ(negone.get_sign(), true);
  EXPECT_EQ(negone.get_biased_exponent(), static_cast<uint16_t>(0x3FFF));
  EXPECT_EQ(negone.get_mantissa(), static_cast<UInt128>(0x0000000000000000)
                                       << 64);
  EXPECT_EQ(negone.uintval(), static_cast<UInt128>(0xBFFF) << 112);
  EXPECT_STREQ(LIBC_NAMESPACE::str(negone).c_str(),
               "0xBFFF0000000000000000000000000000 = "
               "(S: 1, E: 0x3FFF, M: 0x00000000000000000000000000000000)");

  Float128Bits num(float128(1.125));
  EXPECT_EQ(num.get_sign(), false);
  EXPECT_EQ(num.get_biased_exponent(), static_cast<uint16_t>(0x3FFF));
  EXPECT_EQ(num.get_mantissa(), static_cast<UInt128>(0x2) << 108);
  EXPECT_EQ(num.uintval(), static_cast<UInt128>(0x3FFF2) << 108);
  EXPECT_STREQ(LIBC_NAMESPACE::str(num).c_str(),
               "0x3FFF2000000000000000000000000000 = "
               "(S: 0, E: 0x3FFF, M: 0x00002000000000000000000000000000)");

  Float128Bits negnum(float128(-1.125));
  EXPECT_EQ(negnum.get_sign(), true);
  EXPECT_EQ(negnum.get_biased_exponent(), static_cast<uint16_t>(0x3FFF));
  EXPECT_EQ(negnum.get_mantissa(), static_cast<UInt128>(0x2) << 108);
  EXPECT_EQ(negnum.uintval(), static_cast<UInt128>(0xBFFF2) << 108);
  EXPECT_STREQ(LIBC_NAMESPACE::str(negnum).c_str(),
               "0xBFFF2000000000000000000000000000 = "
               "(S: 1, E: 0x3FFF, M: 0x00002000000000000000000000000000)");

  Float128Bits quiet_nan = Float128Bits(Float128Bits::build_quiet_nan(1));
  EXPECT_EQ(quiet_nan.is_quiet_nan(), true);
}
#endif // LIBC_COMPILER_HAS_FLOAT128
