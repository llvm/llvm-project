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
  EXPECT_STREQ(LIBC_NAMESPACE::str(FPBits<float>::inf()).c_str(),
               "(+Infinity)");
  EXPECT_STREQ(LIBC_NAMESPACE::str(FPBits<float>::neg_inf()).c_str(),
               "(-Infinity)");
  EXPECT_STREQ(
      LIBC_NAMESPACE::str(FPBits<float>(FPBits<float>::build_nan(1))).c_str(),
      "(NaN)");

  FPBits<float> zero(0.0f);
  EXPECT_EQ(zero.get_sign(), false);
  EXPECT_EQ(zero.get_unbiased_exponent(), static_cast<uint16_t>(0));
  EXPECT_EQ(zero.get_mantissa(), static_cast<uint32_t>(0));
  EXPECT_EQ(zero.uintval(), static_cast<uint32_t>(0x00000000));
  EXPECT_STREQ(LIBC_NAMESPACE::str(zero).c_str(),
               "0x00000000 = (S: 0, E: 0x0000, M: 0x00000000)");

  FPBits<float> negzero(-0.0f);
  EXPECT_EQ(negzero.get_sign(), true);
  EXPECT_EQ(negzero.get_unbiased_exponent(), static_cast<uint16_t>(0));
  EXPECT_EQ(negzero.get_mantissa(), static_cast<uint32_t>(0));
  EXPECT_EQ(negzero.uintval(), static_cast<uint32_t>(0x80000000));
  EXPECT_STREQ(LIBC_NAMESPACE::str(negzero).c_str(),
               "0x80000000 = (S: 1, E: 0x0000, M: 0x00000000)");

  FPBits<float> one(1.0f);
  EXPECT_EQ(one.get_sign(), false);
  EXPECT_EQ(one.get_unbiased_exponent(), static_cast<uint16_t>(0x7F));
  EXPECT_EQ(one.get_mantissa(), static_cast<uint32_t>(0));
  EXPECT_EQ(one.uintval(), static_cast<uint32_t>(0x3F800000));
  EXPECT_STREQ(LIBC_NAMESPACE::str(one).c_str(),
               "0x3F800000 = (S: 0, E: 0x007F, M: 0x00000000)");

  FPBits<float> negone(-1.0f);
  EXPECT_EQ(negone.get_sign(), true);
  EXPECT_EQ(negone.get_unbiased_exponent(), static_cast<uint16_t>(0x7F));
  EXPECT_EQ(negone.get_mantissa(), static_cast<uint32_t>(0));
  EXPECT_EQ(negone.uintval(), static_cast<uint32_t>(0xBF800000));
  EXPECT_STREQ(LIBC_NAMESPACE::str(negone).c_str(),
               "0xBF800000 = (S: 1, E: 0x007F, M: 0x00000000)");

  FPBits<float> num(1.125f);
  EXPECT_EQ(num.get_sign(), false);
  EXPECT_EQ(num.get_unbiased_exponent(), static_cast<uint16_t>(0x7F));
  EXPECT_EQ(num.get_mantissa(), static_cast<uint32_t>(0x00100000));
  EXPECT_EQ(num.uintval(), static_cast<uint32_t>(0x3F900000));
  EXPECT_STREQ(LIBC_NAMESPACE::str(num).c_str(),
               "0x3F900000 = (S: 0, E: 0x007F, M: 0x00100000)");

  FPBits<float> negnum(-1.125f);
  EXPECT_EQ(negnum.get_sign(), true);
  EXPECT_EQ(negnum.get_unbiased_exponent(), static_cast<uint16_t>(0x7F));
  EXPECT_EQ(negnum.get_mantissa(), static_cast<uint32_t>(0x00100000));
  EXPECT_EQ(negnum.uintval(), static_cast<uint32_t>(0xBF900000));
  EXPECT_STREQ(LIBC_NAMESPACE::str(negnum).c_str(),
               "0xBF900000 = (S: 1, E: 0x007F, M: 0x00100000)");
}

TEST(LlvmLibcFPBitsTest, DoubleType) {
  EXPECT_STREQ(LIBC_NAMESPACE::str(FPBits<double>::inf()).c_str(),
               "(+Infinity)");
  EXPECT_STREQ(LIBC_NAMESPACE::str(FPBits<double>::neg_inf()).c_str(),
               "(-Infinity)");
  EXPECT_STREQ(
      LIBC_NAMESPACE::str(FPBits<double>(FPBits<double>::build_nan(1))).c_str(),
      "(NaN)");

  FPBits<double> zero(0.0);
  EXPECT_EQ(zero.get_sign(), false);
  EXPECT_EQ(zero.get_unbiased_exponent(), static_cast<uint16_t>(0x0000));
  EXPECT_EQ(zero.get_mantissa(), static_cast<uint64_t>(0x0000000000000000));
  EXPECT_EQ(zero.uintval(), static_cast<uint64_t>(0x0000000000000000));
  EXPECT_STREQ(LIBC_NAMESPACE::str(zero).c_str(),
               "0x0000000000000000 = (S: 0, E: 0x0000, M: 0x0000000000000000)");

  FPBits<double> negzero(-0.0);
  EXPECT_EQ(negzero.get_sign(), true);
  EXPECT_EQ(negzero.get_unbiased_exponent(), static_cast<uint16_t>(0x0000));
  EXPECT_EQ(negzero.get_mantissa(), static_cast<uint64_t>(0x0000000000000000));
  EXPECT_EQ(negzero.uintval(), static_cast<uint64_t>(0x8000000000000000));
  EXPECT_STREQ(LIBC_NAMESPACE::str(negzero).c_str(),
               "0x8000000000000000 = (S: 1, E: 0x0000, M: 0x0000000000000000)");

  FPBits<double> one(1.0);
  EXPECT_EQ(one.get_sign(), false);
  EXPECT_EQ(one.get_unbiased_exponent(), static_cast<uint16_t>(0x03FF));
  EXPECT_EQ(one.get_mantissa(), static_cast<uint64_t>(0x0000000000000000));
  EXPECT_EQ(one.uintval(), static_cast<uint64_t>(0x3FF0000000000000));
  EXPECT_STREQ(LIBC_NAMESPACE::str(one).c_str(),
               "0x3FF0000000000000 = (S: 0, E: 0x03FF, M: 0x0000000000000000)");

  FPBits<double> negone(-1.0);
  EXPECT_EQ(negone.get_sign(), true);
  EXPECT_EQ(negone.get_unbiased_exponent(), static_cast<uint16_t>(0x03FF));
  EXPECT_EQ(negone.get_mantissa(), static_cast<uint64_t>(0x0000000000000000));
  EXPECT_EQ(negone.uintval(), static_cast<uint64_t>(0xBFF0000000000000));
  EXPECT_STREQ(LIBC_NAMESPACE::str(negone).c_str(),
               "0xBFF0000000000000 = (S: 1, E: 0x03FF, M: 0x0000000000000000)");

  FPBits<double> num(1.125);
  EXPECT_EQ(num.get_sign(), false);
  EXPECT_EQ(num.get_unbiased_exponent(), static_cast<uint16_t>(0x03FF));
  EXPECT_EQ(num.get_mantissa(), static_cast<uint64_t>(0x0002000000000000));
  EXPECT_EQ(num.uintval(), static_cast<uint64_t>(0x3FF2000000000000));
  EXPECT_STREQ(LIBC_NAMESPACE::str(num).c_str(),
               "0x3FF2000000000000 = (S: 0, E: 0x03FF, M: 0x0002000000000000)");

  FPBits<double> negnum(-1.125);
  EXPECT_EQ(negnum.get_sign(), true);
  EXPECT_EQ(negnum.get_unbiased_exponent(), static_cast<uint16_t>(0x03FF));
  EXPECT_EQ(negnum.get_mantissa(), static_cast<uint64_t>(0x0002000000000000));
  EXPECT_EQ(negnum.uintval(), static_cast<uint64_t>(0xBFF2000000000000));
  EXPECT_STREQ(LIBC_NAMESPACE::str(negnum).c_str(),
               "0xBFF2000000000000 = (S: 1, E: 0x03FF, M: 0x0002000000000000)");
}

#ifdef LIBC_TARGET_ARCH_IS_X86
TEST(LlvmLibcFPBitsTest, X86LongDoubleType) {
  if constexpr (sizeof(long double) == sizeof(double))
    return; // The tests for the "double" type cover for this case.

  EXPECT_STREQ(LIBC_NAMESPACE::str(FPBits<long double>::inf()).c_str(),
               "(+Infinity)");
  EXPECT_STREQ(LIBC_NAMESPACE::str(FPBits<long double>::neg_inf()).c_str(),
               "(-Infinity)");
  EXPECT_STREQ(LIBC_NAMESPACE::str(
                   FPBits<long double>(FPBits<long double>::build_nan(1)))
                   .c_str(),
               "(NaN)");

  FPBits<long double> zero(0.0l);
  EXPECT_EQ(zero.get_sign(), false);
  EXPECT_EQ(zero.get_unbiased_exponent(), static_cast<uint16_t>(0x0000));
  EXPECT_EQ(zero.get_mantissa(), static_cast<UInt128>(0x0000000000000000)
                                     << 64);
  EXPECT_EQ(zero.uintval(), static_cast<UInt128>(0x0000000000000000) << 64);
  EXPECT_STREQ(
      LIBC_NAMESPACE::str(zero).c_str(),
      "0x00000000000000000000000000000000 = "
      "(S: 0, E: 0x0000, I: 0, M: 0x00000000000000000000000000000000)");

  FPBits<long double> negzero(-0.0l);
  EXPECT_EQ(negzero.get_sign(), true);
  EXPECT_EQ(negzero.get_unbiased_exponent(), static_cast<uint16_t>(0x0000));
  EXPECT_EQ(negzero.get_mantissa(), static_cast<UInt128>(0x0000000000000000)
                                        << 64);
  EXPECT_EQ(negzero.uintval(), static_cast<UInt128>(0x1) << 79);
  EXPECT_STREQ(
      LIBC_NAMESPACE::str(negzero).c_str(),
      "0x00000000000080000000000000000000 = "
      "(S: 1, E: 0x0000, I: 0, M: 0x00000000000000000000000000000000)");

  FPBits<long double> one(1.0l);
  EXPECT_EQ(one.get_sign(), false);
  EXPECT_EQ(one.get_unbiased_exponent(), static_cast<uint16_t>(0x3FFF));
  EXPECT_EQ(one.get_mantissa(), static_cast<UInt128>(0x0000000000000000) << 64);
  EXPECT_EQ(one.uintval(), static_cast<UInt128>(0x3FFF8) << 60);
  EXPECT_STREQ(
      LIBC_NAMESPACE::str(one).c_str(),
      "0x0000000000003FFF8000000000000000 = "
      "(S: 0, E: 0x3FFF, I: 1, M: 0x00000000000000000000000000000000)");

  FPBits<long double> negone(-1.0l);
  EXPECT_EQ(negone.get_sign(), true);
  EXPECT_EQ(negone.get_unbiased_exponent(), static_cast<uint16_t>(0x3FFF));
  EXPECT_EQ(negone.get_mantissa(), static_cast<UInt128>(0x0000000000000000)
                                       << 64);
  EXPECT_EQ(negone.uintval(), static_cast<UInt128>(0xBFFF8) << 60);
  EXPECT_STREQ(
      LIBC_NAMESPACE::str(negone).c_str(),
      "0x000000000000BFFF8000000000000000 = "
      "(S: 1, E: 0x3FFF, I: 1, M: 0x00000000000000000000000000000000)");

  FPBits<long double> num(1.125l);
  EXPECT_EQ(num.get_sign(), false);
  EXPECT_EQ(num.get_unbiased_exponent(), static_cast<uint16_t>(0x3FFF));
  EXPECT_EQ(num.get_mantissa(), static_cast<UInt128>(0x1) << 60);
  EXPECT_EQ(num.uintval(), static_cast<UInt128>(0x3FFF9) << 60);
  EXPECT_STREQ(
      LIBC_NAMESPACE::str(num).c_str(),
      "0x0000000000003FFF9000000000000000 = "
      "(S: 0, E: 0x3FFF, I: 1, M: 0x00000000000000001000000000000000)");

  FPBits<long double> negnum(-1.125l);
  EXPECT_EQ(negnum.get_sign(), true);
  EXPECT_EQ(negnum.get_unbiased_exponent(), static_cast<uint16_t>(0x3FFF));
  EXPECT_EQ(negnum.get_mantissa(), static_cast<UInt128>(0x1) << 60);
  EXPECT_EQ(negnum.uintval(), static_cast<UInt128>(0xBFFF9) << 60);
  EXPECT_STREQ(
      LIBC_NAMESPACE::str(negnum).c_str(),
      "0x000000000000BFFF9000000000000000 = "
      "(S: 1, E: 0x3FFF, I: 1, M: 0x00000000000000001000000000000000)");
}
#else
TEST(LlvmLibcFPBitsTest, LongDoubleType) {
#if defined(LONG_DOUBLE_IS_DOUBLE)
  return; // The tests for the "double" type cover for this case.
#else
  EXPECT_STREQ(LIBC_NAMESPACE::str(FPBits<long double>::inf()).c_str(),
               "(+Infinity)");
  EXPECT_STREQ(LIBC_NAMESPACE::str(FPBits<long double>::neg_inf()).c_str(),
               "(-Infinity)");
  EXPECT_STREQ(LIBC_NAMESPACE::str(
                   FPBits<long double>(FPBits<long double>::build_nan(1)))
                   .c_str(),
               "(NaN)");

  FPBits<long double> zero(0.0l);
  EXPECT_EQ(zero.get_sign(), false);
  EXPECT_EQ(zero.get_unbiased_exponent(), static_cast<uint16_t>(0x0000));
  EXPECT_EQ(zero.get_mantissa(), static_cast<UInt128>(0x0000000000000000)
                                     << 64);
  EXPECT_EQ(zero.uintval(), static_cast<UInt128>(0x0000000000000000) << 64);
  EXPECT_STREQ(LIBC_NAMESPACE::str(zero).c_str(),
               "0x00000000000000000000000000000000 = "
               "(S: 0, E: 0x0000, M: 0x00000000000000000000000000000000)");

  FPBits<long double> negzero(-0.0l);
  EXPECT_EQ(negzero.get_sign(), true);
  EXPECT_EQ(negzero.get_unbiased_exponent(), static_cast<uint16_t>(0x0000));
  EXPECT_EQ(negzero.get_mantissa(), static_cast<UInt128>(0x0000000000000000)
                                        << 64);
  EXPECT_EQ(negzero.uintval(), static_cast<UInt128>(0x1) << 127);
  EXPECT_STREQ(LIBC_NAMESPACE::str(negzero).c_str(),
               "0x80000000000000000000000000000000 = "
               "(S: 1, E: 0x0000, M: 0x00000000000000000000000000000000)");

  FPBits<long double> one(1.0l);
  EXPECT_EQ(one.get_sign(), false);
  EXPECT_EQ(one.get_unbiased_exponent(), static_cast<uint16_t>(0x3FFF));
  EXPECT_EQ(one.get_mantissa(), static_cast<UInt128>(0x0000000000000000) << 64);
  EXPECT_EQ(one.uintval(), static_cast<UInt128>(0x3FFF) << 112);
  EXPECT_STREQ(LIBC_NAMESPACE::str(one).c_str(),
               "0x3FFF0000000000000000000000000000 = "
               "(S: 0, E: 0x3FFF, M: 0x00000000000000000000000000000000)");

  FPBits<long double> negone(-1.0l);
  EXPECT_EQ(negone.get_sign(), true);
  EXPECT_EQ(negone.get_unbiased_exponent(), static_cast<uint16_t>(0x3FFF));
  EXPECT_EQ(negone.get_mantissa(), static_cast<UInt128>(0x0000000000000000)
                                       << 64);
  EXPECT_EQ(negone.uintval(), static_cast<UInt128>(0xBFFF) << 112);
  EXPECT_STREQ(LIBC_NAMESPACE::str(negone).c_str(),
               "0xBFFF0000000000000000000000000000 = "
               "(S: 1, E: 0x3FFF, M: 0x00000000000000000000000000000000)");

  FPBits<long double> num(1.125l);
  EXPECT_EQ(num.get_sign(), false);
  EXPECT_EQ(num.get_unbiased_exponent(), static_cast<uint16_t>(0x3FFF));
  EXPECT_EQ(num.get_mantissa(), static_cast<UInt128>(0x2) << 108);
  EXPECT_EQ(num.uintval(), static_cast<UInt128>(0x3FFF2) << 108);
  EXPECT_STREQ(LIBC_NAMESPACE::str(num).c_str(),
               "0x3FFF2000000000000000000000000000 = "
               "(S: 0, E: 0x3FFF, M: 0x00002000000000000000000000000000)");

  FPBits<long double> negnum(-1.125l);
  EXPECT_EQ(negnum.get_sign(), true);
  EXPECT_EQ(negnum.get_unbiased_exponent(), static_cast<uint16_t>(0x3FFF));
  EXPECT_EQ(negnum.get_mantissa(), static_cast<UInt128>(0x2) << 108);
  EXPECT_EQ(negnum.uintval(), static_cast<UInt128>(0xBFFF2) << 108);
  EXPECT_STREQ(LIBC_NAMESPACE::str(negnum).c_str(),
               "0xBFFF2000000000000000000000000000 = "
               "(S: 1, E: 0x3FFF, M: 0x00002000000000000000000000000000)");
#endif
}
#endif
