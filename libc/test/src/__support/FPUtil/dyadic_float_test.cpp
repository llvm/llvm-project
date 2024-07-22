//===-- Unittests for the DyadicFloat class -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/FPUtil/dyadic_float.h"
#include "src/__support/big_int.h"
#include "src/__support/macros/properties/types.h"
#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"
#include "utils/MPFRWrapper/MPFRUtils.h"

using Float128 = LIBC_NAMESPACE::fputil::DyadicFloat<128>;
using Float192 = LIBC_NAMESPACE::fputil::DyadicFloat<192>;
using Float256 = LIBC_NAMESPACE::fputil::DyadicFloat<256>;

TEST(LlvmLibcDyadicFloatTest, BasicConversions) {
  Float128 x(Sign::POS, /*exponent*/ 0,
             /*mantissa*/ Float128::MantissaType(1));
  ASSERT_FP_EQ(1.0f, float(x));
  ASSERT_FP_EQ(1.0, double(x));

  Float128 y(0x1.0p-53);
  ASSERT_FP_EQ(0x1.0p-53f, float(y));
  ASSERT_FP_EQ(0x1.0p-53, double(y));

  Float128 z = quick_add(x, y);

  EXPECT_FP_EQ_ALL_ROUNDING(float(x) + float(y), float(z));
  EXPECT_FP_EQ_ALL_ROUNDING(double(x) + double(y), double(z));
}

TEST(LlvmLibcDyadicFloatTest, QuickAdd) {
  Float192 x(Sign::POS, /*exponent*/ 0,
             /*mantissa*/ Float192::MantissaType(0x123456));
  ASSERT_FP_EQ(0x1.23456p20, double(x));

  Float192 y(0x1.abcdefp-20);
  ASSERT_FP_EQ(0x1.abcdefp-20, double(y));

  Float192 z = quick_add(x, y);
  EXPECT_FP_EQ_ALL_ROUNDING(double(x) + double(y), double(z));
}

TEST(LlvmLibcDyadicFloatTest, QuickMul) {
  Float256 x(Sign::POS, /*exponent*/ 0,
             /*mantissa*/ Float256::MantissaType(0x123456));
  ASSERT_FP_EQ(0x1.23456p20, double(x));

  Float256 y(0x1.abcdefp-25);
  ASSERT_FP_EQ(0x1.abcdefp-25, double(y));

  Float256 z = quick_mul(x, y);
  EXPECT_FP_EQ_ALL_ROUNDING(double(x) * double(y), double(z));
}

#define TEST_EDGE_RANGES(Name, Type)                                           \
  TEST(LlvmLibcDyadicFloatTest, EdgeRanges##Name) {                            \
    using Bits = LIBC_NAMESPACE::fputil::FPBits<Type>;                         \
    using DFType = LIBC_NAMESPACE::fputil::DyadicFloat<Bits::STORAGE_LEN>;     \
    Type max_normal = Bits::max_normal().get_val();                            \
    Type min_normal = Bits::min_normal().get_val();                            \
    Type min_subnormal = Bits::min_subnormal().get_val();                      \
    Type two(2);                                                               \
                                                                               \
    DFType x(min_normal);                                                      \
    EXPECT_FP_EQ_ALL_ROUNDING(min_normal, static_cast<Type>(x));               \
    --x.exponent;                                                              \
    EXPECT_FP_EQ(min_normal / two, static_cast<Type>(x));                      \
                                                                               \
    DFType y(two *min_normal - min_subnormal);                                 \
    --y.exponent;                                                              \
    EXPECT_FP_EQ(min_normal, static_cast<Type>(y));                            \
                                                                               \
    DFType z(min_subnormal);                                                   \
    EXPECT_FP_EQ_ALL_ROUNDING(min_subnormal, static_cast<Type>(z));            \
    --z.exponent;                                                              \
    EXPECT_FP_EQ(Bits::zero().get_val(), static_cast<Type>(z));                \
                                                                               \
    DFType t(max_normal);                                                      \
    EXPECT_FP_EQ_ALL_ROUNDING(max_normal, static_cast<Type>(t));               \
    ++t.exponent;                                                              \
    EXPECT_FP_EQ(Bits::inf().get_val(), static_cast<Type>(t));                 \
  }                                                                            \
  static_assert(true, "Require semicolon.")

TEST_EDGE_RANGES(Float, float);
TEST_EDGE_RANGES(Double, double);
TEST_EDGE_RANGES(LongDouble, long double);
#ifdef LIBC_TYPES_HAS_FLOAT16
TEST_EDGE_RANGES(Float16, float16);
#endif
