//===-- Unittests for Float80 emulated type -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "hdr/limits_macros.h"
#include "src/__support/FPUtil/FEnvImpl.h"
#include "src/__support/FPUtil/float80.h"
#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"

using LIBC_NAMESPACE::Sign;
using LIBC_NAMESPACE::fputil::Float80;
using FPBits = LIBC_NAMESPACE::fputil::FPBits<Float80>;

TEST(LlvmLibcFloat80Test, Operators) {
  Float80 a(1.0f), b(1.0f), c(2.0f), d(3.0f), pa(1.0f), na(-1.0f);

  // comparison operators
  ASSERT_TRUE(a == b);
  ASSERT_TRUE(a == Float80(1.0));
  ASSERT_TRUE(a != c);
  ASSERT_TRUE(b != c);
  ASSERT_TRUE(c > b);
  ASSERT_TRUE(a >= b);
  ASSERT_TRUE(b <= c);
  ASSERT_TRUE(a < c);

  // Unary operators
  ASSERT_TRUE(-pa == na);
  ASSERT_TRUE(-(-pa) == pa);

  // Binary operators
  ASSERT_TRUE((a + b) == c);
  ASSERT_TRUE((a - b) == Float80(0.0f));
  ASSERT_TRUE((c * d) == Float80(6.0f));
  ASSERT_TRUE((Float80(6.0f) / d) == Float80(2.0f));
}

TEST(LlvmLibcFloat80Test, SpecialValues) {
  Float80 inf = FPBits::inf(Sign::POS).get_val();
  Float80 neg_inf = FPBits::inf(Sign::NEG).get_val();
  Float80 nan = FPBits::quiet_nan().get_val();

  // checking operators with special values
  ASSERT_TRUE(Float80(0.0f) == Float80(-0.0f)); // +0.0 == -0.0 is true
  ASSERT_TRUE(Float80(0.0f) == Float80(0.0f));
  ASSERT_TRUE(inf == inf);
  ASSERT_TRUE(-inf == neg_inf);
  ASSERT_TRUE((inf + Float80(1.0f)) == inf);
  ASSERT_TRUE(inf + inf == inf);
  ASSERT_TRUE(nan != nan);
  ASSERT_TRUE(!(nan == nan));
  ASSERT_TRUE(nan != Float80(0.0f));
}

TEST(LlvmLibcFloat80Test, IntegerConversion) {
  // Float80 to Integer conversion test
  ASSERT_EQ(static_cast<int>(Float80(0.0f)), 0);
  ASSERT_EQ(static_cast<int>(Float80(-0.0f)), 0);
  ASSERT_EQ(static_cast<int>(Float80(1.0f)), 1);
  ASSERT_EQ(static_cast<int>(Float80(-1.0f)), -1);
  ASSERT_EQ(static_cast<long long>(Float80(1000000000.0)),
            static_cast<long long>(1000000000));
  ASSERT_EQ(static_cast<unsigned>(Float80(7.0f)), 7U);
  ASSERT_EQ(static_cast<int>(Float80(1.9f)), 1);

  // Border values
  LIBC_NAMESPACE::fputil::clear_except(FE_ALL_EXCEPT);
  ASSERT_EQ(static_cast<int>(Float80(INT_MAX)), INT_MAX);
  ASSERT_EQ(static_cast<long long>(Float80(LLONG_MAX)), LLONG_MAX);
  ASSERT_EQ(static_cast<unsigned>(Float80(UINT_MAX)), UINT_MAX);
  EXPECT_EQ(LIBC_NAMESPACE::fputil::test_except(FE_INVALID), 0);

  // FP exceptions
  LIBC_NAMESPACE::fputil::clear_except(FE_ALL_EXCEPT);
  ASSERT_EQ(static_cast<int>(FPBits::quiet_nan().get_val()), INT_MAX);
  EXPECT_FP_EXCEPTION(FE_INVALID);

  LIBC_NAMESPACE::fputil::clear_except(FE_ALL_EXCEPT);
  ASSERT_EQ(static_cast<int>(FPBits::inf().get_val()), INT_MAX);
  EXPECT_FP_EXCEPTION(FE_INVALID);

  // Extreme values
  LIBC_NAMESPACE::fputil::clear_except(FE_ALL_EXCEPT);
  ASSERT_EQ(static_cast<int>(Float80(1e300)), INT_MAX);
  EXPECT_FP_EXCEPTION(FE_INVALID);

  LIBC_NAMESPACE::fputil::clear_except(FE_ALL_EXCEPT);
  ASSERT_EQ(static_cast<int>(FPBits::inf(Sign::POS).get_val()), INT_MAX);
  EXPECT_FP_EXCEPTION(FE_INVALID);

  // Small values
  LIBC_NAMESPACE::fputil::clear_except(FE_ALL_EXCEPT);
  ASSERT_EQ(static_cast<int>(Float80(1e-300)), 0);
  ASSERT_EQ(static_cast<int>(Float80(0.5)), 0);
  EXPECT_EQ(LIBC_NAMESPACE::fputil::test_except(FE_INVALID), 0);
}

#ifdef LIBC_TYPES_LONG_DOUBLE_IS_X86_FLOAT80
TEST(LlvmLibcFloat80Test, randomTest) {
  using FPBitsL = LIBC_NAMESPACE::fputil::FPBits<long double>;

  const FPBitsL::StorageType EDGE_CASES[] = {
      FPBitsL::zero(Sign::POS).uintval(),
      FPBitsL::zero(Sign::NEG).uintval(),
      FPBitsL::inf(Sign::POS).uintval(),
      FPBitsL::inf(Sign::NEG).uintval(),
      FPBitsL::quiet_nan().uintval(),
      FPBitsL::min_subnormal(Sign::POS).uintval(),
      FPBitsL::min_subnormal(Sign::NEG).uintval(),
      FPBitsL::max_subnormal(Sign::POS).uintval(),
      FPBitsL::max_subnormal(Sign::NEG).uintval(),
      FPBitsL::min_normal(Sign::POS).uintval(),
      FPBitsL::min_normal(Sign::NEG).uintval(),
      FPBitsL::max_normal(Sign::POS).uintval(),
      FPBitsL::max_normal(Sign::NEG).uintval(),
      FPBitsL::one(Sign::POS).uintval(),
      FPBitsL::one(Sign::NEG).uintval(),
  };

  for (FPBitsL::StorageType bits : EDGE_CASES) {
    long double native = FPBitsL(bits).get_val();

    Float80 f80_temp = LIBC_NAMESPACE::fputil::cast<Float80>(native);
    EXPECT_EQ(FPBits(f80_temp).uintval(), bits);

    long double ld_temp =
        LIBC_NAMESPACE::fputil::cast<long double>(FPBits(bits).get_val());
    EXPECT_EQ(FPBitsL(ld_temp).uintval(), bits);
  }
}

#endif // LIBC_TYPES_LONG_DOUBLE_IS_X86_FLOAT80

TEST(LlvmLibcFloat80Test, FromIntegralTypes) {
  // Integer to Float80 conversion test
  ASSERT_EQ(FPBits(Float80(42)).uintval(), FPBits(Float80(42.0f)).uintval());
  ASSERT_EQ(FPBits(Float80(0)).uintval(), FPBits(Float80(0.0f)).uintval());
  ASSERT_EQ(FPBits(Float80(7U)).uintval(), FPBits(Float80(7.0f)).uintval());
  ASSERT_EQ(FPBits(Float80(123456789LL)).uintval(),
            FPBits(Float80(123456789.0)).uintval());

  // 2147483648.0 or 2^31 is out of bound in signed and not in unsigned
  LIBC_NAMESPACE::fputil::clear_except(FE_ALL_EXCEPT);
  ASSERT_EQ(static_cast<int>(Float80(2147483648.0)), INT_MAX);
  EXPECT_FP_EXCEPTION(FE_INVALID);

  LIBC_NAMESPACE::fputil::clear_except(FE_ALL_EXCEPT);
  ASSERT_EQ(static_cast<unsigned>(Float80(2147483648.0)), 2147483648U);
  EXPECT_EQ(LIBC_NAMESPACE::fputil::test_except(FE_INVALID), 0);
}
