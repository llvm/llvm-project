//===-- Unittests for Float128 emulated type ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "hdr/limits_macros.h"
#include "src/__support/FPUtil/FEnvImpl.h"
#include "src/__support/FPUtil/float128.h"
#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"

using LIBC_NAMESPACE::Sign;
using LIBC_NAMESPACE::fputil::Float128;
using FPBits = LIBC_NAMESPACE::fputil::FPBits<Float128>;

TEST(LlvmLibcFloat128Test, Operators) {
  Float128 a(1.0f), b(1.0f), c(2.0f), d(3.0f), pa(1.0f), na(-1.0f);

  // comparison operators
  ASSERT_TRUE(a == b);
  ASSERT_TRUE(a == Float128(1.0));
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
  ASSERT_TRUE((a - b) == Float128(0.0f));
  ASSERT_TRUE((c * d) == Float128(6.0f));
  ASSERT_TRUE((Float128(6.0f) / d) == Float128(2.0f));

  // Compound assignment operators
  a += Float128(1.0f);
  ASSERT_TRUE(a == c);
  b -= Float128(1.0f);
  ASSERT_TRUE(b == Float128(0.0f));
  c *= Float128(2.0f);
  ASSERT_TRUE(c == Float128(4.0f));
  d /= Float128(3.0f);
  ASSERT_TRUE(d == Float128(1.0f));
}

TEST(LlvmLibcFloat128Test, SpecialValues) {
  Float128 inf = FPBits::inf(Sign::POS).get_val();
  Float128 neg_inf = FPBits::inf(Sign::NEG).get_val();
  Float128 nan = FPBits::quiet_nan().get_val();

  // checking operators with special values
  ASSERT_TRUE(Float128(0.0f) == Float128(-0.0f)); // +0.0 == -0.0 is true
  ASSERT_TRUE(Float128(0.0f) == Float128(0.0f));
  ASSERT_TRUE(inf == inf);
  ASSERT_TRUE(-inf == neg_inf);
  ASSERT_TRUE((inf + Float128(1.0f)) == inf);
  ASSERT_TRUE(inf + inf == inf);
  ASSERT_TRUE(nan != nan);
  ASSERT_TRUE(!(nan == nan));
  ASSERT_TRUE(nan != Float128(0.0f));
}

TEST(LlvmLibcFloat128Test, IntegerConversion) {
  // Float128 to Integer conversion test
  ASSERT_EQ(static_cast<int>(Float128(0.0f)), 0);
  ASSERT_EQ(static_cast<int>(Float128(-0.0f)), 0);
  ASSERT_EQ(static_cast<int>(Float128(1.0f)), 1);
  ASSERT_EQ(static_cast<int>(Float128(-1.0)), -1);
  ASSERT_EQ(static_cast<long long>(Float128(1000000000.0)),
            static_cast<long long>(1000000000));
  ASSERT_EQ(static_cast<unsigned>(Float128(7.0f)), 7U);
  ASSERT_EQ(static_cast<int>(Float128(-1.5)), -1);
  ASSERT_EQ(static_cast<int>(Float128(-1.9)), -1);
  ASSERT_EQ(static_cast<int>(Float128(1.9f)), 1);

  // Extreme values
  ASSERT_EQ(static_cast<int>(Float128(INT_MAX)), INT_MAX);
  ASSERT_EQ(static_cast<int>(Float128(INT_MIN)), INT_MIN);
  ASSERT_EQ(static_cast<long long>(Float128(LLONG_MAX)), LLONG_MAX);
  ASSERT_EQ(static_cast<long long>(Float128(LLONG_MIN)), LLONG_MIN);
  ASSERT_EQ(static_cast<unsigned>(Float128(UINT_MAX)), UINT_MAX);
  ASSERT_EQ(static_cast<unsigned>(Float128(0U)), 0U);

  // FP exceptions
  LIBC_NAMESPACE::fputil::clear_except(FE_ALL_EXCEPT);
  ASSERT_EQ(static_cast<int>(FPBits::quiet_nan().get_val()), 0);
  EXPECT_FP_EXCEPTION(FE_INVALID);

  LIBC_NAMESPACE::fputil::clear_except(FE_ALL_EXCEPT);
  ASSERT_EQ(static_cast<int>(FPBits::inf().get_val()), 0);
  EXPECT_FP_EXCEPTION(FE_INVALID);
}

TEST(LlvmLibcFloat128Test, FromIntegralTypes) {
  // Integer to float128 conversion test
  ASSERT_TRUE(Float128(42) == Float128(42.0f));
  ASSERT_TRUE(Float128(-42) == Float128(-42.0f));
  ASSERT_TRUE(Float128(0) == Float128(0.0f));
  ASSERT_TRUE(Float128(7U) == Float128(7.0f));
  ASSERT_TRUE(Float128(-7LL) == Float128(-7.0));
  ASSERT_TRUE(Float128(123456789LL) == Float128(123456789.0));
}
