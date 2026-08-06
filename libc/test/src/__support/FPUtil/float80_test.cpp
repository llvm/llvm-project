//===-- Unittests for Float80 emulated type ------------------------------===//
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

TEST(LlvmLibcFloat80Test, temp) { Float80 a(1.0f); }

TEST(LlvmLibcFloat80Test, IntegerConversion) {
  // Float80 to Integer conversion test
  ASSERT_EQ(static_cast<int>(Float80(0.0f)), 0);
  ASSERT_EQ(static_cast<int>(Float80(1.0f)), 1);
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
