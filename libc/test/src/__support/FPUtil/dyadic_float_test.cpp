//===-- Unittests for the DyadicFloat class -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/FPUtil/dyadic_float.h"
#include "src/__support/UInt.h"
#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"
#include "utils/MPFRWrapper/MPFRUtils.h"

using Float128 = LIBC_NAMESPACE::fputil::DyadicFloat<128>;
using Float192 = LIBC_NAMESPACE::fputil::DyadicFloat<192>;
using Float256 = LIBC_NAMESPACE::fputil::DyadicFloat<256>;

TEST(LlvmLibcDyadicFloatTest, BasicConversions) {
  Float128 x(/*sign*/ false, /*exponent*/ 0,
             /*mantissa*/ Float128::MantissaType(1));
  volatile float xf = float(x);
  volatile double xd = double(x);
  ASSERT_FP_EQ(1.0f, xf);
  ASSERT_FP_EQ(1.0, xd);

  Float128 y(0x1.0p-53);
  volatile float yf = float(y);
  volatile double yd = double(y);
  ASSERT_FP_EQ(0x1.0p-53f, yf);
  ASSERT_FP_EQ(0x1.0p-53, yd);

  Float128 z = quick_add(x, y);

  EXPECT_FP_EQ_ALL_ROUNDING(xf + yf, float(z));
  EXPECT_FP_EQ_ALL_ROUNDING(xd + yd, double(z));
}

TEST(LlvmLibcDyadicFloatTest, QuickAdd) {
  Float192 x(/*sign*/ false, /*exponent*/ 0,
             /*mantissa*/ Float192::MantissaType(0x123456));
  volatile double xd = double(x);
  ASSERT_FP_EQ(0x1.23456p20, xd);

  Float192 y(0x1.abcdefp-20);
  volatile double yd = double(y);
  ASSERT_FP_EQ(0x1.abcdefp-20, yd);

  Float192 z = quick_add(x, y);

  EXPECT_FP_EQ_ALL_ROUNDING(xd + yd, (volatile double)(z));
}

TEST(LlvmLibcDyadicFloatTest, QuickMul) {
  Float256 x(/*sign*/ false, /*exponent*/ 0,
             /*mantissa*/ Float256::MantissaType(0x123456));
  volatile double xd = double(x);
  ASSERT_FP_EQ(0x1.23456p20, xd);

  Float256 y(0x1.abcdefp-25);
  volatile double yd = double(y);
  ASSERT_FP_EQ(0x1.abcdefp-25, yd);

  Float256 z = quick_mul(x, y);

  EXPECT_FP_EQ_ALL_ROUNDING(xd * yd, double(z));
}
