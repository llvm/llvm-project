//===-- Unittests for atan2f128 -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/FPUtil/FPBits.h"
#include "src/__support/integer_literals.h"
#include "src/math/atan2f128.h"
#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"

using LIBC_NAMESPACE::fputil::Float128;
using LIBC_NAMESPACE::operator""_u128;

// The public atan2f128 returns the native float128; reinterpret its bits
// directly so the comparison never goes through a value conversion.
using NativeFPBits = LIBC_NAMESPACE::fputil::FPBits<float128>;

using LlvmLibcAtan2f128Test = LIBC_NAMESPACE::testing::FPTest<Float128>;

TEST_F(LlvmLibcAtan2f128Test, SpecialNumbers) {
  EXPECT_FP_EQ_ALL_ROUNDING(aNaN,
                            Float128(LIBC_NAMESPACE::atan2f128(aNaN, zero)));
  EXPECT_FP_EQ_ALL_ROUNDING(
      aNaN, Float128(LIBC_NAMESPACE::atan2f128(Float128(1.0), aNaN)));
  EXPECT_FP_EQ_ALL_ROUNDING(zero,
                            Float128(LIBC_NAMESPACE::atan2f128(zero, zero)));
  EXPECT_FP_EQ_ALL_ROUNDING(
      neg_zero, Float128(LIBC_NAMESPACE::atan2f128(neg_zero, zero)));
  EXPECT_FP_EQ_ALL_ROUNDING(
      zero, Float128(LIBC_NAMESPACE::atan2f128(Float128(1.0), inf)));
  EXPECT_FP_EQ_ALL_ROUNDING(
      neg_zero, Float128(LIBC_NAMESPACE::atan2f128(Float128(-1.0), inf)));

  Float128 x, y, r, actual;

  // 0x1.ffffffffffffffffffffffffffe7p1
  x.bits = 0x4000FFFFFFFFFFFFFFFFFFFFFFFFFFE7_u128;
  // 0x1.fffffffffffffffffffffffffff2p1
  y.bits = 0x4000FFFFFFFFFFFFFFFFFFFFFFFFFFF2_u128;
  // 0x1.921fb54442d18469898cc51701b3p-1
  r.bits = 0x3FFE921FB54442D18469898CC51701B3_u128;
  actual.bits = NativeFPBits(LIBC_NAMESPACE::atan2f128(x, y)).uintval();
  EXPECT_FP_EQ(r, actual);

  // -0x1.f122e07fff556143p+3524
  x.bits = 0xCDC3F122E07FFF556143000000000000_u128;
  // 0x1.f122e07fff55615b75p+6316
  y.bits = 0x58ABF122E07FFF55615B750000000000_u128;
  // -0x1.ffffffffffffffe6cfcdc604fc99p-2793
  r.bits = 0xB516FFFFFFFFFFFFFFE6CFCDC604FC99_u128;
  actual.bits = NativeFPBits(LIBC_NAMESPACE::atan2f128(x, y)).uintval();
  EXPECT_FP_EQ(r, actual);
}
