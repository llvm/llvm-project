//===-- Unittests for stdbit ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDSList-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "test/UnitTest/Test.h"

/*
 * The intent of this test is validate that the generic math macros operate as
 * intended
 */
#include "include/llvm-libc-macros/generic-math-macros.h"


// INF can be defined as a number with zeroed out mantissa and 1s in the 
// exponent
static uint32_t positive_infinity = 0x7F800000;
static uint32_t negative_infinity = 0xFF800000;
static const float pos_inf = *(float *) &positive_infinity;
static const float neg_inf = *(float *) &negative_infinity;

// NaN can be defined as a number with all 1s in the exponent
static uint32_t positive_nan = 0x7F800001;
static uint32_t negative_nan = 0xFF800001;
static const float pos_nan = *(float *) &positive_nan;
static const float neg_nan = *(float *) &negative_nan;

/*
 * As with IEEE 754-1985, the biased-exponent field is filled with all 1 bits 
 * to indicate either infinity (trailing significand field = 0) or a NaN 
 * (trailing significand field ≠ 0)
 */

TEST(LlvmLibcGenericMath, TypeGenericMacroMathIsfinite) {
  EXPECT_EQ(isfinite(3.14), 1);
  EXPECT_EQ(isfinite(3.14 / 0.0), 0);
  EXPECT_EQ(isfinite(pos_inf), 0);
  EXPECT_EQ(isfinite(neg_inf), 0);
  EXPECT_EQ(isfinite(pos_nan), 0);
  EXPECT_EQ(isfinite(neg_nan), 0);
}

TEST(LlvmLibcGenericMath, TypeGenericMacroMathIsinf) {
  EXPECT_EQ(isinf(3.14), 0);
  EXPECT_EQ(isinf(pos_inf), 1);
  EXPECT_EQ(isinf(neg_inf), 1);
  EXPECT_EQ(isnan(0.0 * pos_inf), 1);   // multiply 0 by infinity
  EXPECT_EQ(isinf(3.14 / 0.0), 1);      // division by 0
  EXPECT_EQ(isnan(3.14 / pos_inf), 0);  // division by infinity
}

TEST(LlvmLibcGenericMath, TypeGenericMacroMathIsnan) {
  EXPECT_EQ(isnan(3.14), 0);
  EXPECT_EQ(isnan(pos_nan), 1);
  EXPECT_EQ(isnan(neg_nan), 1);
  EXPECT_EQ(isnan(0.0 * pos_inf), 1);     // multiply 0 by infinity
  EXPECT_EQ(isnan(3.14 / 0.0), 0);        // division by 0
  EXPECT_EQ(isnan(3.14 / pos_inf), 0);    // division by infinity
  EXPECT_EQ(isnan(pos_inf / neg_inf), 1); // pos infinity / neg infinity
}

/*
TEST(LlvmLibcGenericMath, TypeGenericMacroMathSignbit) {
  // float

  // double

}*/
