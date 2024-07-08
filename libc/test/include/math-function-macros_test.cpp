//===-- Unittests for generic-math-macros ---------------------------------===//
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
#include "include/llvm-libc-macros/math-function-macros.h"

// INF can be defined as a number with zeroed out mantissa and 1s in the 
// exponent
static uint32_t pos_inf_bits = 0x7F800000;
static uint32_t neg_inf_bits = 0xFF800000;
static const float pos_inf = *(float *)&pos_inf_bits;
static const float neg_inf = *(float *)&neg_inf_bits;

// NaN can be defined as a number with all 1s in the exponent
static uint32_t pos_nan_bits = 0x7F800001;
static uint32_t neg_nan_bits = 0xFF800001;
static const float pos_nan = *(float *)&pos_nan_bits;
static const float neg_nan = *(float *)&neg_nan_bits;

#define PI 3.14159265358979323846
#define CASE_DIV_BY_ZERO            PI / 0.0
#define CASE_DIV_BY_POS_INF         PI / pos_inf
#define CASE_DIV_BY_NEG_INF         PI / neg_inf
#define CASE_MULT_ZERO_BY_POS_INF   0 * pos_inf
#define CASE_MULT_ZERO_BY_NEG_INF   0 * neg_inf

/*
 * As with IEEE 754-1985, the biased-exponent field is filled with all 1 bits 
 * to indicate either infinity (trailing significand field = 0) or a NaN 
 * (trailing significand field ≠ 0)
 */

/*
TEST(LlvmLibcMathFunctionMacros, TypeMathFunctionMacroIsfinite) {
  EXPECT_EQ(isfinite(pos_inf), 0);
  EXPECT_EQ(isfinite(neg_inf), 0);
  EXPECT_EQ(isfinite(pos_nan), 0);
  EXPECT_EQ(isfinite(neg_nan), 0);
  EXPECT_EQ(isfinite(CASE_DIV_BY_ZERO), 0);
  EXPECT_EQ(isfinite(PI), 1);
}

TEST(LlvmLibcMathFunctionMacros, TypeMathFunctionMacroIsinf) {
  EXPECT_EQ(isinf(PI), 0);
  EXPECT_EQ(isinf(CASE_DIV_BY_POS_INF), 0);
  EXPECT_EQ(isinf(CASE_DIV_BY_NEG_INF), 0);
  EXPECT_EQ(isinf(CASE_MULT_ZERO_BY_POS_INF), 0);
  EXPECT_EQ(isinf(CASE_MULT_ZERO_BY_NEG_INF), 0);
  EXPECT_EQ(isinf(pos_inf), 1);
  EXPECT_EQ(isinf(neg_inf), 1);
  EXPECT_EQ(isinf(CASE_DIV_BY_ZERO), 1);
}

TEST(LlvmLibcMathFunctionMacros, TypeMathFunctionMacroIsnan) {
  EXPECT_EQ(isnan(static_cast<float>(PI)), 0); 
  EXPECT_EQ(isnan(static_cast<double>(PI)), 0); 
  EXPECT_EQ(isnan(static_cast<long double>(PI)), 0); 

  EXPECT_EQ(isnan(CASE_DIV_BY_ZERO), 0);
  EXPECT_EQ(isnan(CASE_DIV_BY_POS_INF), 0);
  EXPECT_EQ(isnan(CASE_DIV_BY_NEG_INF), 0);
  
  EXPECT_EQ(isnan(pos_nan), 1);
  EXPECT_EQ(isnan(neg_nan), 1);
  EXPECT_EQ(isnan(CASE_MULT_ZERO_BY_POS_INF), 1);
  EXPECT_EQ(isnan(CASE_MULT_ZERO_BY_NEG_INF), 1);
  EXPECT_EQ(isnan(pos_inf / neg_inf), 1);
}

TEST(LlvmLibcMathFunctionMacros, TypeMathFunctionMacroSignbit) {
  EXPECT_EQ(signbit(static_cast<float>(PI)), 0);
  EXPECT_EQ(signbit(static_cast<double>(PI)), 0);
  EXPECT_EQ(signbit(static_cast<long double>(PI)), 0);
  EXPECT_EQ(signbit(pos_inf), 0);
  EXPECT_EQ(signbit(pos_nan), 0);

  EXPECT_EQ(signbit(static_cast<float>(-PI)), 1);
  EXPECT_EQ(signbit(static_cast<double>(-PI)), 1);
  EXPECT_EQ(signbit(static_cast<long double>(-PI)), 1);
  EXPECT_EQ(signbit(neg_inf), 1);
  EXPECT_EQ(signbit(neg_nan), 1);
}
*/

LIST_MATH_FUNCTION_MACROS_TEST


