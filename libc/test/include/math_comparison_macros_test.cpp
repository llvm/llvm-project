//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unittests for math comparison macros.
///
//===----------------------------------------------------------------------===//

#include "src/__support/macros/properties/types.h"
#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"

#include "include/llvm-libc-macros/math-function-macros.h"

template <typename T>
class MathComparisonMacrosTest : public LIBC_NAMESPACE::testing::Test {
  DECLARE_SPECIAL_CONSTANTS(T)

public:
  void testComparisons() {
    const T neg_two = T(-2);
    const T neg_one = T(-1);
    const T pos_one = T(1);
    const T pos_two = T(2);

    // Strictly ordered values: neg_inf < neg_two < neg_one < -0.0 == +0.0 <
    // min_denormal < min_normal < pos_one < pos_two < max_normal < inf.
    const T ordered[] = {neg_inf,  neg_max_normal, neg_two,    neg_one,
                         neg_zero, min_denormal,   min_normal, pos_one,
                         pos_two,  max_normal,     inf};
    constexpr size_t N = sizeof(ordered) / sizeof(ordered[0]);

    for (size_t i = 0; i < N; ++i) {
      for (size_t j = 0; j < N; ++j) {
        EXPECT_EQ(isgreater(ordered[i], ordered[j]), i > j ? 1 : 0);
        EXPECT_EQ(isgreaterequal(ordered[i], ordered[j]), i >= j ? 1 : 0);
        EXPECT_EQ(isless(ordered[i], ordered[j]), i < j ? 1 : 0);
        EXPECT_EQ(islessequal(ordered[i], ordered[j]), i <= j ? 1 : 0);
        EXPECT_EQ(islessgreater(ordered[i], ordered[j]), i != j ? 1 : 0);
        EXPECT_EQ(isunordered(ordered[i], ordered[j]), 0);
      }
    }

    // +0.0 and -0.0 compare equal and ordered.
    EXPECT_EQ(isgreater(zero, neg_zero), 0);
    EXPECT_EQ(isgreater(neg_zero, zero), 0);
    EXPECT_EQ(isgreaterequal(zero, neg_zero), 1);
    EXPECT_EQ(isgreaterequal(neg_zero, zero), 1);
    EXPECT_EQ(isless(zero, neg_zero), 0);
    EXPECT_EQ(isless(neg_zero, zero), 0);
    EXPECT_EQ(islessequal(zero, neg_zero), 1);
    EXPECT_EQ(islessequal(neg_zero, zero), 1);
    EXPECT_EQ(islessgreater(zero, neg_zero), 0);
    EXPECT_EQ(islessgreater(neg_zero, zero), 0);
    EXPECT_EQ(isunordered(zero, neg_zero), 0);
    EXPECT_EQ(isunordered(neg_zero, zero), 0);

    // NaN comparisons return 0, except isunordered which returns 1.
    const T nans[] = {aNaN, neg_aNaN, sNaN, neg_sNaN};
    const T vals[] = {neg_inf, neg_one, neg_zero, zero, pos_one, inf, aNaN};
    for (T nan_val : nans) {
      for (T v : vals) {
        EXPECT_EQ(isgreater(nan_val, v), 0);
        EXPECT_EQ(isgreater(v, nan_val), 0);
        EXPECT_EQ(isgreaterequal(nan_val, v), 0);
        EXPECT_EQ(isgreaterequal(v, nan_val), 0);
        EXPECT_EQ(isless(nan_val, v), 0);
        EXPECT_EQ(isless(v, nan_val), 0);
        EXPECT_EQ(islessequal(nan_val, v), 0);
        EXPECT_EQ(islessequal(v, nan_val), 0);
        EXPECT_EQ(islessgreater(nan_val, v), 0);
        EXPECT_EQ(islessgreater(v, nan_val), 0);
        EXPECT_EQ(isunordered(nan_val, v), 1);
        EXPECT_EQ(isunordered(v, nan_val), 1);
      }
    }

    // Macro arguments with side effects must be evaluated once.
    T lhs = pos_one;
    T rhs = pos_two;
    EXPECT_EQ(isless(lhs++, rhs++), 1);
    EXPECT_FP_EQ(lhs, pos_two);
    EXPECT_FP_EQ(rhs, T(3));
  }
};

using LlvmLibcMathComparisonMacrosFloatTest = MathComparisonMacrosTest<float>;
TEST_F(LlvmLibcMathComparisonMacrosFloatTest, All) { testComparisons(); }

using LlvmLibcMathComparisonMacrosDoubleTest = MathComparisonMacrosTest<double>;
TEST_F(LlvmLibcMathComparisonMacrosDoubleTest, All) { testComparisons(); }

using LlvmLibcMathComparisonMacrosLongDoubleTest =
    MathComparisonMacrosTest<long double>;
TEST_F(LlvmLibcMathComparisonMacrosLongDoubleTest, All) { testComparisons(); }

#ifdef LIBC_TYPES_HAS_FLOAT16
using LlvmLibcMathComparisonMacrosFloat16Test =
    MathComparisonMacrosTest<float16>;
TEST_F(LlvmLibcMathComparisonMacrosFloat16Test, All) { testComparisons(); }
#endif // LIBC_TYPES_HAS_FLOAT16

#ifdef LIBC_TYPES_FLOAT128_IS_NOT_LONG_DOUBLE
using LlvmLibcMathComparisonMacrosFloat128Test =
    MathComparisonMacrosTest<float128>;
TEST_F(LlvmLibcMathComparisonMacrosFloat128Test, All) { testComparisons(); }
#endif // LIBC_TYPES_FLOAT128_IS_NOT_LONG_DOUBLE

TEST(LlvmLibcMathComparisonMacrosMixedTest, MixedTypes) {
  EXPECT_EQ(isgreater(2.0f, 1.0), 1);
  EXPECT_EQ(isgreaterequal(1.0f, 1.0L), 1);
  EXPECT_EQ(isless(1.0f, 2.0), 1);
  EXPECT_EQ(islessequal(2.0, 2.0L), 1);
  EXPECT_EQ(islessgreater(1.0f, 2.0L), 1);
  EXPECT_EQ(isunordered(__builtin_nanf(""), 1.0), 1);
  EXPECT_EQ(isunordered(1.0f, __builtin_nanl("")), 1);
}
