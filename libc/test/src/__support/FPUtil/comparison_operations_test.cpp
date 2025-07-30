//===-- Unittests for comparison operations on floating-point numbers -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/FPUtil/bfloat16.h"
#include "src/__support/FPUtil/comparison_operations.h"
#include "src/__support/macros/properties/types.h"
#include "test/UnitTest/FEnvSafeTest.h"
#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"

using LIBC_NAMESPACE::fputil::equals;
using LIBC_NAMESPACE::fputil::greater_than;
using LIBC_NAMESPACE::fputil::greater_than_or_equals;
using LIBC_NAMESPACE::fputil::less_than;
using LIBC_NAMESPACE::fputil::less_than_or_equals;

using BFloat16 = LIBC_NAMESPACE::fputil::BFloat16;

template <typename T>
class ComparisonOperationsTest : public LIBC_NAMESPACE::testing::FEnvSafeTest {
  DECLARE_SPECIAL_CONSTANTS(T)

  static constexpr T normal1 = T(3.14);
  static constexpr T neg_normal1 = T(-3.14);
  static constexpr T normal2 = T(2.71);
  static constexpr T small = T(0.1);
  static constexpr T neg_small = T(-0.1);
  static constexpr T large = T(10000.0);
  static constexpr T neg_large = T(-10000.0);

public:
  void test_equals() {
    EXPECT_TRUE(equals(neg_zero, neg_zero));
    EXPECT_TRUE(equals(zero, neg_zero));
    EXPECT_TRUE(equals(neg_zero, zero));

    EXPECT_TRUE(equals(inf, inf));
    EXPECT_TRUE(equals(neg_inf, neg_inf));
    EXPECT_FALSE(equals(inf, neg_inf));
    EXPECT_FALSE(equals(neg_inf, inf));

    EXPECT_TRUE(equals(normal1, normal1));
    EXPECT_TRUE(equals(normal2, normal2));
    EXPECT_FALSE(equals(normal1, normal2));
    EXPECT_FALSE(equals(normal1, neg_normal1));

    auto test_qnan = [&](T x, T y) {
      LIBC_NAMESPACE::fputil::clear_except(FE_ALL_EXCEPT);
      EXPECT_FALSE(equals(x, y));
      EXPECT_FP_EXCEPTION(0);
    };

    test_qnan(aNaN, aNaN);
    test_qnan(aNaN, neg_aNaN);
    test_qnan(aNaN, zero);
    test_qnan(aNaN, inf);
    test_qnan(aNaN, normal1);

    test_qnan(neg_aNaN, neg_aNaN);
    test_qnan(neg_aNaN, aNaN);
    test_qnan(neg_aNaN, zero);
    test_qnan(neg_aNaN, inf);
    test_qnan(neg_aNaN, normal1);

    auto test_snan = [&](T x, T y) {
      LIBC_NAMESPACE::fputil::clear_except(FE_ALL_EXCEPT);
      EXPECT_FALSE(equals(x, y));
      EXPECT_FP_EXCEPTION(FE_INVALID);
    };

    test_snan(sNaN, sNaN);
    test_snan(sNaN, neg_sNaN);
    test_snan(sNaN, aNaN);
    test_snan(sNaN, neg_aNaN);
    test_snan(sNaN, zero);
    test_snan(sNaN, neg_zero);
    test_snan(sNaN, inf);
    test_snan(sNaN, neg_inf);
    test_snan(sNaN, normal1);

    test_snan(neg_sNaN, neg_sNaN);
    test_snan(neg_sNaN, sNaN);
    test_snan(neg_sNaN, aNaN);
    test_snan(neg_sNaN, neg_aNaN);
    test_snan(neg_sNaN, zero);
    test_snan(neg_sNaN, neg_zero);
    test_snan(neg_sNaN, inf);
    test_snan(neg_sNaN, neg_inf);
    test_snan(neg_sNaN, normal1);
  }

  void test_less_than() {
    EXPECT_TRUE(less_than(neg_small, small));
    EXPECT_TRUE(less_than(small, large));

    EXPECT_TRUE(less_than(neg_large, neg_small));
    EXPECT_FALSE(less_than(large, small));
    EXPECT_FALSE(less_than(small, neg_small));

    EXPECT_FALSE(less_than(zero, neg_zero));
    EXPECT_FALSE(less_than(neg_zero, zero));
    EXPECT_FALSE(less_than(zero, zero));

    EXPECT_TRUE(less_than(neg_small, zero));
    EXPECT_TRUE(less_than(neg_zero, small));
    EXPECT_FALSE(less_than(small, zero));

    EXPECT_TRUE(less_than(neg_inf, inf));
    EXPECT_TRUE(less_than(neg_inf, neg_small));
    EXPECT_TRUE(less_than(small, inf));
    EXPECT_FALSE(less_than(inf, small));

    EXPECT_FALSE(less_than(small, small));
    EXPECT_FALSE(less_than(neg_inf, neg_inf));

    auto test_qnan = [&](T x, T y) {
      LIBC_NAMESPACE::fputil::clear_except(FE_ALL_EXCEPT);
      EXPECT_FALSE(less_than(x, y));
      EXPECT_FP_EXCEPTION(FE_INVALID);
    };

    test_qnan(aNaN, small);
    test_qnan(small, aNaN);
    test_qnan(aNaN, aNaN);
    test_qnan(neg_aNaN, neg_small);
    test_qnan(neg_small, neg_aNaN);
    test_qnan(neg_aNaN, neg_aNaN);

    auto test_snan = [&](T x, T y) {
      LIBC_NAMESPACE::fputil::clear_except(FE_ALL_EXCEPT);
      EXPECT_FALSE(less_than(x, y));
      EXPECT_FP_EXCEPTION(FE_INVALID);
    };

    test_snan(sNaN, small);
    test_snan(sNaN, neg_small);
    test_snan(sNaN, zero);
    test_snan(sNaN, inf);
    test_snan(sNaN, aNaN);
    test_snan(sNaN, sNaN);

    test_snan(neg_sNaN, small);
    test_snan(neg_sNaN, neg_small);
    test_snan(neg_sNaN, zero);
    test_snan(neg_sNaN, inf);
    test_snan(neg_sNaN, aNaN);
    test_snan(neg_sNaN, neg_sNaN);
  }

  void test_greater_than() {
    EXPECT_TRUE(greater_than(large, neg_small));
    EXPECT_TRUE(greater_than(neg_small, neg_large));

    EXPECT_FALSE(greater_than(large, large));
    EXPECT_FALSE(greater_than(neg_small, large));

    EXPECT_FALSE(greater_than(zero, neg_zero));
    EXPECT_FALSE(greater_than(neg_zero, zero));

    EXPECT_TRUE(greater_than(inf, neg_inf));
    EXPECT_TRUE(greater_than(inf, large));
    EXPECT_TRUE(greater_than(large, neg_inf));
    EXPECT_FALSE(greater_than(neg_inf, inf));

    EXPECT_FALSE(greater_than(large, large));
    EXPECT_FALSE(greater_than(inf, inf));

    auto test_qnan = [&](T x, T y) {
      LIBC_NAMESPACE::fputil::clear_except(FE_ALL_EXCEPT);
      EXPECT_FALSE(greater_than(x, y));
      EXPECT_FP_EXCEPTION(FE_INVALID);
    };

    test_qnan(aNaN, large);
    test_qnan(large, aNaN);
    test_qnan(aNaN, aNaN);
    test_qnan(neg_aNaN, neg_small);
    test_qnan(neg_small, neg_aNaN);
    test_qnan(neg_aNaN, neg_aNaN);

    auto test_snan = [&](T x, T y) {
      LIBC_NAMESPACE::fputil::clear_except(FE_ALL_EXCEPT);
      EXPECT_FALSE(greater_than(x, y));
      EXPECT_FP_EXCEPTION(FE_INVALID);
    };

    test_snan(sNaN, large);
    test_snan(sNaN, neg_small);
    test_snan(sNaN, zero);
    test_snan(sNaN, inf);
    test_snan(sNaN, aNaN);
    test_snan(sNaN, sNaN);

    test_snan(neg_sNaN, large);
    test_snan(neg_sNaN, neg_small);
    test_snan(neg_sNaN, zero);
    test_snan(neg_sNaN, inf);
    test_snan(neg_sNaN, aNaN);
    test_snan(neg_sNaN, neg_sNaN);
  }

  void test_less_than_or_equals() {
    EXPECT_TRUE(less_than_or_equals(neg_small, small));
    EXPECT_TRUE(less_than_or_equals(small, large));
    EXPECT_TRUE(less_than_or_equals(neg_inf, small));

    EXPECT_TRUE(less_than_or_equals(small, small));
    EXPECT_TRUE(less_than_or_equals(zero, neg_zero));
    EXPECT_TRUE(less_than_or_equals(inf, inf));

    EXPECT_FALSE(less_than_or_equals(small, neg_small));
    EXPECT_FALSE(less_than_or_equals(large, small));
    EXPECT_FALSE(less_than_or_equals(inf, small));

    EXPECT_TRUE(less_than_or_equals(neg_large, small));
    EXPECT_FALSE(less_than_or_equals(large, neg_small));

    auto test_qnan = [&](T x, T y) {
      LIBC_NAMESPACE::fputil::clear_except(FE_ALL_EXCEPT);
      EXPECT_FALSE(less_than_or_equals(x, y));
      EXPECT_FP_EXCEPTION(FE_INVALID);
    };

    test_qnan(aNaN, small);
    test_qnan(small, aNaN);
    test_qnan(aNaN, aNaN);
    test_qnan(neg_aNaN, neg_small);
    test_qnan(neg_small, neg_aNaN);
    test_qnan(neg_aNaN, neg_aNaN);

    auto test_snan = [&](T x, T y) {
      LIBC_NAMESPACE::fputil::clear_except(FE_ALL_EXCEPT);
      EXPECT_FALSE(less_than_or_equals(x, y));
      EXPECT_FP_EXCEPTION(FE_INVALID);
    };

    test_snan(sNaN, small);
    test_snan(sNaN, neg_small);
    test_snan(sNaN, zero);
    test_snan(sNaN, inf);
    test_snan(sNaN, aNaN);
    test_snan(sNaN, sNaN);

    test_snan(neg_sNaN, small);
    test_snan(neg_sNaN, neg_small);
    test_snan(neg_sNaN, zero);
    test_snan(neg_sNaN, inf);
    test_snan(neg_sNaN, aNaN);
    test_snan(neg_sNaN, neg_sNaN);
  }

  void test_greater_than_or_equals() {
    EXPECT_TRUE(greater_than_or_equals(small, neg_small));
    EXPECT_TRUE(greater_than_or_equals(large, small));
    EXPECT_TRUE(greater_than_or_equals(inf, small));

    EXPECT_TRUE(greater_than_or_equals(small, small));
    EXPECT_TRUE(greater_than_or_equals(zero, neg_zero));
    EXPECT_TRUE(greater_than_or_equals(neg_inf, neg_inf));

    EXPECT_FALSE(greater_than_or_equals(neg_small, small));
    EXPECT_FALSE(greater_than_or_equals(small, large));
    EXPECT_FALSE(greater_than_or_equals(neg_inf, small));

    EXPECT_TRUE(greater_than_or_equals(large, neg_small));
    EXPECT_FALSE(greater_than_or_equals(neg_large, small));

    auto test_qnan = [&](T x, T y) {
      LIBC_NAMESPACE::fputil::clear_except(FE_ALL_EXCEPT);
      EXPECT_FALSE(greater_than_or_equals(x, y));
      EXPECT_FP_EXCEPTION(FE_INVALID);
    };

    test_qnan(aNaN, small);
    test_qnan(small, aNaN);
    test_qnan(aNaN, aNaN);
    test_qnan(neg_aNaN, neg_small);
    test_qnan(neg_small, neg_aNaN);
    test_qnan(neg_aNaN, neg_aNaN);

    auto test_snan = [&](T x, T y) {
      LIBC_NAMESPACE::fputil::clear_except(FE_ALL_EXCEPT);
      EXPECT_FALSE(greater_than_or_equals(x, y));
      EXPECT_FP_EXCEPTION(FE_INVALID);
    };

    test_snan(sNaN, small);
    test_snan(sNaN, neg_small);
    test_snan(sNaN, zero);
    test_snan(sNaN, inf);
    test_snan(sNaN, aNaN);
    test_snan(sNaN, sNaN);

    test_snan(neg_sNaN, small);
    test_snan(neg_sNaN, neg_small);
    test_snan(neg_sNaN, zero);
    test_snan(neg_sNaN, inf);
    test_snan(neg_sNaN, aNaN);
    test_snan(neg_sNaN, neg_sNaN);
  }
};

#define TEST_COMPARISON_OPS(Name, Type)                                        \
  using LlvmLibc##Name##ComparisonOperationsTest =                             \
      ComparisonOperationsTest<Type>;                                          \
  TEST_F(LlvmLibc##Name##ComparisonOperationsTest, Equals) { test_equals(); }  \
  TEST_F(LlvmLibc##Name##ComparisonOperationsTest, LessThan) {                 \
    test_less_than();                                                          \
  }                                                                            \
  TEST_F(LlvmLibc##Name##ComparisonOperationsTest, GreaterThan) {              \
    test_greater_than();                                                       \
  }                                                                            \
  TEST_F(LlvmLibc##Name##ComparisonOperationsTest, LessThanOrEquals) {         \
    test_less_than_or_equals();                                                \
  }                                                                            \
  TEST_F(LlvmLibc##Name##ComparisonOperationsTest, GreaterThanOrEquals) {      \
    test_greater_than_or_equals();                                             \
  }

TEST_COMPARISON_OPS(Float, float)
TEST_COMPARISON_OPS(Double, double)
TEST_COMPARISON_OPS(LongDouble, long double)

#ifdef LIBC_TYPES_HAS_FLOAT16
TEST_COMPARISON_OPS(Float16, float16)
#endif // LIBC_TYPES_HAS_FLOAT16

#ifdef LIBC_TYPES_HAS_FLOAT128
TEST_COMPARISON_OPS(Float128, float128)
#endif // LIBC_TYPES_HAS_FLOAT128

TEST_COMPARISON_OPS(BFloat16, BFloat16)
