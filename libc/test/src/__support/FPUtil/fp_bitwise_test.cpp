//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unittests for bitwise operations on floating-point numbers.
///
//===----------------------------------------------------------------------===//

#include "src/__support/FPUtil/FPBits.h"
#include "src/__support/FPUtil/bfloat16.h"
#include "src/__support/FPUtil/fp_bitwise.h"
#include "src/__support/macros/properties/types.h"
#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"

using LIBC_NAMESPACE::fputil::BFloat16;
using LIBC_NAMESPACE::fputil::fp_and;
using LIBC_NAMESPACE::fputil::fp_andnot;
using LIBC_NAMESPACE::fputil::fp_not;
using LIBC_NAMESPACE::fputil::fp_or;
using LIBC_NAMESPACE::fputil::fp_xor;
using LIBC_NAMESPACE::fputil::FPBits;
using LIBC_NAMESPACE::fputil::scale_exp2;

template <typename T>
class LlvmLibcFPBitwiseTest : public LIBC_NAMESPACE::testing::Test {
  DECLARE_SPECIAL_CONSTANTS(T)

  static constexpr T pos_val = T(1.5);
  static constexpr T neg_val = T(-1.5);
  static constexpr T other_val = T(1.75);

  static constexpr StorageType bits_of(T x) { return FPBits(x).uintval(); }

public:
  void test_and() {
    EXPECT_EQ(bits_of(fp_and(pos_val, zero)), bits_of(zero));
    EXPECT_EQ(bits_of(fp_and(neg_val, zero)), bits_of(zero));
    EXPECT_EQ(bits_of(fp_and(pos_val, neg_zero)), bits_of(zero));
    EXPECT_EQ(bits_of(fp_and(neg_val, neg_zero)), bits_of(neg_zero));
    EXPECT_EQ(bits_of(fp_and(pos_val, pos_val)), bits_of(pos_val));
    EXPECT_EQ(bits_of(fp_and(neg_val, neg_val)), bits_of(neg_val));
    EXPECT_EQ(bits_of(fp_and(neg_val, pos_val)), bits_of(pos_val));
    EXPECT_EQ(bits_of(fp_and(pos_val, other_val)),
              static_cast<StorageType>(bits_of(pos_val) & bits_of(other_val)));
  }

  void test_or() {
    EXPECT_EQ(bits_of(fp_or(pos_val, zero)), bits_of(pos_val));
    EXPECT_EQ(bits_of(fp_or(neg_val, zero)), bits_of(neg_val));
    EXPECT_EQ(bits_of(fp_or(pos_val, neg_zero)), bits_of(neg_val));
    EXPECT_EQ(bits_of(fp_or(neg_val, neg_zero)), bits_of(neg_val));
    EXPECT_EQ(bits_of(fp_or(pos_val, pos_val)), bits_of(pos_val));
    EXPECT_EQ(bits_of(fp_or(pos_val, other_val)),
              static_cast<StorageType>(bits_of(pos_val) | bits_of(other_val)));
  }

  void test_xor() {
    EXPECT_EQ(bits_of(fp_xor(pos_val, zero)), bits_of(pos_val));
    EXPECT_EQ(bits_of(fp_xor(neg_val, zero)), bits_of(neg_val));
    EXPECT_EQ(bits_of(fp_xor(pos_val, neg_zero)), bits_of(neg_val));
    EXPECT_EQ(bits_of(fp_xor(neg_val, neg_zero)), bits_of(pos_val));
    EXPECT_EQ(bits_of(fp_xor(neg_val, pos_val)), bits_of(neg_zero));
    EXPECT_EQ(bits_of(fp_xor(pos_val, pos_val)), bits_of(zero));
    EXPECT_EQ(bits_of(fp_xor(neg_val, neg_val)), bits_of(zero));
    EXPECT_EQ(bits_of(fp_xor(pos_val, other_val)),
              static_cast<StorageType>(bits_of(pos_val) ^ bits_of(other_val)));
  }

  void test_andnot() {
    EXPECT_EQ(bits_of(fp_andnot(pos_val, zero)), bits_of(pos_val));
    EXPECT_EQ(bits_of(fp_andnot(neg_val, zero)), bits_of(neg_val));
    EXPECT_EQ(bits_of(fp_andnot(pos_val, neg_zero)), bits_of(pos_val));
    EXPECT_EQ(bits_of(fp_andnot(neg_val, neg_zero)), bits_of(pos_val));
    EXPECT_EQ(bits_of(fp_andnot(pos_val, pos_val)), bits_of(zero));
    EXPECT_EQ(bits_of(fp_andnot(neg_val, neg_val)), bits_of(zero));
    EXPECT_EQ(bits_of(fp_andnot(other_val, pos_val)),
              static_cast<StorageType>(bits_of(other_val) & ~bits_of(pos_val)));
  }

  void test_not() {
    EXPECT_EQ(bits_of(fp_not(fp_not(zero))), bits_of(zero));
    EXPECT_EQ(bits_of(fp_not(fp_not(neg_zero))), bits_of(neg_zero));
    EXPECT_EQ(bits_of(fp_not(fp_not(pos_val))), bits_of(pos_val));
    EXPECT_EQ(bits_of(fp_not(fp_not(neg_val))), bits_of(neg_val));
    EXPECT_EQ(bits_of(fp_and(pos_val, fp_not(pos_val))), bits_of(zero));
    EXPECT_EQ(bits_of(fp_and(neg_val, fp_not(neg_zero))), bits_of(pos_val));
    EXPECT_EQ(bits_of(fp_andnot(pos_val, other_val)),
              bits_of(fp_and(pos_val, fp_not(other_val))));
  }

  void test_scale_exp2() {
    EXPECT_EQ(bits_of(scale_exp2(pos_val, 0)), bits_of(pos_val));
    EXPECT_EQ(bits_of(scale_exp2(neg_val, 0)), bits_of(neg_val));
    EXPECT_EQ(bits_of(scale_exp2(pos_val, 2)), bits_of(T(6.0)));
    EXPECT_EQ(bits_of(scale_exp2(neg_val, 2)), bits_of(T(-6.0)));
    EXPECT_EQ(bits_of(scale_exp2(pos_val, -2)), bits_of(T(0.375)));
    EXPECT_EQ(bits_of(scale_exp2(neg_val, -2)), bits_of(T(-0.375)));
  }
};

using FPBitwiseTypes = LIBC_NAMESPACE::testing::TypeList<float, double,
#ifdef LIBC_TYPES_HAS_FLOAT16
                                                         float16,
#endif // LIBC_TYPES_HAS_FLOAT16
#ifdef LIBC_TYPES_HAS_NATIVE_FLOAT128
                                                         float128,
#endif // LIBC_TYPES_HAS_NATIVE_FLOAT128
                                                         BFloat16>;

TYPED_TEST_F(LlvmLibcFPBitwiseTest, And, FPBitwiseTypes) { this->test_and(); }
TYPED_TEST_F(LlvmLibcFPBitwiseTest, Or, FPBitwiseTypes) { this->test_or(); }
TYPED_TEST_F(LlvmLibcFPBitwiseTest, Xor, FPBitwiseTypes) { this->test_xor(); }
TYPED_TEST_F(LlvmLibcFPBitwiseTest, AndNot, FPBitwiseTypes) {
  this->test_andnot();
}
TYPED_TEST_F(LlvmLibcFPBitwiseTest, Not, FPBitwiseTypes) { this->test_not(); }
TYPED_TEST_F(LlvmLibcFPBitwiseTest, ScaleExp2, FPBitwiseTypes) {
  this->test_scale_exp2();
}

TEST(LlvmLibcFPBitwiseConstexprTest, ConstexprEval) {
  static_assert(FPBits<double>(fp_xor(-1.5, 1.5)).uintval() ==
                FPBits<double>::SIGN_MASK);
  static_assert(FPBits<double>(fp_andnot(-1.5, -0.0)).uintval() ==
                FPBits<double>(1.5).uintval());
  static_assert(FPBits<double>(fp_or(1.5, -0.0)).uintval() ==
                FPBits<double>(-1.5).uintval());
  static_assert(FPBits<double>(fp_and(-1.5, -0.0)).uintval() ==
                FPBits<double>::SIGN_MASK);
  static_assert(FPBits<double>(fp_not(fp_not(1.5))).uintval() ==
                FPBits<double>(1.5).uintval());
  static_assert(FPBits<double>(scale_exp2(1.5, 2)).uintval() ==
                FPBits<double>(6.0).uintval());
  static_assert(FPBits<double>(scale_exp2(1.5, -2)).uintval() ==
                FPBits<double>(0.375).uintval());

  static_assert(FPBits<float>(fp_xor(-1.5f, 1.5f)).uintval() ==
                FPBits<float>::SIGN_MASK);
  static_assert(FPBits<float>(fp_andnot(-1.5f, -0.0f)).uintval() ==
                FPBits<float>(1.5f).uintval());
  static_assert(FPBits<float>(scale_exp2(1.5f, 2)).uintval() ==
                FPBits<float>(6.0f).uintval());
  static_assert(FPBits<float>(scale_exp2(1.5f, -2)).uintval() ==
                FPBits<float>(0.375f).uintval());
}
