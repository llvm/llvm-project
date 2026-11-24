//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unit tests for floating-point integer check functions.
///
//===----------------------------------------------------------------------===//

#include "src/__support/FPUtil/FPBits.h"
#include "src/__support/FPUtil/bfloat16.h"
#include "src/__support/FPUtil/float128.h"
#include "src/__support/FPUtil/is_integer.h"
#include "src/__support/macros/properties/types.h"
#include "test/UnitTest/FEnvSafeTest.h"
#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"

using LIBC_NAMESPACE::fputil::BFloat16;
using LIBC_NAMESPACE::fputil::Float128;
using LIBC_NAMESPACE::fputil::is_integer;
using LIBC_NAMESPACE::fputil::is_odd_integer;

template <typename T>
class IsIntegerTest : public LIBC_NAMESPACE::testing::FEnvSafeTest {
  DECLARE_SPECIAL_CONSTANTS(T)

  static T get_two_pow_frac() {
    T result = T(1.0);
    for (int i = 0; i < FPBits::FRACTION_LEN; ++i)
      result *= T(2.0);
    return result;
  }

public:
  void test_is_integer() {
    EXPECT_TRUE(is_integer(zero));
    EXPECT_TRUE(is_integer(neg_zero));

    EXPECT_TRUE(is_integer(T(1.0)));
    EXPECT_TRUE(is_integer(T(-1.0)));
    EXPECT_TRUE(is_integer(T(2.0)));
    EXPECT_TRUE(is_integer(T(-2.0)));
    EXPECT_TRUE(is_integer(T(3.0)));
    EXPECT_TRUE(is_integer(T(-3.0)));
    EXPECT_TRUE(is_integer(T(10.0)));
    EXPECT_TRUE(is_integer(T(-10.0)));
    EXPECT_TRUE(is_integer(T(100.0)));
    EXPECT_TRUE(is_integer(T(-100.0)));

    EXPECT_FALSE(is_integer(T(0.5)));
    EXPECT_FALSE(is_integer(T(-0.5)));
    EXPECT_FALSE(is_integer(T(1.5)));
    EXPECT_FALSE(is_integer(T(-1.5)));
    EXPECT_FALSE(is_integer(T(0.25)));
    EXPECT_FALSE(is_integer(T(-0.25)));
    EXPECT_FALSE(is_integer(T(0.75)));
    EXPECT_FALSE(is_integer(T(-0.75)));

    EXPECT_FALSE(is_integer(min_denormal));
    EXPECT_FALSE(is_integer(neg_min_denormal));
    EXPECT_FALSE(is_integer(max_denormal));

    T two_pow_frac = get_two_pow_frac();
    EXPECT_TRUE(is_integer(two_pow_frac));
    EXPECT_TRUE(is_integer(-two_pow_frac));

    EXPECT_TRUE(is_integer(T(two_pow_frac - T(1.0))));
    EXPECT_TRUE(is_integer(T(-(two_pow_frac - T(1.0)))));

    EXPECT_TRUE(is_integer(T(two_pow_frac + T(1.0))));
    EXPECT_TRUE(is_integer(T(-(two_pow_frac + T(1.0)))));

    EXPECT_FALSE(is_integer(T(two_pow_frac - T(1.0) + T(0.5))));
    EXPECT_FALSE(is_integer(T(-(two_pow_frac - T(1.0) + T(0.5)))));
  }

  void test_is_odd_integer() {
    EXPECT_FALSE(is_odd_integer(zero));
    EXPECT_FALSE(is_odd_integer(neg_zero));

    EXPECT_TRUE(is_odd_integer(T(1.0)));
    EXPECT_TRUE(is_odd_integer(T(-1.0)));
    EXPECT_FALSE(is_odd_integer(T(2.0)));
    EXPECT_FALSE(is_odd_integer(T(-2.0)));
    EXPECT_TRUE(is_odd_integer(T(3.0)));
    EXPECT_TRUE(is_odd_integer(T(-3.0)));
    EXPECT_FALSE(is_odd_integer(T(4.0)));
    EXPECT_FALSE(is_odd_integer(T(-4.0)));
    EXPECT_TRUE(is_odd_integer(T(5.0)));
    EXPECT_TRUE(is_odd_integer(T(-5.0)));
    EXPECT_FALSE(is_odd_integer(T(10.0)));
    EXPECT_FALSE(is_odd_integer(T(-10.0)));
    EXPECT_TRUE(is_odd_integer(T(11.0)));
    EXPECT_TRUE(is_odd_integer(T(-11.0)));

    EXPECT_FALSE(is_odd_integer(T(0.5)));
    EXPECT_FALSE(is_odd_integer(T(-0.5)));
    EXPECT_FALSE(is_odd_integer(T(1.5)));
    EXPECT_FALSE(is_odd_integer(T(-1.5)));
    EXPECT_FALSE(is_odd_integer(T(2.5)));
    EXPECT_FALSE(is_odd_integer(T(-2.5)));

    EXPECT_FALSE(is_odd_integer(min_denormal));
    EXPECT_FALSE(is_odd_integer(neg_min_denormal));
    EXPECT_FALSE(is_odd_integer(max_denormal));

    T two_pow_frac = get_two_pow_frac();
    EXPECT_FALSE(is_odd_integer(two_pow_frac));
    EXPECT_FALSE(is_odd_integer(-two_pow_frac));

    EXPECT_TRUE(is_odd_integer(T(two_pow_frac - T(1.0))));
    EXPECT_TRUE(is_odd_integer(T(-(two_pow_frac - T(1.0)))));

    EXPECT_TRUE(is_odd_integer(T(two_pow_frac + T(1.0))));
    EXPECT_TRUE(is_odd_integer(T(-(two_pow_frac + T(1.0)))));

    EXPECT_FALSE(is_odd_integer(T(two_pow_frac + T(2.0))));
    EXPECT_FALSE(is_odd_integer(T(-(two_pow_frac + T(2.0)))));

    T two_pow_frac_plus_1 = two_pow_frac * T(2.0);
    EXPECT_FALSE(is_odd_integer(two_pow_frac_plus_1));
    EXPECT_FALSE(is_odd_integer(-two_pow_frac_plus_1));
  }
};

#define TEST_IS_INTEGER(Name, Type)                                            \
  using LlvmLibc##Name##IsIntegerTest = IsIntegerTest<Type>;                   \
  TEST_F(LlvmLibc##Name##IsIntegerTest, IsInteger) { test_is_integer(); }      \
  TEST_F(LlvmLibc##Name##IsIntegerTest, IsOddInteger) { test_is_odd_integer(); }

TEST_IS_INTEGER(Float, float)
TEST_IS_INTEGER(Double, double)
TEST_IS_INTEGER(LongDouble, long double)

#ifdef LIBC_TYPES_HAS_FLOAT16
TEST_IS_INTEGER(Float16, float16)
#endif // LIBC_TYPES_HAS_FLOAT16

#ifdef LIBC_TYPES_HAS_NATIVE_FLOAT128
TEST_IS_INTEGER(DFloat128, float128)
#endif // LIBC_TYPES_HAS_NATIVE_FLOAT128

TEST_IS_INTEGER(Float128, Float128)
TEST_IS_INTEGER(BFloat16, BFloat16)

TEST(LlvmLibcIsIntegerConstexprTest, ConstexprEval) {
  static_assert(is_integer(0.0));
  static_assert(is_integer(-0.0));
  static_assert(is_integer(1.0));
  static_assert(is_integer(-1.0));
  static_assert(is_integer(2.0));
  static_assert(is_integer(-2.0));
  static_assert(!is_integer(0.5));
  static_assert(!is_integer(-0.5));
  static_assert(!is_integer(1.5));
  static_assert(!is_integer(-1.5));

  static_assert(!is_odd_integer(0.0));
  static_assert(!is_odd_integer(-0.0));
  static_assert(is_odd_integer(1.0));
  static_assert(is_odd_integer(-1.0));
  static_assert(!is_odd_integer(2.0));
  static_assert(!is_odd_integer(-2.0));
  static_assert(is_odd_integer(3.0));
  static_assert(is_odd_integer(-3.0));
  static_assert(!is_odd_integer(0.5));
  static_assert(!is_odd_integer(-0.5));
  static_assert(!is_odd_integer(1.5));
  static_assert(!is_odd_integer(-1.5));

  static_assert(is_integer(0.0f));
  static_assert(is_integer(-0.0f));
  static_assert(is_integer(1.0f));
  static_assert(is_integer(-1.0f));
  static_assert(is_integer(2.0f));
  static_assert(!is_integer(0.5f));

  static_assert(!is_odd_integer(0.0f));
  static_assert(is_odd_integer(1.0f));
  static_assert(!is_odd_integer(2.0f));
  static_assert(is_odd_integer(3.0f));
  static_assert(!is_odd_integer(0.5f));
}
