//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unittests for multiply_add, multiply_sub, and sub_multiply.
///
//===----------------------------------------------------------------------===//

#include "src/__support/FPUtil/double_double.h"
#include "src/__support/FPUtil/dyadic_float.h"
#include "src/__support/FPUtil/multiply_add.h"
#include "src/__support/frac128.h"
#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"

using LIBC_NAMESPACE::Frac128;
using LIBC_NAMESPACE::fputil::DoubleDouble;
using LIBC_NAMESPACE::fputil::DyadicFloat;
using LIBC_NAMESPACE::fputil::multiply_add;
using LIBC_NAMESPACE::fputil::multiply_sub;
using LIBC_NAMESPACE::fputil::sub_multiply;

template <typename T>
class LlvmLibcMultiplyAddTest : public LIBC_NAMESPACE::testing::Test {
  DECLARE_SPECIAL_CONSTANTS(T)

public:
  void test_identities() {
    constexpr T XS[5] = {T(0.0), T(-0.0), T(1.5), T(-2.25), T(0x1.3p-5)};
    constexpr T YS[4] = {T(0.0), T(0.75), T(-1.25), T(0x1.7p-4)};
    constexpr T ZS[4] = {T(0.0), T(-0.0), T(2.5), T(-0.625)};

    for (T x : XS) {
      for (T y : YS) {
        for (T z : ZS) {
          EXPECT_FP_EQ_ALL_ROUNDING(multiply_add(x, y, -z),
                                    multiply_sub(x, y, z));
          EXPECT_FP_EQ_ALL_ROUNDING(multiply_add(-y, z, x),
                                    sub_multiply(x, y, z));
          EXPECT_FP_EQ_ALL_ROUNDING(multiply_add(y, -z, x),
                                    sub_multiply(x, y, z));
          // Explicit template parameter calls:
          EXPECT_FP_EQ_ALL_ROUNDING(multiply_add<T>(x, y, z),
                                    multiply_add(x, y, z));
          EXPECT_FP_EQ_ALL_ROUNDING(multiply_sub<T>(x, y, z),
                                    multiply_sub(x, y, z));
          EXPECT_FP_EQ_ALL_ROUNDING(sub_multiply<T>(x, y, z),
                                    sub_multiply(x, y, z));
        }
      }
    }

    // Directed rounding on subnormals: x - x * 2^-25 vs x + x * (-2^-25).
    const T eps = T(0x1.0p-25);
    EXPECT_FP_EQ_ALL_ROUNDING(multiply_add(min_denormal, -eps, min_denormal),
                              sub_multiply(min_denormal, min_denormal, eps));
    EXPECT_FP_EQ_ALL_ROUNDING(
        multiply_add(neg_min_denormal, -eps, neg_min_denormal),
        sub_multiply(neg_min_denormal, neg_min_denormal, eps));
  }

  void test_fma_precision() {
    if constexpr (LIBC_NAMESPACE::fputil::internal::HasHardwareFma<T>::value) {
      // With a = 1 + 2^-12, a * a = 1 + 2^-11 + 2^-24:
      // single-rounded FMA preserves the 2^-24 low bit when 1 is cancelled.
      const T a = T(1.0) + T(0x1.0p-12);
      const T expected = T(0x1.0p-11) + T(0x1.0p-24);
      EXPECT_FP_EQ(expected, multiply_add(a, a, T(-1.0)));
      EXPECT_FP_EQ(expected, multiply_sub(a, a, T(1.0)));
      EXPECT_FP_EQ(-expected, sub_multiply(T(1.0), a, a));
    }
  }

  void test_constexpr() {
#if LIBC_HAS_BUILTIN_IS_CONSTANT_EVALUATED
    static_assert(multiply_add(T(1.5), T(2.0), T(0.25)) == T(3.25));
    static_assert(multiply_sub(T(1.5), T(2.0), T(0.25)) == T(2.75));
    static_assert(sub_multiply(T(3.5), T(1.5), T(2.0)) == T(0.5));
#endif
  }
};

using MultiplyAddFPTypes = LIBC_NAMESPACE::testing::Types<float, double>;

TYPED_TEST_SUITE(LlvmLibcMultiplyAddTest, MultiplyAddFPTypes);

TYPED_TEST(LlvmLibcMultiplyAddTest, Identities) { this->test_identities(); }

TYPED_TEST(LlvmLibcMultiplyAddTest, FmaPrecision) {
  this->test_fma_precision();
}

TYPED_TEST(LlvmLibcMultiplyAddTest, ConstexprEval) { this->test_constexpr(); }

TEST(LlvmLibcMultiplyAddStructTest, DoubleDoubleOps) {
  const DoubleDouble a{0x1.0p-60, 2.0};
  const DoubleDouble b{0.0, 0.5};
  const DoubleDouble c{0x1.0p-62, 0.25};

  // sub overloads:
  DoubleDouble s_dd = LIBC_NAMESPACE::fputil::sub(a, c);
  EXPECT_FP_EQ(1.75, s_dd.hi);
  EXPECT_FP_EQ(0x1.0p-60 - 0x1.0p-62, s_dd.lo);

  DoubleDouble s_ds = LIBC_NAMESPACE::fputil::sub(a, 0.25);
  EXPECT_FP_EQ(1.75, s_ds.hi);
  EXPECT_FP_EQ(0x1.0p-60, s_ds.lo);

  DoubleDouble s_sd = LIBC_NAMESPACE::fputil::sub(2.0, c);
  EXPECT_FP_EQ(1.75, s_sd.hi);
  EXPECT_FP_EQ(-0x1.0p-62, s_sd.lo);

  // multiply_add(0.5, 0.25 + 2^-62, 2.0 + 2^-60) = 2.125 + 2^-60 + 2^-63.
  DoubleDouble ma = multiply_add(b, c, a);
  EXPECT_FP_EQ(2.125, ma.hi);
  EXPECT_FP_EQ(0x1.0p-60 + 0x1.0p-63, ma.lo);

  // multiply_sub(2.0 + 2^-60, 0.5, 0.25 + 2^-62) = 0.75 + 2^-61 - 2^-62.
  DoubleDouble ms = multiply_sub(a, b, c);
  EXPECT_FP_EQ(0.75, ms.hi);
  EXPECT_FP_EQ(0x1.0p-61 - 0x1.0p-62, ms.lo);

  // sub_multiply(2.0 + 2^-60, 0.5, 0.25 + 2^-62) = 1.875 + 2^-60 - 2^-63.
  DoubleDouble sm = sub_multiply(a, b, c);
  EXPECT_FP_EQ(1.875, sm.hi);
  EXPECT_FP_EQ(0x1.0p-60 - 0x1.0p-63, sm.lo);
}

TEST(LlvmLibcMultiplyAddStructTest, DyadicFloatOps) {
  using DFloat128 = DyadicFloat<128>;
  constexpr DFloat128 A(2.0);
  constexpr DFloat128 B(0.5);
  constexpr DFloat128 C(0.25);

  constexpr DFloat128 MA = multiply_add(B, C, A);
  constexpr DFloat128 MS = multiply_sub(B, C, A);
  constexpr DFloat128 SM = sub_multiply(A, B, C);
  static_assert(MS.mantissa == SM.mantissa && MS.exponent == SM.exponent &&
                MS.sign != SM.sign);

  EXPECT_FP_EQ(2.125, static_cast<double>(MA));
  EXPECT_FP_EQ(-1.875, static_cast<double>(MS));
  EXPECT_FP_EQ(1.875, static_cast<double>(SM));
}

TEST(LlvmLibcMultiplyAddStructTest, Frac128Ops) {
  constexpr Frac128 A({0, 0x8000'0000'0000'0000ULL}); // 0.5
  constexpr Frac128 B({0, 0x8000'0000'0000'0000ULL}); // 0.5
  constexpr Frac128 C({0, 0x2000'0000'0000'0000ULL}); // 0.125

  // 0.5 * 0.5 + 0.125 = 0.375; 0.5 * 0.5 - 0.125 = 0.125.
  constexpr Frac128 EXP_SUM({0, 0x6000'0000'0000'0000ULL});  // 0.375
  constexpr Frac128 EXP_DIFF({0, 0x2000'0000'0000'0000ULL}); // 0.125
  // 0.5 - 0.5 * 0.125 = 0.4375.
  constexpr Frac128 EXP_SM({0, 0x7000'0000'0000'0000ULL}); // 0.4375

  static_assert(EXP_SUM == multiply_add(A, B, C));
  static_assert(EXP_DIFF == multiply_sub(A, B, C));
  static_assert(EXP_SM == sub_multiply(A, B, C));

  EXPECT_TRUE(EXP_SUM == multiply_add(A, B, C));
  EXPECT_TRUE(EXP_DIFF == multiply_sub(A, B, C));
  EXPECT_TRUE(EXP_SM == sub_multiply(A, B, C));
}
