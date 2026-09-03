//===-- Utility class to test fxdivi functions ------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/CPP/type_traits/conditional.h"
#include "test/UnitTest/Test.h"

#include "hdr/signal_macros.h"
#include "src/__support/CPP/limits.h"
#include "src/__support/fixed_point/fx_rep.h"

namespace cpp = LIBC_NAMESPACE::cpp;

template <typename FXType, typename IntType>
class FxDiviTest : public LIBC_NAMESPACE::testing::Test {
  using FXRep = LIBC_NAMESPACE::fixed_point::FXRep<FXType>;

  static constexpr FXType fx_max = FXRep::MAX();
  static constexpr FXType fx_min = FXRep::MIN();
  static constexpr FXType fx_zero = FXRep::ZERO();
  static constexpr FXType epsilon = FXRep::EPS();
  static constexpr FXType one_half = FXRep::ONE_HALF();
  static constexpr FXType one_fourth = FXRep::ONE_FOURTH();
  static constexpr FXType one_eighth = FXRep::ONE_EIGHTH();

  static constexpr bool is_signed = (FXRep::SIGN_LEN > 0);
  static constexpr bool has_integral = (FXRep::INTEGRAL_LEN > 0);
  static constexpr int F = FXRep::FRACTION_LEN;

  static constexpr auto abs_diff = [](FXType a, FXType b) {
    return (a > b) ? (a - b) : (b - a);
  };

  using CompType =
      cpp::conditional_t<is_signed, long accum, unsigned long accum>;

  // Here, expected() uses CompType's own division operator as a reference which
  // has its own independent error bounds (division operator on CompType can
  // have almost 2 ulp of error as per ISO/IEC TR 18037:2008(E),
  // clause 4.1.6.2.1). The fxdivi implementation being tested here has an exact
  // fast path when the denominator is a power of 2, so do not use this helper
  // for such denominators. Instead, use the actual expected fixed-point
  // literal.
  static constexpr auto expected = [](IntType n, IntType d) -> FXType {
    return static_cast<FXType>(static_cast<CompType>(n) /
                               static_cast<CompType>(d));
  };

public:
  typedef FXType (*FxDiviFunc)(IntType, IntType);

  void testBasicNumbers(FxDiviFunc func) {
    EXPECT_TRUE(abs_diff(func(1, 3), expected(1, 3)) <= epsilon);
    EXPECT_TRUE(abs_diff(func(2, 3), expected(2, 3)) <= epsilon);
    EXPECT_EQ(func(3, 4), 3 * one_fourth);
    EXPECT_TRUE(abs_diff(func(5, 7), expected(5, 7)) <= epsilon);
    if constexpr (is_signed) {
      EXPECT_TRUE(abs_diff(func(-5, 7), expected(-5, 7)) <= epsilon);
    }

    EXPECT_TRUE(abs_diff(func(1043, 2764), expected(1043, 2764)) <= epsilon);
    EXPECT_TRUE(abs_diff(func(60000, 720293), expected(60000, 720293)) <=
                epsilon);

    EXPECT_EQ(func(128, 256), one_half);
    EXPECT_EQ(func(1, 2), one_half);
    EXPECT_EQ(func(1, 4), one_fourth);
    EXPECT_EQ(func(1, 8), one_eighth);
    EXPECT_EQ(func(1, 16), static_cast<FXType>(0.0625));
    if constexpr (is_signed) {
      EXPECT_EQ(func(-1, 2), -one_half);
      EXPECT_EQ(func(1, -4), -one_fourth);
      EXPECT_EQ(func(-1, 8), -one_eighth);
      EXPECT_EQ(func(1, -16), static_cast<FXType>(-0.0625));
    }

    if constexpr (has_integral) {
      EXPECT_TRUE(abs_diff(func(27, 23), expected(27, 23)) <= epsilon);
    }
  }

  void testEdgeCases(FxDiviFunc func) {
    constexpr IntType int_max = cpp::numeric_limits<IntType>::max();

    EXPECT_EQ(func(0, 10), fx_zero);
    if constexpr (is_signed) {
      EXPECT_EQ(func(0, -10), fx_zero);
    }

    if constexpr (is_signed && (F < cpp::numeric_limits<IntType>::digits)) {
      constexpr IntType edge = static_cast<IntType>(1) << F;
      EXPECT_EQ(func(-edge, edge), static_cast<FXType>(-1));
      if constexpr (has_integral) {
        EXPECT_EQ(func(edge - 1, edge), static_cast<FXType>(1) - epsilon);
      } else {
        EXPECT_EQ(func(edge - 1, edge), fx_max);
      }
    }

    if constexpr (has_integral) {
      EXPECT_EQ(func(int_max, int_max), static_cast<FXType>(1));
      EXPECT_TRUE(abs_diff(func(int_max - 1, int_max),
                           static_cast<FXType>(1) - epsilon) <= epsilon);
      EXPECT_EQ(func(int_max, 1), fx_max);
    } else {
      EXPECT_EQ(func(int_max, int_max), fx_max);
      EXPECT_EQ(func(int_max - 1, int_max), fx_max);
      EXPECT_EQ(func(27, 23), fx_max);
    }

    // Cannot EXPECT_EQ even though int_max is a power of 2 because rounding
    // direction for magnitudes smaller than the representable precision is
    // implementation defined. The result must be within 1 ulp.
    EXPECT_TRUE(abs_diff(func(1, int_max), fx_zero) <= epsilon);

    if constexpr (is_signed) {
      constexpr IntType int_min = cpp::numeric_limits<IntType>::min();

      if constexpr (has_integral) {
        EXPECT_TRUE(abs_diff(func(int_min, int_max), static_cast<FXType>(-1)) <=
                    epsilon);
        EXPECT_EQ(func(int_min, int_min), static_cast<FXType>(1));
      } else {
        EXPECT_EQ(func(int_min, int_max), fx_min);
        EXPECT_EQ(func(int_min, int_min), fx_max);
      }

      EXPECT_EQ(func(int_min, 1), fx_min);

      // Cannot EXPECT_EQ even though int_min is a power of 2 because rounding
      // direction for magnitudes smaller than the representable precision is
      // implementation defined. The result must be within 1 ulp.
      EXPECT_TRUE(abs_diff(func(1, int_min), fx_zero) <= epsilon);

      EXPECT_EQ(func(int_min, -1), fx_max);
      EXPECT_EQ(func(int_max, -1), fx_min);
    }

    if constexpr (has_integral) {
      EXPECT_EQ(func(1, 1), static_cast<FXType>(1));
      EXPECT_EQ(func(2, 1), static_cast<FXType>(2));
      EXPECT_EQ(func(3, 1), static_cast<FXType>(3));
    } else {
      EXPECT_EQ(func(1, 1), fx_max);
      EXPECT_EQ(func(2, 1), fx_max);
      EXPECT_EQ(func(3, 1), fx_max);
    }

    if constexpr (is_signed) {
      EXPECT_EQ(func(-1, 1), static_cast<FXType>(-1));
      EXPECT_EQ(func(1, -1), static_cast<FXType>(-1));

      if constexpr (has_integral) {
        EXPECT_EQ(func(-1, -1), static_cast<FXType>(1));
        EXPECT_EQ(func(3, -1), static_cast<FXType>(-3));
        EXPECT_EQ(func(-3, -1), static_cast<FXType>(3));
        EXPECT_EQ(func(-3, 1), static_cast<FXType>(-3));
      } else {
        EXPECT_EQ(func(-1, -1), fx_max);
        EXPECT_EQ(func(3, -1), fx_min);
        EXPECT_EQ(func(-3, -1), fx_max);
      }
    }

    if constexpr (has_integral) {
      constexpr IntType over_max =
          static_cast<IntType>(6) *
          (static_cast<IntType>(1) << FXRep::INTEGRAL_LEN);
      EXPECT_EQ(func(over_max, 3), fx_max);
      constexpr IntType at_max = static_cast<IntType>(1) << FXRep::INTEGRAL_LEN;
      EXPECT_EQ(func(at_max, 1), fx_max);
    }
  }

  void testWideOperands(FxDiviFunc func) {
    if constexpr (sizeof(IntType) * 8 > 32) {
      constexpr IntType big_pow2 = static_cast<IntType>(1) << 40;

      if constexpr (has_integral) {
        EXPECT_EQ(func(big_pow2, big_pow2), static_cast<FXType>(1));
      } else {
        EXPECT_EQ(func(big_pow2, big_pow2), fx_max);
      }
      EXPECT_EQ(func(big_pow2, big_pow2 << 1), one_half);
      if constexpr (is_signed) {
        EXPECT_EQ(func(-big_pow2, big_pow2), static_cast<FXType>(-1));
      }

      constexpr IntType big_non_pow2 = big_pow2 + 7;
      EXPECT_TRUE(abs_diff(func(3, big_non_pow2), fx_zero) <= epsilon);
      EXPECT_EQ(func(big_non_pow2, big_non_pow2 << 1), one_half);
    }
  }

  void testInvalidNumbers(FxDiviFunc func) {
    EXPECT_DEATH([func] { func(1, 0); }, WITH_SIGNAL(-1));
    if constexpr (is_signed) {
      EXPECT_DEATH([func] { func(-1, 0); }, WITH_SIGNAL(-1));
    }
  }
};

#if defined(LIBC_ADD_NULL_CHECKS)
#define LIST_FXDIVI_TESTS(Name, FXType, IntType, func)                         \
  using LlvmLibc##Name##Divi##Test = FxDiviTest<FXType, IntType>;              \
  TEST_F(LlvmLibc##Name##Divi##Test, InvalidNumbers) {                         \
    testInvalidNumbers(&func);                                                 \
  }                                                                            \
  TEST_F(LlvmLibc##Name##Divi##Test, BasicNumbers) {                           \
    testBasicNumbers(&func);                                                   \
  }                                                                            \
  TEST_F(LlvmLibc##Name##Divi##Test, EdgeCases) { testEdgeCases(&func); }      \
  TEST_F(LlvmLibc##Name##Divi##Test, WideOperands) {                           \
    testWideOperands(&func);                                                   \
  }                                                                            \
  static_assert(true, "Require semicolon.")
#else
#define LIST_FXDIVI_TESTS(Name, FXType, IntType, func)                         \
  using LlvmLibc##Name##Divi##Test = FxDiviTest<FXType, IntType>;              \
  TEST_F(LlvmLibc##Name##Divi##Test, BasicNumbers) {                           \
    testBasicNumbers(&func);                                                   \
  }                                                                            \
  TEST_F(LlvmLibc##Name##Divi##Test, EdgeCases) { testEdgeCases(&func); }      \
  TEST_F(LlvmLibc##Name##Divi##Test, WideOperands) {                           \
    testWideOperands(&func);                                                   \
  }                                                                            \
  static_assert(true, "Require semicolon.")
#endif // LIBC_ADD_NULL_CHECKS
