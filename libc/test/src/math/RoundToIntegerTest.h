//===-- Utility class to test different flavors of [l|ll]round --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_TEST_SRC_MATH_ROUNDTOINTEGERTEST_H
#define LLVM_LIBC_TEST_SRC_MATH_ROUNDTOINTEGERTEST_H

#include "test/UnitTest/RoundingModeUtils.h"
#include <cfenv>
#undef LIBC_MATH_USE_SYSTEM_FENV

#include "src/__support/CPP/algorithm.h"
#include "src/__support/FPUtil/FEnvImpl.h"
#include "src/__support/FPUtil/FPBits.h"
#include "src/__support/macros/properties/architectures.h"
#include "test/UnitTest/FEnvSafeTest.h"
#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"
#include "utils/MPFRWrapper/MPFRUtils.h"

#include "hdr/math_macros.h"

namespace mpfr = LIBC_NAMESPACE::testing::mpfr;
using LIBC_NAMESPACE::Sign;

template <typename FloatType, typename IntType, bool TestModes = false>
class RoundToIntegerTestTemplate
    : public LIBC_NAMESPACE::testing::FEnvSafeTest {
public:
  typedef IntType (*RoundToIntegerFunc)(FloatType);

private:
  using FPBits = LIBC_NAMESPACE::fputil::FPBits<FloatType>;
  using FPTest = LIBC_NAMESPACE::testing::FPTest<FloatType>;
  using RoundingMode = LIBC_NAMESPACE::fputil::testing::RoundingMode;
  using StorageType = typename FPBits::StorageType;

  const FloatType zero = FPBits::zero().get_val();
  const FloatType neg_zero = FPBits::zero(Sign::NEG).get_val();
  const FloatType inf = FPBits::inf().get_val();
  const FloatType neg_inf = FPBits::inf(Sign::NEG).get_val();
  const FloatType nan = FPBits::quiet_nan().get_val();

  static constexpr StorageType MAX_NORMAL = FPBits::max_normal().uintval();
  static constexpr StorageType MIN_NORMAL = FPBits::min_normal().uintval();
  static constexpr StorageType MAX_SUBNORMAL =
      FPBits::max_subnormal().uintval();
  static constexpr StorageType MIN_SUBNORMAL =
      FPBits::min_subnormal().uintval();

  static constexpr IntType INTEGER_MIN = IntType(1)
                                         << (sizeof(IntType) * 8 - 1);
  static constexpr IntType INTEGER_MAX = -(INTEGER_MIN + 1);

  void test_one_input(RoundToIntegerFunc func, FloatType input,
                      IntType expected, bool expectError) {
    libc_errno = 0;
    LIBC_NAMESPACE::fputil::clear_except(FE_ALL_EXCEPT);
    ASSERT_EQ(func(input), expected);
    // TODO: Handle the !expectError case. It used to expect
    // 0 for errno and exceptions, but this doesn't hold for
    // all math functions using RoundToInteger test:
    // https://github.com/llvm/llvm-project/pull/88816
    if (expectError) {
      ASSERT_FP_EXCEPTION(FE_INVALID);
      ASSERT_MATH_ERRNO(EDOM);
    }
  }

public:
  void SetUp() override {
    LIBC_NAMESPACE::testing::FEnvSafeTest::SetUp();

    if (math_errhandling & MATH_ERREXCEPT) {
      // We will disable all exceptions so that the test will not
      // crash with SIGFPE. We can still use fetestexcept to check
      // if the appropriate flag was raised.
      LIBC_NAMESPACE::fputil::disable_except(FE_ALL_EXCEPT);
    }
  }

  void testInfinityAndNaN(RoundToIntegerFunc func) {
    libc_errno = 0;
    LIBC_NAMESPACE::fputil::clear_except(FE_ALL_EXCEPT);
    ASSERT_EQ_ALL_ROUNDING(INTEGER_MAX, func(inf));
    ASSERT_EQ_ALL_ROUNDING(INTEGER_MIN, func(neg_inf));
    ASSERT_FP_EXCEPTION(FE_INVALID);
    ASSERT_MATH_ERRNO(EDOM);
    // This is currently never enabled, the
    // LLVM_LIBC_IMPLEMENTATION_DEFINED_TEST_BEHAVIOR CMake option in
    // libc/CMakeLists.txt is not forwarded to C++.
#if LIBC_COPT_IMPLEMENTATION_DEFINED_TEST_BEHAVIOR
    // Result is not well-defined, we always returns INTEGER_MAX
    test_one_input(func, nan, INTEGER_MAX, true);
#endif // LIBC_COPT_IMPLEMENTATION_DEFINED_TEST_BEHAVIOR
  }

  void testRoundNumbers(RoundToIntegerFunc func) {
    ASSERT_EQ_ALL_ROUNDING(IntType(0), func(zero));
    ASSERT_EQ_ALL_ROUNDING(IntType(0), func(neg_zero));
    ASSERT_EQ_ALL_ROUNDING(IntType(1), func(FloatType(1.0)));
    ASSERT_EQ_ALL_ROUNDING(IntType(-1), func(FloatType(-1.0)));
    ASSERT_EQ_ALL_ROUNDING(IntType(10), func(FloatType(10.0)));
    ASSERT_EQ_ALL_ROUNDING(IntType(-10), func(FloatType(-10.0)));
    ASSERT_EQ_ALL_ROUNDING(IntType(1232), func(FloatType(1232.0)));
    ASSERT_EQ_ALL_ROUNDING(IntType(-1232), func(FloatType(-1232.0)));

    // The rest of this function compares with an equivalent MPFR function
    // which rounds floating point numbers to long values. There is no MPFR
    // function to round to long long or wider integer values. So, we will
    // the remaining tests only if the width of IntType less than equal to that
    // of long.
    if (sizeof(IntType) > sizeof(long))
      return;

    constexpr int EXPONENT_LIMIT = sizeof(IntType) * 8 - 1;
    constexpr int BIASED_EXPONENT_LIMIT = EXPONENT_LIMIT + FPBits::EXP_BIAS;
    if (BIASED_EXPONENT_LIMIT > FPBits::MAX_BIASED_EXPONENT)
      return;
    // We start with 1.0 so that the implicit bit for x86 long doubles
    // is set.
    FPBits bits(FloatType(1.0));
    bits.set_biased_exponent(BIASED_EXPONENT_LIMIT);
    bits.set_sign(Sign::NEG);
    bits.set_mantissa(0);

    FloatType x = bits.get_val();
    long mpfr_result;
    bool erangeflag = mpfr::round_to_long(x, mpfr_result);
    ASSERT_FALSE(erangeflag);
    ASSERT_EQ_ALL_ROUNDING(IntType(mpfr_result), func(x));
  }

  void testFractions(RoundToIntegerFunc func) {
    constexpr FloatType FRACTIONS[] = {
        FloatType(0.5),    FloatType(-0.5),  FloatType(0.115),
        FloatType(-0.115), FloatType(0.715), FloatType(-0.715),
    };
    if (TestModes) {
      for (auto mpfr_mode : FPTest::ROUNDING_MODES) {
        for (FloatType x : FRACTIONS) {
          long mpfr_long_result;
          bool erangeflag = mpfr::round_to_long(x, mpfr_mode, mpfr_long_result);
          ASSERT_FALSE(erangeflag);
          ASSERT_EQ_ROUNDING_MODE(IntType(mpfr_long_result), func(x),
                                  mpfr_mode);
        }
      }
    } else {
      for (FloatType x : FRACTIONS) {
        long mpfr_long_result;
        bool erangeflag = mpfr::round_to_long(x, mpfr_long_result);
        ASSERT_FALSE(erangeflag);
        test_one_input(func, x, IntType(mpfr_long_result), false);
      }
    }
  }

  void testIntegerOverflow(RoundToIntegerFunc func) {
    // This function compares with an equivalent MPFR function which rounds
    // floating point numbers to long values. There is no MPFR function to
    // round to long long or wider integer values. So, we will peform the
    // comparisons in this function only if the width of IntType less than equal
    // to that of long.
    if (sizeof(IntType) > sizeof(long))
      return;

    constexpr int EXPONENT_LIMIT = sizeof(IntType) * 8 - 1;
    constexpr int BIASED_EXPONENT_LIMIT = EXPONENT_LIMIT + FPBits::EXP_BIAS;
    if (BIASED_EXPONENT_LIMIT > FPBits::MAX_BIASED_EXPONENT)
      return;
    // We start with 1.0 so that the implicit bit for x86 long doubles
    // is set.
    FPBits bits(FloatType(1.0));
    bits.set_biased_exponent(BIASED_EXPONENT_LIMIT);
    bits.set_sign(Sign::NEG);
    bits.set_mantissa(FPBits::FRACTION_MASK);

    FloatType x = bits.get_val();
    if (TestModes) {
      for (auto m : FPTest::ROUNDING_MODES) {
        LIBC_NAMESPACE::fputil::testing::ForceRoundingMode _r(m);
        if (_r.success)
          test_one_input(func, x, INTEGER_MIN, true);
      }
    } else {
      test_one_input(func, x, INTEGER_MIN, true);
    }
  }

  void testSubnormalRange(RoundToIntegerFunc func) {
    constexpr int COUNT = 1'231;
    constexpr StorageType STEP = LIBC_NAMESPACE::cpp::max(
        static_cast<StorageType>((MAX_SUBNORMAL - MIN_SUBNORMAL) / COUNT),
        StorageType(1));
    for (StorageType i = MIN_SUBNORMAL; i <= MAX_SUBNORMAL; i += STEP) {
      FloatType x = FPBits(i).get_val();
      if (x == FloatType(0.0))
        continue;
      // All subnormal numbers should round to zero.
      if (TestModes) {
        if (x > 0) {
          ASSERT_EQ_ROUNDING_UPWARD(IntType(1), func(x));
          ASSERT_EQ_ROUNDING_DOWNWARD(IntType(0), func(x));
          ASSERT_EQ_ROUNDING_TOWARD_ZERO(IntType(0), func(x));
          ASSERT_EQ_ROUNDING_NEAREST(IntType(0), func(x));
        } else {
          ASSERT_EQ_ROUNDING_UPWARD(IntType(0), func(x));
          ASSERT_EQ_ROUNDING_DOWNWARD(IntType(-1), func(x));
          ASSERT_EQ_ROUNDING_TOWARD_ZERO(IntType(0), func(x));
          ASSERT_EQ_ROUNDING_NEAREST(IntType(0), func(x));
        }
      } else {
        test_one_input(func, x, IntType(0), false);
      }
    }
  }

  void testNormalRange(RoundToIntegerFunc func) {
    // This function compares with an equivalent MPFR function which rounds
    // floating point numbers to long values. There is no MPFR function to
    // round to long long or wider integer values. So, we will peform the
    // comparisons in this function only if the width of IntType less than equal
    // to that of long.
    if (sizeof(IntType) > sizeof(long))
      return;

    constexpr int COUNT = 1'231;
    constexpr StorageType STEP = LIBC_NAMESPACE::cpp::max(
        static_cast<StorageType>((MAX_NORMAL - MIN_NORMAL) / COUNT),
        StorageType(1));
    for (StorageType i = MIN_NORMAL; i <= MAX_NORMAL; i += STEP) {
      FPBits xbits(i);
      FloatType x = xbits.get_val();
      // In normal range on x86 platforms, the long double implicit 1 bit can be
      // zero making the numbers NaN. We will skip them.
      if (xbits.is_nan())
        continue;

      if (TestModes) {
        for (auto m : FPTest::ROUNDING_MODES) {
          long mpfr_long_result;
          bool erangeflag = mpfr::round_to_long(x, m, mpfr_long_result);
          LIBC_NAMESPACE::fputil::testing::ForceRoundingMode _r(m);
          if (_r.success) {
            if (erangeflag)
              test_one_input(func, x, x > 0 ? INTEGER_MAX : INTEGER_MIN, true);
            else
              test_one_input(func, x, IntType(mpfr_long_result), false);
          }
        }
      } else {
        long mpfr_long_result;
        bool erangeflag = mpfr::round_to_long(x, mpfr_long_result);
        if (erangeflag)
          test_one_input(func, x, x > 0 ? INTEGER_MAX : INTEGER_MIN, true);
        else
          test_one_input(func, x, IntType(mpfr_long_result), false);
      }
    }
  }
};

#define LIST_ROUND_TO_INTEGER_TESTS_HELPER(FloatType, IntType, func,           \
                                           TestModes)                          \
  using LlvmLibcRoundToIntegerTest =                                           \
      RoundToIntegerTestTemplate<FloatType, IntType, TestModes>;               \
  TEST_F(LlvmLibcRoundToIntegerTest, InfinityAndNaN) {                         \
    testInfinityAndNaN(&func);                                                 \
  }                                                                            \
  TEST_F(LlvmLibcRoundToIntegerTest, RoundNumbers) {                           \
    testRoundNumbers(&func);                                                   \
  }                                                                            \
  TEST_F(LlvmLibcRoundToIntegerTest, Fractions) { testFractions(&func); }      \
  TEST_F(LlvmLibcRoundToIntegerTest, IntegerOverflow) {                        \
    testIntegerOverflow(&func);                                                \
  }                                                                            \
  TEST_F(LlvmLibcRoundToIntegerTest, SubnormalRange) {                         \
    testSubnormalRange(&func);                                                 \
  }                                                                            \
  TEST_F(LlvmLibcRoundToIntegerTest, NormalRange) { testNormalRange(&func); }

#define LIST_ROUND_TO_INTEGER_TESTS(FloatType, IntType, func)                  \
  LIST_ROUND_TO_INTEGER_TESTS_HELPER(FloatType, IntType, func, false)

#define LIST_ROUND_TO_INTEGER_TESTS_WITH_MODES(FloatType, IntType, func)       \
  LIST_ROUND_TO_INTEGER_TESTS_HELPER(FloatType, IntType, func, true)

#endif // LLVM_LIBC_TEST_SRC_MATH_ROUNDTOINTEGERTEST_H
