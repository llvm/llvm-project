//===-- Utility class to test different flavors of [l|ll]round --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_TEST_SRC_MATH_SMOKE_ROUNDTOINTEGERTEST_H
#define LLVM_LIBC_TEST_SRC_MATH_SMOKE_ROUNDTOINTEGERTEST_H

#include "test/UnitTest/RoundingModeUtils.h"
#undef LIBC_MATH_USE_SYSTEM_FENV

#include "src/__support/CPP/algorithm.h"
#include "src/__support/FPUtil/FEnvImpl.h"
#include "src/__support/FPUtil/FPBits.h"
#include "src/__support/libc_errno.h"
#include "test/UnitTest/FEnvSafeTest.h"
#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"

#include "hdr/math_macros.h"

template <typename FloatType, typename IntType, bool TestModes = false>
class RoundToIntegerTestTemplate
    : public LIBC_NAMESPACE::testing::FEnvSafeTest {
public:
  typedef IntType (*RoundToIntegerFunc)(FloatType);

private:
  DECLARE_SPECIAL_CONSTANTS(FloatType)
  using FPTest = LIBC_NAMESPACE::testing::FPTest<FloatType>;
  using RoundingMode = LIBC_NAMESPACE::fputil::testing::RoundingMode;

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
    ASSERT_EQ_ALL_ROUNDING_1(INTEGER_MAX, func(inf));
    ASSERT_EQ_ALL_ROUNDING_1(INTEGER_MIN, func(neg_inf));
    ASSERT_FP_EXCEPTION(FE_INVALID);
    ASSERT_MATH_ERRNO(EDOM);
    // This is currently never enabled, the
    // LLVM_LIBC_IMPLEMENTATION_DEFINED_TEST_BEHAVIOR CMake option in
    // libc/CMakeLists.txt is not forwarded to C++.
#if LIBC_COPT_IMPLEMENTATION_DEFINED_TEST_BEHAVIOR
    // Result is not well-defined, we always returns INTEGER_MAX
    test_one_input(func, aNaN, INTEGER_MAX, true);
#endif // LIBC_COPT_IMPLEMENTATION_DEFINED_TEST_BEHAVIOR
  }

  void testRoundNumbers(RoundToIntegerFunc func) {
    ASSERT_EQ_ALL_ROUNDING_1(IntType(0), func(zero));
    ASSERT_EQ_ALL_ROUNDING_1(IntType(0), func(neg_zero));
    ASSERT_EQ_ALL_ROUNDING_1(IntType(1), func(FloatType(1.0)));
    ASSERT_EQ_ALL_ROUNDING_1(IntType(-1), func(FloatType(-1.0)));
    ASSERT_EQ_ALL_ROUNDING_1(IntType(10), func(FloatType(10.0)));
    ASSERT_EQ_ALL_ROUNDING_1(IntType(-10), func(FloatType(-10.0)));
    ASSERT_EQ_ALL_ROUNDING_1(IntType(1232), func(FloatType(1232.0)));
    ASSERT_EQ_ALL_ROUNDING_1(IntType(-1232), func(FloatType(-1232.0)));
  }

  void testSubnormalRange(RoundToIntegerFunc func) {
    // Arbitrary, trades off completeness with testing time (esp. on failure)
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
  TEST_F(LlvmLibcRoundToIntegerTest, SubnormalRange) {                         \
    testSubnormalRange(&func);                                                 \
  }

#define LIST_ROUND_TO_INTEGER_TESTS(FloatType, IntType, func)                  \
  LIST_ROUND_TO_INTEGER_TESTS_HELPER(FloatType, IntType, func, false)

// The GPU target does not support different rounding modes.
#ifdef LIBC_TARGET_ARCH_IS_GPU
#define LIST_ROUND_TO_INTEGER_TESTS_WITH_MODES(FloatType, IntType, func)       \
  LIST_ROUND_TO_INTEGER_TESTS_HELPER(FloatType, IntType, func, false)
#else
#define LIST_ROUND_TO_INTEGER_TESTS_WITH_MODES(FloatType, IntType, func)       \
  LIST_ROUND_TO_INTEGER_TESTS_HELPER(FloatType, IntType, func, true)
#endif

#endif // LLVM_LIBC_TEST_SRC_MATH_SMOKE_ROUNDTOINTEGERTEST_H
