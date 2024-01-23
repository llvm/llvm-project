//===-- FPMatchers.h --------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_UTILS_UNITTEST_FPMATCHER_H
#define LLVM_LIBC_UTILS_UNITTEST_FPMATCHER_H

#include "src/__support/CPP/type_traits.h"
#include "src/__support/FPUtil/FEnvImpl.h"
#include "src/__support/FPUtil/FPBits.h"
#include "src/__support/FPUtil/fpbits_str.h"
#include "test/UnitTest/RoundingModeUtils.h"
#include "test/UnitTest/StringUtils.h"
#include "test/UnitTest/Test.h"

#include <math.h>

namespace LIBC_NAMESPACE {
namespace testing {

template <typename T, TestCond Condition> class FPMatcher : public Matcher<T> {
  static_assert(cpp::is_floating_point_v<T>,
                "FPMatcher can only be used with floating point values.");
  static_assert(Condition == TestCond::EQ || Condition == TestCond::NE,
                "Unsupported FPMatcher test condition.");

  T expected;
  T actual;

public:
  FPMatcher(T expectedValue) : expected(expectedValue) {}

  bool match(T actualValue) {
    actual = actualValue;
    fputil::FPBits<T> actualBits(actual), expectedBits(expected);
    if (Condition == TestCond::EQ)
      return (actualBits.is_nan() && expectedBits.is_nan()) ||
             (actualBits.uintval() == expectedBits.uintval());

    // If condition == TestCond::NE.
    if (actualBits.is_nan())
      return !expectedBits.is_nan();
    return expectedBits.is_nan() ||
           (actualBits.uintval() != expectedBits.uintval());
  }

  void explainError() override {
    tlog << "Expected floating point value: "
         << str(fputil::FPBits<T>(expected)) << '\n';
    tlog << "Actual floating point value: " << str(fputil::FPBits<T>(actual))
         << '\n';
  }
};

template <TestCond C, typename T> FPMatcher<T, C> getMatcher(T expectedValue) {
  return FPMatcher<T, C>(expectedValue);
}

template <typename T> struct FPTest : public Test {
  using FPBits = LIBC_NAMESPACE::fputil::FPBits<T>;
  using StorageType = typename FPBits::StorageType;
  using Sign = LIBC_NAMESPACE::fputil::Sign;
  static constexpr StorageType STORAGE_MAX =
      LIBC_NAMESPACE::cpp::numeric_limits<StorageType>::max();
  static constexpr T zero = T(FPBits::zero(Sign::POS));
  static constexpr T neg_zero = T(FPBits::zero(Sign::NEG));
  static constexpr T aNaN = T(FPBits::build_quiet_nan());
  static constexpr T sNaN = T(FPBits::build_nan(Sign::POS, 1));
  static constexpr T inf = T(FPBits::inf(Sign::POS));
  static constexpr T neg_inf = T(FPBits::inf(Sign::NEG));
  static constexpr T min_normal = T(FPBits::min_normal());
  static constexpr T max_normal = T(FPBits::max_normal());
  static constexpr T min_denormal = T(FPBits::min_subnormal());
  static constexpr T max_denormal = T(FPBits::max_subnormal());

  static constexpr int N_ROUNDING_MODES = 4;
  static constexpr fputil::testing::RoundingMode ROUNDING_MODES[4] = {
      fputil::testing::RoundingMode::Nearest,
      fputil::testing::RoundingMode::Upward,
      fputil::testing::RoundingMode::Downward,
      fputil::testing::RoundingMode::TowardZero,
  };
};

} // namespace testing
} // namespace LIBC_NAMESPACE

#define DECLARE_SPECIAL_CONSTANTS(T)                                           \
  using FPBits = LIBC_NAMESPACE::fputil::FPBits<T>;                            \
  using StorageType = typename FPBits::StorageType;                            \
  using Sign = LIBC_NAMESPACE::fputil::Sign;                                   \
  static constexpr StorageType STORAGE_MAX =                                   \
      LIBC_NAMESPACE::cpp::numeric_limits<StorageType>::max();                 \
  const T zero = T(FPBits::zero(Sign::POS));                                   \
  const T neg_zero = T(FPBits::zero(Sign::NEG));                               \
  const T aNaN = T(FPBits::build_quiet_nan());                                 \
  const T sNaN = T(FPBits::build_nan(Sign::POS, 1));                           \
  const T inf = T(FPBits::inf(Sign::POS));                                     \
  const T neg_inf = T(FPBits::inf(Sign::NEG));                                 \
  const T min_normal = T(FPBits::min_normal());                                \
  const T max_normal = T(FPBits::max_normal());                                \
  const T min_denormal = T(FPBits::min_subnormal());                           \
  const T max_denormal = T(FPBits::max_subnormal());

#define EXPECT_FP_EQ(expected, actual)                                         \
  EXPECT_THAT(actual, LIBC_NAMESPACE::testing::getMatcher<                     \
                          LIBC_NAMESPACE::testing::TestCond::EQ>(expected))

#define TEST_FP_EQ(expected, actual)                                           \
  LIBC_NAMESPACE::testing::getMatcher<LIBC_NAMESPACE::testing::TestCond::EQ>(  \
      expected)                                                                \
      .match(actual)

#define EXPECT_FP_IS_NAN(actual) EXPECT_TRUE((actual) != (actual))

#define ASSERT_FP_EQ(expected, actual)                                         \
  ASSERT_THAT(actual, LIBC_NAMESPACE::testing::getMatcher<                     \
                          LIBC_NAMESPACE::testing::TestCond::EQ>(expected))

#define EXPECT_FP_NE(expected, actual)                                         \
  EXPECT_THAT(actual, LIBC_NAMESPACE::testing::getMatcher<                     \
                          LIBC_NAMESPACE::testing::TestCond::NE>(expected))

#define ASSERT_FP_NE(expected, actual)                                         \
  ASSERT_THAT(actual, LIBC_NAMESPACE::testing::getMatcher<                     \
                          LIBC_NAMESPACE::testing::TestCond::NE>(expected))

#define EXPECT_MATH_ERRNO(expected)                                            \
  do {                                                                         \
    if (math_errhandling & MATH_ERRNO) {                                       \
      int actual = libc_errno;                                                 \
      libc_errno = 0;                                                          \
      EXPECT_EQ(actual, expected);                                             \
    }                                                                          \
  } while (0)

#define ASSERT_MATH_ERRNO(expected)                                            \
  do {                                                                         \
    if (math_errhandling & MATH_ERRNO) {                                       \
      int actual = libc_errno;                                                 \
      libc_errno = 0;                                                          \
      ASSERT_EQ(actual, expected);                                             \
    }                                                                          \
  } while (0)

#define EXPECT_FP_EXCEPTION(expected)                                          \
  do {                                                                         \
    if (math_errhandling & MATH_ERREXCEPT) {                                   \
      EXPECT_GE(LIBC_NAMESPACE::fputil::test_except(FE_ALL_EXCEPT) &           \
                    (expected),                                                \
                expected);                                                     \
    }                                                                          \
  } while (0)

#define ASSERT_FP_EXCEPTION(expected)                                          \
  do {                                                                         \
    if (math_errhandling & MATH_ERREXCEPT) {                                   \
      ASSERT_GE(LIBC_NAMESPACE::fputil::test_except(FE_ALL_EXCEPT) &           \
                    (expected),                                                \
                expected);                                                     \
    }                                                                          \
  } while (0)

#define EXPECT_FP_EQ_WITH_EXCEPTION(expected_val, actual_val, expected_except) \
  do {                                                                         \
    LIBC_NAMESPACE::fputil::clear_except(FE_ALL_EXCEPT);                       \
    EXPECT_FP_EQ(expected_val, actual_val);                                    \
    if (math_errhandling & MATH_ERREXCEPT) {                                   \
      EXPECT_GE(LIBC_NAMESPACE::fputil::test_except(FE_ALL_EXCEPT) &           \
                    (expected_except),                                         \
                expected_except);                                              \
      LIBC_NAMESPACE::fputil::clear_except(FE_ALL_EXCEPT);                     \
    }                                                                          \
  } while (0)

#define EXPECT_FP_IS_NAN_WITH_EXCEPTION(actual_val, expected_except)           \
  do {                                                                         \
    LIBC_NAMESPACE::fputil::clear_except(FE_ALL_EXCEPT);                       \
    EXPECT_FP_IS_NAN(actual_val);                                              \
    if (math_errhandling & MATH_ERREXCEPT) {                                   \
      EXPECT_GE(LIBC_NAMESPACE::fputil::test_except(FE_ALL_EXCEPT) &           \
                    (expected_except),                                         \
                expected_except);                                              \
      LIBC_NAMESPACE::fputil::clear_except(FE_ALL_EXCEPT);                     \
    }                                                                          \
  } while (0)

#define EXPECT_FP_EQ_ALL_ROUNDING(expected, actual)                            \
  do {                                                                         \
    using namespace LIBC_NAMESPACE::fputil::testing;                           \
    ForceRoundingMode __r1(RoundingMode::Nearest);                             \
    if (__r1.success) {                                                        \
      EXPECT_FP_EQ((expected), (actual));                                      \
    }                                                                          \
    ForceRoundingMode __r2(RoundingMode::Upward);                              \
    if (__r2.success) {                                                        \
      EXPECT_FP_EQ((expected), (actual));                                      \
    }                                                                          \
    ForceRoundingMode __r3(RoundingMode::Downward);                            \
    if (__r3.success) {                                                        \
      EXPECT_FP_EQ((expected), (actual));                                      \
    }                                                                          \
    ForceRoundingMode __r4(RoundingMode::TowardZero);                          \
    if (__r4.success) {                                                        \
      EXPECT_FP_EQ((expected), (actual));                                      \
    }                                                                          \
  } while (0)

#endif // LLVM_LIBC_UTILS_UNITTEST_FPMATCHER_H
