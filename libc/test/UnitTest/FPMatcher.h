//===-- FPMatchers.h --------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_TEST_UNITTEST_FPMATCHER_H
#define LLVM_LIBC_TEST_UNITTEST_FPMATCHER_H

#include "src/__support/CPP/array.h"
#include "src/__support/CPP/type_traits.h"
#include "src/__support/FPUtil/FEnvImpl.h"
#include "src/__support/FPUtil/FPBits.h"
#include "src/__support/FPUtil/fpbits_str.h"
#include "test/UnitTest/RoundingModeUtils.h"
#include "test/UnitTest/StringUtils.h"
#include "test/UnitTest/Test.h"

#include "hdr/math_macros.h"

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

} // namespace testing
} // namespace LIBC_NAMESPACE

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
      int actual = LIBC_NAMESPACE::libc_errno;                                 \
      LIBC_NAMESPACE::libc_errno = 0;                                          \
      EXPECT_EQ(actual, expected);                                             \
    }                                                                          \
  } while (0)

#define ASSERT_MATH_ERRNO(expected)                                            \
  do {                                                                         \
    if (math_errhandling & MATH_ERRNO) {                                       \
      int actual = LIBC_NAMESPACE::libc_errno;                                 \
      LIBC_NAMESPACE::libc_errno = 0;                                          \
      ASSERT_EQ(actual, expected);                                             \
    }                                                                          \
  } while (0)

#define EXPECT_FP_EXCEPTION_HAPPENED(expected)                                 \
  do {                                                                         \
    if (math_errhandling & MATH_ERREXCEPT) {                                   \
      EXPECT_EQ(LIBC_NAMESPACE::fputil::test_except(FE_ALL_EXCEPT) &           \
                    ((expected) ? (expected) : FE_ALL_EXCEPT),                 \
                (expected));                                                   \
    }                                                                          \
  } while (0)

#define ASSERT_FP_EXCEPTION_HAPPENED(expected)                                 \
  do {                                                                         \
    if (math_errhandling & MATH_ERREXCEPT) {                                   \
      ASSERT_EQ(LIBC_NAMESPACE::fputil::test_except(FE_ALL_EXCEPT) &           \
                    ((expected) ? (expected) : FE_ALL_EXCEPT),                 \
                (expected));                                                   \
    }                                                                          \
  } while (0)

#define EXPECT_FP_EQ_WITH_EXCEPTION(expected_val, actual_val, expected_except) \
  do {                                                                         \
    LIBC_NAMESPACE::fputil::clear_except(FE_ALL_EXCEPT);                       \
    EXPECT_FP_EQ(expected_val, actual_val);                                    \
    EXPECT_FP_EXCEPTION_HAPPENED(expected_except);                             \
  } while (0)

#define EXPECT_FP_IS_NAN_WITH_EXCEPTION(actual_val, expected_except)           \
  do {                                                                         \
    LIBC_NAMESPACE::fputil::clear_except(FE_ALL_EXCEPT);                       \
    EXPECT_FP_IS_NAN(actual_val);                                              \
    EXPECT_FP_EXCEPTION_HAPPENED(expected_except);                             \
  } while (0)

#define FOR_ROUNDING_(rounding_mode, expr_or_statement)                        \
  do {                                                                         \
    using namespace LIBC_NAMESPACE::fputil::testing;                           \
    ForceRoundingMode __r((rounding_mode));                                    \
    if (__r.success) {                                                         \
      expr_or_statement;                                                       \
    }                                                                          \
  } while (0)

#define FOR_ALL_ROUNDING_(expr_or_statement)                                   \
  do {                                                                         \
    using namespace LIBC_NAMESPACE::fputil::testing;                           \
    FOR_ROUNDING_(RoundingMode::Nearest, expr_or_statement);                   \
    FOR_ROUNDING_(RoundingMode::Upward, expr_or_statement);                    \
    FOR_ROUNDING_(RoundingMode::Downward, expr_or_statement);                  \
    FOR_ROUNDING_(RoundingMode::TowardZero, expr_or_statement);                \
  } while (0)

#define EXPECT_FP_EQ_ALL_ROUNDING(expected, actual)                            \
  FOR_ALL_ROUNDING_(EXPECT_FP_EQ((expected), (actual)))

#define EXPECT_FP_EQ_ROUNDING_MODE(expected, actual, rounding_mode)            \
  FOR_ROUNDING_(rounding_mode, EXPECT_FP_EQ((expected), (actual)))

#define EXPECT_FP_EQ_ROUNDING_NEAREST(expected, actual)                        \
  EXPECT_FP_EQ_ROUNDING_MODE((expected), (actual), RoundingMode::Nearest)

#define EXPECT_FP_EQ_ROUNDING_UPWARD(expected, actual)                         \
  EXPECT_FP_EQ_ROUNDING_MODE((expected), (actual), RoundingMode::Upward)

#define EXPECT_FP_EQ_ROUNDING_DOWNWARD(expected, actual)                       \
  EXPECT_FP_EQ_ROUNDING_MODE((expected), (actual), RoundingMode::Downward)

#define EXPECT_FP_EQ_ROUNDING_TOWARD_ZERO(expected, actual)                    \
  EXPECT_FP_EQ_ROUNDING_MODE((expected), (actual), RoundingMode::TowardZero)

#endif // LLVM_LIBC_TEST_UNITTEST_FPMATCHER_H
