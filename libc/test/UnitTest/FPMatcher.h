//===-- FPMatchers.h --------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_UTILS_UNITTEST_FPMATCHER_H
#define LLVM_LIBC_UTILS_UNITTEST_FPMATCHER_H

#include "src/__support/FPUtil/FEnvImpl.h"
#include "src/__support/FPUtil/FPBits.h"
#include "test/UnitTest/RoundingModeUtils.h"
#include "test/UnitTest/StringUtils.h"
#include "test/UnitTest/Test.h"

#include <math.h>

namespace __llvm_libc {
namespace fputil {
namespace testing {

template <typename ValType>
cpp::enable_if_t<cpp::is_floating_point_v<ValType>, void>
describeValue(const char *label, ValType value) {
  __llvm_libc::testing::tlog << label;

  FPBits<ValType> bits(value);
  if (bits.is_nan()) {
    __llvm_libc::testing::tlog << "(NaN)";
  } else if (bits.is_inf()) {
    if (bits.get_sign())
      __llvm_libc::testing::tlog << "(-Infinity)";
    else
      __llvm_libc::testing::tlog << "(+Infinity)";
  } else {
    constexpr int exponentWidthInHex =
        (fputil::ExponentWidth<ValType>::VALUE - 1) / 4 + 1;
    constexpr int mantissaWidthInHex =
        (fputil::MantissaWidth<ValType>::VALUE - 1) / 4 + 1;
    constexpr int bitsWidthInHex =
        sizeof(typename fputil::FPBits<ValType>::UIntType) * 2;

    __llvm_libc::testing::tlog
        << "0x"
        << int_to_hex<typename fputil::FPBits<ValType>::UIntType>(
               bits.uintval(), bitsWidthInHex)
        << ", (S | E | M) = (" << (bits.get_sign() ? '1' : '0') << " | 0x"
        << int_to_hex<uint16_t>(bits.get_unbiased_exponent(),
                                exponentWidthInHex)
        << " | 0x"
        << int_to_hex<typename fputil::FPBits<ValType>::UIntType>(
               bits.get_mantissa(), mantissaWidthInHex)
        << ")";
  }

  __llvm_libc::testing::tlog << '\n';
}

template <typename T, __llvm_libc::testing::TestCondition Condition>
class FPMatcher : public __llvm_libc::testing::Matcher<T> {
  static_assert(__llvm_libc::cpp::is_floating_point_v<T>,
                "FPMatcher can only be used with floating point values.");
  static_assert(Condition == __llvm_libc::testing::Cond_EQ ||
                    Condition == __llvm_libc::testing::Cond_NE,
                "Unsupported FPMathcer test condition.");

  T expected;
  T actual;

public:
  FPMatcher(T expectedValue) : expected(expectedValue) {}

  bool match(T actualValue) {
    actual = actualValue;
    fputil::FPBits<T> actualBits(actual), expectedBits(expected);
    if (Condition == __llvm_libc::testing::Cond_EQ)
      return (actualBits.is_nan() && expectedBits.is_nan()) ||
             (actualBits.uintval() == expectedBits.uintval());

    // If condition == Cond_NE.
    if (actualBits.is_nan())
      return !expectedBits.is_nan();
    return expectedBits.is_nan() ||
           (actualBits.uintval() != expectedBits.uintval());
  }

  void explainError() override {
    describeValue("Expected floating point value: ", expected);
    describeValue("  Actual floating point value: ", actual);
  }
};

template <__llvm_libc::testing::TestCondition C, typename T>
FPMatcher<T, C> getMatcher(T expectedValue) {
  return FPMatcher<T, C>(expectedValue);
}

} // namespace testing
} // namespace fputil
} // namespace __llvm_libc

#define DECLARE_SPECIAL_CONSTANTS(T)                                           \
  using FPBits = __llvm_libc::fputil::FPBits<T>;                               \
  using UIntType = typename FPBits::UIntType;                                  \
  const T zero = T(FPBits::zero());                                            \
  const T neg_zero = T(FPBits::neg_zero());                                    \
  const T aNaN = T(FPBits::build_quiet_nan(1));                                \
  const T inf = T(FPBits::inf());                                              \
  const T neg_inf = T(FPBits::neg_inf());

#define EXPECT_FP_EQ(expected, actual)                                         \
  EXPECT_THAT(                                                                 \
      actual,                                                                  \
      __llvm_libc::fputil::testing::getMatcher<__llvm_libc::testing::Cond_EQ>( \
          expected))

#define EXPECT_FP_IS_NAN(actual) EXPECT_TRUE((actual) != (actual))

#define ASSERT_FP_EQ(expected, actual)                                         \
  ASSERT_THAT(                                                                 \
      actual,                                                                  \
      __llvm_libc::fputil::testing::getMatcher<__llvm_libc::testing::Cond_EQ>( \
          expected))

#define EXPECT_FP_NE(expected, actual)                                         \
  EXPECT_THAT(                                                                 \
      actual,                                                                  \
      __llvm_libc::fputil::testing::getMatcher<__llvm_libc::testing::Cond_NE>( \
          expected))

#define ASSERT_FP_NE(expected, actual)                                         \
  ASSERT_THAT(                                                                 \
      actual,                                                                  \
      __llvm_libc::fputil::testing::getMatcher<__llvm_libc::testing::Cond_NE>( \
          expected))

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
      EXPECT_GE(__llvm_libc::fputil::test_except(FE_ALL_EXCEPT) & expected,    \
                expected);                                                     \
    }                                                                          \
  } while (0)

#define ASSERT_FP_EXCEPTION(expected)                                          \
  do {                                                                         \
    if (math_errhandling & MATH_ERREXCEPT) {                                   \
      ASSERT_GE(__llvm_libc::fputil::test_except(FE_ALL_EXCEPT) & expected,    \
                expected);                                                     \
    }                                                                          \
  } while (0)

#define EXPECT_FP_EQ_WITH_EXCEPTION(expected_val, actual_val, expected_except) \
  do {                                                                         \
    __llvm_libc::fputil::clear_except(FE_ALL_EXCEPT);                          \
    EXPECT_FP_EQ(expected_val, actual_val);                                    \
    if (math_errhandling & MATH_ERREXCEPT) {                                   \
      EXPECT_GE(__llvm_libc::fputil::test_except(FE_ALL_EXCEPT) &              \
                    expected_except,                                           \
                expected_except);                                              \
      __llvm_libc::fputil::clear_except(FE_ALL_EXCEPT);                        \
    }                                                                          \
  } while (0)

#define EXPECT_FP_IS_NAN_WITH_EXCEPTION(actual_val, expected_except)           \
  do {                                                                         \
    __llvm_libc::fputil::clear_except(FE_ALL_EXCEPT);                          \
    EXPECT_FP_IS_NAN(actual_val);                                              \
    if (math_errhandling & MATH_ERREXCEPT) {                                   \
      EXPECT_GE(__llvm_libc::fputil::test_except(FE_ALL_EXCEPT) &              \
                    expected_except,                                           \
                expected_except);                                              \
      __llvm_libc::fputil::clear_except(FE_ALL_EXCEPT);                        \
    }                                                                          \
  } while (0)

#define EXPECT_FP_EQ_ALL_ROUNDING(expected, actual)                            \
  do {                                                                         \
    using namespace __llvm_libc::fputil::testing;                              \
    ForceRoundingMode __r1(RoundingMode::Nearest);                             \
    EXPECT_FP_EQ((expected), (actual));                                        \
    ForceRoundingMode __r2(RoundingMode::Upward);                              \
    EXPECT_FP_EQ((expected), (actual));                                        \
    ForceRoundingMode __r3(RoundingMode::Downward);                            \
    EXPECT_FP_EQ((expected), (actual));                                        \
    ForceRoundingMode __r4(RoundingMode::TowardZero);                          \
    EXPECT_FP_EQ((expected), (actual));                                        \
  } while (0)

#endif // LLVM_LIBC_UTILS_UNITTEST_FPMATCHER_H
