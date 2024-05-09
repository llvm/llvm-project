//===-- FPTest.h -----------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===---------------------------------------------------------------------===//

#ifndef LLVM_LIBC_TEST_UNITTEST_FPTEST_H
#define LLVM_LIBC_TEST_UNITTEST_FPTEST_H

#include "src/__support/CPP/utility.h"
#include "src/__support/FPUtil/FEnvImpl.h"
#include "src/__support/FPUtil/FPBits.h"
#include "test/UnitTest/ErrnoSafeTest.h"      // Test fixture for clearing errno
#include "test/UnitTest/ErrnoSetterMatcher.h" // Per-assertion clear/check errno
#include "test/UnitTest/FEnvSafeTest.h"       // Test fixture for resetting fenv
#include "test/UnitTest/FPExceptMatcher.h" // Per-assertion clear/check fp exns
#include "test/UnitTest/FPMatcher.h"       // Matchers/assertions for fp values
#include "test/UnitTest/Test.h"

#define DECLARE_SPECIAL_CONSTANTS(T)                                           \
  using FPBits = LIBC_NAMESPACE::fputil::FPBits<T>;                            \
  using StorageType = typename FPBits::StorageType;                            \
                                                                               \
  static constexpr StorageType STORAGE_MAX =                                   \
      LIBC_NAMESPACE::cpp::numeric_limits<StorageType>::max();                 \
  const T zero = FPBits::zero(Sign::POS).get_val();                            \
  const T neg_zero = FPBits::zero(Sign::NEG).get_val();                        \
  const T aNaN = FPBits::quiet_nan().get_val();                                \
  const T sNaN = FPBits::signaling_nan().get_val();                            \
  const T inf = FPBits::inf(Sign::POS).get_val();                              \
  const T neg_inf = FPBits::inf(Sign::NEG).get_val();                          \
  const T min_normal = FPBits::min_normal().get_val();                         \
  const T max_normal = FPBits::max_normal(Sign::POS).get_val();                \
  const T neg_max_normal = FPBits::max_normal(Sign::NEG).get_val();            \
  const T min_denormal = FPBits::min_subnormal(Sign::POS).get_val();           \
  const T neg_min_denormal = FPBits::min_subnormal(Sign::NEG).get_val();       \
  const T max_denormal = FPBits::max_subnormal().get_val();                    \
  static constexpr int UNKNOWN_MATH_ROUNDING_DIRECTION = 99;                   \
  static constexpr LIBC_NAMESPACE::cpp::array<int, 6>                          \
      MATH_ROUNDING_DIRECTIONS_INCLUDING_UNKNOWN = {                           \
          FP_INT_UPWARD,     FP_INT_DOWNWARD,                                  \
          FP_INT_TOWARDZERO, FP_INT_TONEARESTFROMZERO,                         \
          FP_INT_TONEAREST,  UNKNOWN_MATH_ROUNDING_DIRECTION,                  \
  };

namespace LIBC_NAMESPACE::testing {

template <typename T>
struct FPTest : public ErrnoSafeTest, public FEnvSafeTest {
  using FPBits = LIBC_NAMESPACE::fputil::FPBits<T>;
  using StorageType = typename FPBits::StorageType;
  static constexpr StorageType STORAGE_MAX =
      LIBC_NAMESPACE::cpp::numeric_limits<StorageType>::max();
  static constexpr T zero = FPBits::zero(Sign::POS).get_val();
  static constexpr T neg_zero = FPBits::zero(Sign::NEG).get_val();
  static constexpr T aNaN = FPBits::quiet_nan().get_val();
  static constexpr T sNaN = FPBits::signaling_nan().get_val();
  static constexpr T inf = FPBits::inf(Sign::POS).get_val();
  static constexpr T neg_inf = FPBits::inf(Sign::NEG).get_val();
  static constexpr T min_normal = FPBits::min_normal().get_val();
  static constexpr T max_normal = FPBits::max_normal().get_val();
  static constexpr T min_denormal = FPBits::min_subnormal().get_val();
  static constexpr T max_denormal = FPBits::max_subnormal().get_val();

  static constexpr int N_ROUNDING_MODES = 4;
  static constexpr fputil::testing::RoundingMode ROUNDING_MODES[4] = {
      fputil::testing::RoundingMode::Nearest,
      fputil::testing::RoundingMode::Upward,
      fputil::testing::RoundingMode::Downward,
      fputil::testing::RoundingMode::TowardZero,
  };

  void TearDown() override { FEnvSafeTest::TearDown(); }

  void SetUp() override { ErrnoSafeTest::SetUp(); }
};

} // namespace LIBC_NAMESPACE::testing

// Does not return the value of `expr_or_statement`, i.e., intended usage
// is: `EXPECT_ERRNO_FPEXC(EDOM, FE_INVALID, EXPECT_FP_EQ(..., ...));` or
// ```
// EXPECT_ERRNO_FPEXC(EDOM, FE_INVALID, {
//   stmt;
//   ...
// });
// ```
// Ensures that fp excepts and errno are cleared before executing
// `expr_or_statement` Checking (expected_fexn = 0) ensures that no exceptions
// were set
#define EXPECT_ERRNO_FP_EXCEPT(expected_errno, expected_fexn,                  \
                               expr_or_statement)                              \
  EXPECT_FP_EXCEPT((expected_fexn), EXPECT_ERRNO((expected_errno), {           \
                     expr_or_statement;                                        \
                     if (!(math_errhandling & MATH_ERRNO))                     \
                       break;                                                  \
                   }))

#define EXPECT_NO_ERRNO_FP_EXCEPT(expr_or_statement)                           \
  EXPECT_ERRNO_FP_EXCEPT(0, 0, expr_or_statement)

#define EXPECT_ERRNO_FP_EXCEPT_ALL_ROUNDING(expected_errno, expected_fexn,     \
                                            expr_or_statement)                 \
  FOR_ALL_ROUNDING_(EXPECT_ERRNO_FP_EXCEPT((expected_errno), (expected_fexn),  \
                                           expr_or_statement))

#define EXPECT_NO_ERRNO_FP_EXCEPT_ALL_ROUNDING(expr_or_statement)              \
  FOR_ALL_ROUNDING_(EXPECT_NO_ERRNO_FP_EXCEPT(expr_or_statement))

#endif // LLVM_LIBC_TEST_UNITTEST_FPTEST_H
