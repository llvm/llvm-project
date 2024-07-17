//===-- Utility class to test different flavors of float div --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_TEST_SRC_MATH_SMOKE_DIVTEST_H
#define LLVM_LIBC_TEST_SRC_MATH_SMOKE_DIVTEST_H

#include "hdr/fenv_macros.h"
#include "src/__support/FPUtil/BasicOperations.h"
#include "test/UnitTest/FEnvSafeTest.h"
#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/RoundingModeUtils.h"
#include "test/UnitTest/Test.h"

template <typename OutType, typename InType>
class DivTest : public LIBC_NAMESPACE::testing::FEnvSafeTest {

  DECLARE_SPECIAL_CONSTANTS(OutType)

  struct InConstants {
    DECLARE_SPECIAL_CONSTANTS(InType)
  };

  using InFPBits = typename InConstants::FPBits;
  using InStorageType = typename InConstants::StorageType;

public:
  using DivFunc = OutType (*)(InType, InType);

  void test_special_numbers(DivFunc func) {
    EXPECT_FP_IS_NAN(func(aNaN, aNaN));
    EXPECT_FP_IS_NAN_WITH_EXCEPTION(func(sNaN, sNaN), FE_INVALID);

    InType qnan_42 = InFPBits::quiet_nan(Sign::POS, 0x42).get_val();
    EXPECT_FP_EQ(InType(0x42.0p+0),
                 LIBC_NAMESPACE::fputil::getpayload(func(qnan_42, zero)));
    EXPECT_FP_EQ(InType(0x42.0p+0),
                 LIBC_NAMESPACE::fputil::getpayload(func(zero, qnan_42)));

    if constexpr (sizeof(OutType) < sizeof(InType)) {
      InStorageType max_payload = InFPBits::FRACTION_MASK >> 1;
      InType qnan_max = InFPBits::quiet_nan(Sign::POS, max_payload).get_val();
      EXPECT_FP_EQ(zero,
                   LIBC_NAMESPACE::fputil::getpayload(func(qnan_max, zero)));
      EXPECT_FP_EQ(zero,
                   LIBC_NAMESPACE::fputil::getpayload(func(zero, qnan_max)));
      EXPECT_FP_EQ(InType(0x42.0p+0),
                   LIBC_NAMESPACE::fputil::getpayload(func(qnan_max, qnan_42)));
      EXPECT_FP_EQ(InType(0x42.0p+0),
                   LIBC_NAMESPACE::fputil::getpayload(func(qnan_42, qnan_max)));
    }

    EXPECT_FP_EQ(inf, func(inf, zero));
    EXPECT_FP_EQ(neg_inf, func(neg_inf, zero));
    EXPECT_FP_EQ(neg_inf, func(inf, neg_zero));
    EXPECT_FP_EQ(inf, func(neg_inf, neg_zero));
  }

  void test_division_by_zero(DivFunc func) {
    EXPECT_FP_EQ_WITH_EXCEPTION(inf, func(InType(1.0), zero), FE_DIVBYZERO);
    EXPECT_FP_EQ_WITH_EXCEPTION(neg_inf, func(InType(-1.0), zero),
                                FE_DIVBYZERO);
    EXPECT_FP_EQ_WITH_EXCEPTION(neg_inf, func(InType(1.0), neg_zero),
                                FE_DIVBYZERO);
    EXPECT_FP_EQ_WITH_EXCEPTION(inf, func(InType(1.0), zero), FE_DIVBYZERO);
  }

  void test_invalid_operations(DivFunc func) {
    EXPECT_FP_IS_NAN_WITH_EXCEPTION(func(zero, zero), FE_INVALID);
    EXPECT_FP_IS_NAN_WITH_EXCEPTION(func(neg_zero, zero), FE_INVALID);
    EXPECT_FP_IS_NAN_WITH_EXCEPTION(func(zero, neg_zero), FE_INVALID);
    EXPECT_FP_IS_NAN_WITH_EXCEPTION(func(neg_zero, neg_zero), FE_INVALID);

    EXPECT_FP_IS_NAN_WITH_EXCEPTION(func(inf, inf), FE_INVALID);
    EXPECT_MATH_ERRNO(EDOM);
    EXPECT_FP_IS_NAN_WITH_EXCEPTION(func(neg_inf, inf), FE_INVALID);
    EXPECT_MATH_ERRNO(EDOM);
    EXPECT_FP_IS_NAN_WITH_EXCEPTION(func(inf, neg_inf), FE_INVALID);
    EXPECT_MATH_ERRNO(EDOM);
    EXPECT_FP_IS_NAN_WITH_EXCEPTION(func(neg_inf, neg_inf), FE_INVALID);
    EXPECT_MATH_ERRNO(EDOM);
  }

  void test_range_errors(DivFunc func) {
    using namespace LIBC_NAMESPACE::fputil::testing;

    if (ForceRoundingMode r(RoundingMode::Nearest); r.success) {
      EXPECT_FP_EQ_WITH_EXCEPTION(inf, func(max_normal, min_normal),
                                  FE_OVERFLOW | FE_INEXACT);
      EXPECT_MATH_ERRNO(ERANGE);
      EXPECT_FP_EQ_WITH_EXCEPTION(-inf, func(neg_max_normal, min_denormal),
                                  FE_OVERFLOW | FE_INEXACT);
      EXPECT_MATH_ERRNO(ERANGE);

      EXPECT_FP_EQ_WITH_EXCEPTION(zero, func(min_denormal, max_normal),
                                  FE_UNDERFLOW | FE_INEXACT);
      EXPECT_MATH_ERRNO(ERANGE);
      EXPECT_FP_EQ_WITH_EXCEPTION(neg_zero, func(neg_min_denormal, max_normal),
                                  FE_UNDERFLOW | FE_INEXACT);
      EXPECT_MATH_ERRNO(ERANGE);
    }

    if (ForceRoundingMode r(RoundingMode::TowardZero); r.success) {
      EXPECT_FP_EQ_WITH_EXCEPTION(max_normal, func(max_normal, min_normal),
                                  FE_OVERFLOW | FE_INEXACT);
      EXPECT_FP_EQ_WITH_EXCEPTION(neg_max_normal,
                                  func(neg_max_normal, min_denormal),
                                  FE_OVERFLOW | FE_INEXACT);

      EXPECT_FP_EQ_WITH_EXCEPTION(zero, func(min_denormal, max_normal),
                                  FE_UNDERFLOW | FE_INEXACT);
      EXPECT_MATH_ERRNO(ERANGE);
      EXPECT_FP_EQ_WITH_EXCEPTION(neg_zero, func(neg_min_denormal, max_normal),
                                  FE_UNDERFLOW | FE_INEXACT);
      EXPECT_MATH_ERRNO(ERANGE);
    }

    if (ForceRoundingMode r(RoundingMode::Downward); r.success) {
      EXPECT_FP_EQ_WITH_EXCEPTION(max_normal, func(max_normal, min_normal),
                                  FE_OVERFLOW | FE_INEXACT);
      EXPECT_FP_EQ_WITH_EXCEPTION(-inf, func(neg_max_normal, min_denormal),
                                  FE_OVERFLOW | FE_INEXACT);
      EXPECT_MATH_ERRNO(ERANGE);

      EXPECT_FP_EQ_WITH_EXCEPTION(zero, func(min_denormal, max_normal),
                                  FE_UNDERFLOW | FE_INEXACT);
      EXPECT_MATH_ERRNO(ERANGE);
      EXPECT_FP_EQ_WITH_EXCEPTION(neg_min_denormal,
                                  func(neg_min_denormal, max_normal),
                                  FE_UNDERFLOW | FE_INEXACT);
      EXPECT_MATH_ERRNO(ERANGE);
    }

    if (ForceRoundingMode r(RoundingMode::Upward); r.success) {
      EXPECT_FP_EQ_WITH_EXCEPTION(inf, func(max_normal, min_normal),
                                  FE_OVERFLOW | FE_INEXACT);
      EXPECT_MATH_ERRNO(ERANGE);
      EXPECT_FP_EQ_WITH_EXCEPTION(neg_max_normal,
                                  func(neg_max_normal, min_denormal),
                                  FE_OVERFLOW | FE_INEXACT);

      EXPECT_FP_EQ_WITH_EXCEPTION(min_denormal, func(min_denormal, max_normal),
                                  FE_UNDERFLOW | FE_INEXACT);
      EXPECT_MATH_ERRNO(ERANGE);
      EXPECT_FP_EQ_WITH_EXCEPTION(neg_zero, func(neg_min_denormal, max_normal),
                                  FE_UNDERFLOW | FE_INEXACT);
      EXPECT_MATH_ERRNO(ERANGE);
    }
  }

  void test_inexact_results(DivFunc func) {
    func(InType(1.0), InType(3.0));
    EXPECT_FP_EXCEPTION(FE_INEXACT);
  }
};

#define LIST_DIV_TESTS(OutType, InType, func)                                  \
  using LlvmLibcDivTest = DivTest<OutType, InType>;                            \
  TEST_F(LlvmLibcDivTest, SpecialNumbers) { test_special_numbers(&func); }     \
  TEST_F(LlvmLibcDivTest, DivisionByZero) { test_division_by_zero(&func); }    \
  TEST_F(LlvmLibcDivTest, InvalidOperations) {                                 \
    test_invalid_operations(&func);                                            \
  }                                                                            \
  TEST_F(LlvmLibcDivTest, RangeErrors) { test_range_errors(&func); }           \
  TEST_F(LlvmLibcDivTest, InexactResults) { test_inexact_results(&func); }

#endif // LLVM_LIBC_TEST_SRC_MATH_SMOKE_DIVTEST_H
