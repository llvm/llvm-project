//===-- Utility class to test different flavors of float sub ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_TEST_SRC_MATH_SMOKE_SUBTEST_H
#define LLVM_LIBC_TEST_SRC_MATH_SMOKE_SUBTEST_H

#include "hdr/fenv_macros.h"
#include "src/__support/FPUtil/BasicOperations.h"
#include "test/UnitTest/FEnvSafeTest.h"
#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"

template <typename OutType, typename InType>
class SubTest : public LIBC_NAMESPACE::testing::FEnvSafeTest {

  DECLARE_SPECIAL_CONSTANTS(OutType)

  struct InConstants {
    DECLARE_SPECIAL_CONSTANTS(InType)
  };

  using InFPBits = typename InConstants::FPBits;
  using InStorageType = typename InConstants::StorageType;

  InConstants in;

public:
  using SubFunc = OutType (*)(InType, InType);

  void test_special_numbers(SubFunc func) {
    EXPECT_FP_IS_NAN(func(aNaN, aNaN));
    EXPECT_FP_IS_NAN_WITH_EXCEPTION(func(sNaN, sNaN), FE_INVALID);

    InType qnan_42 = InFPBits::quiet_nan(Sign::POS, 0x42).get_val();
    EXPECT_FP_IS_NAN(func(qnan_42, zero));
    EXPECT_FP_IS_NAN(func(zero, qnan_42));

    EXPECT_FP_EQ(inf, func(inf, zero));
    EXPECT_FP_EQ(neg_inf, func(neg_inf, zero));
    EXPECT_FP_EQ(inf, func(inf, neg_zero));
    EXPECT_FP_EQ(neg_inf, func(neg_inf, neg_zero));
  }

  void test_invalid_operations(SubFunc func) {
    EXPECT_FP_IS_NAN_WITH_EXCEPTION(func(inf, inf), FE_INVALID);
    EXPECT_FP_IS_NAN_WITH_EXCEPTION(func(neg_inf, neg_inf), FE_INVALID);
  }

  void test_range_errors(SubFunc func) {
    using namespace LIBC_NAMESPACE::fputil::testing;

    if (ForceRoundingMode r(RoundingMode::Nearest); r.success) {
      EXPECT_FP_EQ_WITH_EXCEPTION(inf, func(max_normal, neg_max_normal),
                                  FE_OVERFLOW | FE_INEXACT);
      EXPECT_MATH_ERRNO(ERANGE);
      EXPECT_FP_EQ_WITH_EXCEPTION(-inf, func(neg_max_normal, max_normal),
                                  FE_OVERFLOW | FE_INEXACT);
      EXPECT_MATH_ERRNO(ERANGE);

      EXPECT_FP_EQ_WITH_EXCEPTION(zero,
                                  func(in.min_denormal, in.neg_min_denormal),
                                  FE_UNDERFLOW | FE_INEXACT);
      EXPECT_MATH_ERRNO(ERANGE);
      EXPECT_FP_EQ_WITH_EXCEPTION(neg_zero,
                                  func(in.neg_min_denormal, in.min_denormal),
                                  FE_UNDERFLOW | FE_INEXACT);
      EXPECT_MATH_ERRNO(ERANGE);
    }

    if (ForceRoundingMode r(RoundingMode::TowardZero); r.success) {
      EXPECT_FP_EQ_WITH_EXCEPTION(max_normal, func(max_normal, neg_max_normal),
                                  FE_OVERFLOW | FE_INEXACT);
      EXPECT_FP_EQ_WITH_EXCEPTION(neg_max_normal,
                                  func(neg_max_normal, max_normal),
                                  FE_OVERFLOW | FE_INEXACT);

      EXPECT_FP_EQ_WITH_EXCEPTION(zero,
                                  func(in.min_denormal, in.neg_min_denormal),
                                  FE_UNDERFLOW | FE_INEXACT);
      EXPECT_MATH_ERRNO(ERANGE);
      EXPECT_FP_EQ_WITH_EXCEPTION(neg_zero,
                                  func(in.neg_min_denormal, in.min_denormal),
                                  FE_UNDERFLOW | FE_INEXACT);
      EXPECT_MATH_ERRNO(ERANGE);
    }

    if (ForceRoundingMode r(RoundingMode::Downward); r.success) {
      EXPECT_FP_EQ_WITH_EXCEPTION(max_normal, func(max_normal, neg_max_normal),
                                  FE_OVERFLOW | FE_INEXACT);
      EXPECT_FP_EQ_WITH_EXCEPTION(-inf, func(neg_max_normal, max_normal),
                                  FE_OVERFLOW | FE_INEXACT);
      EXPECT_MATH_ERRNO(ERANGE);

      EXPECT_FP_EQ_WITH_EXCEPTION(zero,
                                  func(in.min_denormal, in.neg_min_denormal),
                                  FE_UNDERFLOW | FE_INEXACT);
      EXPECT_MATH_ERRNO(ERANGE);
      EXPECT_FP_EQ_WITH_EXCEPTION(neg_min_denormal,
                                  func(in.neg_min_denormal, in.min_denormal),
                                  FE_UNDERFLOW | FE_INEXACT);
      EXPECT_MATH_ERRNO(ERANGE);
    }

    if (ForceRoundingMode r(RoundingMode::Upward); r.success) {
      EXPECT_FP_EQ_WITH_EXCEPTION(inf, func(max_normal, neg_max_normal),
                                  FE_OVERFLOW | FE_INEXACT);
      EXPECT_MATH_ERRNO(ERANGE);
      EXPECT_FP_EQ_WITH_EXCEPTION(neg_max_normal,
                                  func(neg_max_normal, max_normal),
                                  FE_OVERFLOW | FE_INEXACT);

      EXPECT_FP_EQ_WITH_EXCEPTION(min_denormal,
                                  func(in.min_denormal, in.neg_min_denormal),
                                  FE_UNDERFLOW | FE_INEXACT);
      EXPECT_MATH_ERRNO(ERANGE);
      EXPECT_FP_EQ_WITH_EXCEPTION(neg_zero,
                                  func(in.neg_min_denormal, in.min_denormal),
                                  FE_UNDERFLOW | FE_INEXACT);
      EXPECT_MATH_ERRNO(ERANGE);
    }
  }

  void test_inexact_results(SubFunc func) {
    func(InType(1.0), min_denormal);
    EXPECT_FP_EXCEPTION(FE_INEXACT);
  }
};

#define LIST_SUB_TESTS(OutType, InType, func)                                  \
  using LlvmLibcSubTest = SubTest<OutType, InType>;                            \
  TEST_F(LlvmLibcSubTest, SpecialNumbers) { test_special_numbers(&func); }     \
  TEST_F(LlvmLibcSubTest, InvalidOperations) {                                 \
    test_invalid_operations(&func);                                            \
  }                                                                            \
  TEST_F(LlvmLibcSubTest, RangeErrors) { test_range_errors(&func); }           \
  TEST_F(LlvmLibcSubTest, InexactResults) { test_inexact_results(&func); }

#endif // LLVM_LIBC_TEST_SRC_MATH_SMOKE_SUBTEST_H
