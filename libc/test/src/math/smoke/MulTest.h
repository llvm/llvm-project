//===-- Utility class to test different flavors of float mul ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_TEST_SRC_MATH_SMOKE_MULTEST_H
#define LLVM_LIBC_TEST_SRC_MATH_SMOKE_MULTEST_H

#include "hdr/errno_macros.h"
#include "hdr/fenv_macros.h"
#include "src/__support/FPUtil/BasicOperations.h"
#include "test/UnitTest/FEnvSafeTest.h"
#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"

template <typename OutType, typename InType>
class MulTest : public LIBC_NAMESPACE::testing::FEnvSafeTest {

  DECLARE_SPECIAL_CONSTANTS(OutType)

  struct InConstants {
    DECLARE_SPECIAL_CONSTANTS(InType)
  };

  using InFPBits = typename InConstants::FPBits;
  using InStorageType = typename InConstants::StorageType;

  InConstants in;

public:
  using MulFunc = OutType (*)(InType, InType);

  void test_special_numbers(MulFunc func) {
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

    EXPECT_FP_EQ(inf, func(inf, InType(1.0)));
    EXPECT_FP_EQ(neg_inf, func(neg_inf, InType(1.0)));
    EXPECT_FP_EQ(neg_inf, func(inf, InType(-1.0)));
    EXPECT_FP_EQ(inf, func(neg_inf, InType(-1.0)));

    EXPECT_FP_EQ_ALL_ROUNDING(zero, func(zero, zero));
    EXPECT_FP_EQ_ALL_ROUNDING(zero, func(neg_zero, neg_zero));
    EXPECT_FP_EQ_ALL_ROUNDING(neg_zero, func(zero, neg_zero));
    EXPECT_FP_EQ_ALL_ROUNDING(neg_zero, func(neg_zero, zero));

    EXPECT_FP_EQ_ALL_ROUNDING(OutType(1.0), func(1.0, 1.0));
    EXPECT_FP_EQ_ALL_ROUNDING(OutType(15.0), func(3.0, 5.0));
    EXPECT_FP_EQ_ALL_ROUNDING(OutType(0x1.0p-13), func(0x1.0p+1, 0x1.0p-14));
    EXPECT_FP_EQ_ALL_ROUNDING(OutType(0x1.0p-10), func(0x1.0p+2, 0x1.0p-12));
  }

  void test_invalid_operations(MulFunc func) {
    EXPECT_FP_IS_NAN_WITH_EXCEPTION(func(inf, zero), FE_INVALID);
    EXPECT_FP_IS_NAN_WITH_EXCEPTION(func(inf, neg_zero), FE_INVALID);
    EXPECT_FP_IS_NAN_WITH_EXCEPTION(func(neg_inf, zero), FE_INVALID);
    EXPECT_FP_IS_NAN_WITH_EXCEPTION(func(neg_inf, neg_zero), FE_INVALID);
  }

  void test_range_errors(MulFunc func) {
    using namespace LIBC_NAMESPACE::fputil::testing;

    if (ForceRoundingMode r(RoundingMode::Nearest); r.success) {
      EXPECT_FP_EQ_WITH_EXCEPTION(inf, func(max_normal, max_normal),
                                  FE_OVERFLOW | FE_INEXACT);
      EXPECT_MATH_ERRNO(ERANGE);
      EXPECT_FP_EQ_WITH_EXCEPTION(neg_inf, func(neg_max_normal, max_normal),
                                  FE_OVERFLOW | FE_INEXACT);
      EXPECT_MATH_ERRNO(ERANGE);

      EXPECT_FP_EQ_WITH_EXCEPTION(zero, func(in.min_denormal, in.min_denormal),
                                  FE_UNDERFLOW | FE_INEXACT);
      EXPECT_MATH_ERRNO(ERANGE);
      EXPECT_FP_EQ_WITH_EXCEPTION(neg_zero,
                                  func(in.neg_min_denormal, in.min_denormal),
                                  FE_UNDERFLOW | FE_INEXACT);
      EXPECT_MATH_ERRNO(ERANGE);
    }

    if (ForceRoundingMode r(RoundingMode::TowardZero); r.success) {
      EXPECT_FP_EQ_WITH_EXCEPTION(max_normal, func(max_normal, max_normal),
                                  FE_OVERFLOW | FE_INEXACT);
      EXPECT_FP_EQ_WITH_EXCEPTION(neg_max_normal,
                                  func(neg_max_normal, max_normal),
                                  FE_OVERFLOW | FE_INEXACT);

      EXPECT_FP_EQ_WITH_EXCEPTION(zero, func(in.min_denormal, in.min_denormal),
                                  FE_UNDERFLOW | FE_INEXACT);
      EXPECT_MATH_ERRNO(ERANGE);
      EXPECT_FP_EQ_WITH_EXCEPTION(neg_zero,
                                  func(in.neg_min_denormal, in.min_denormal),
                                  FE_UNDERFLOW | FE_INEXACT);
      EXPECT_MATH_ERRNO(ERANGE);
    }

    if (ForceRoundingMode r(RoundingMode::Downward); r.success) {
      EXPECT_FP_EQ_WITH_EXCEPTION(max_normal, func(max_normal, max_normal),
                                  FE_OVERFLOW | FE_INEXACT);
      EXPECT_FP_EQ_WITH_EXCEPTION(neg_inf, func(neg_max_normal, max_normal),
                                  FE_OVERFLOW | FE_INEXACT);
      EXPECT_MATH_ERRNO(ERANGE);

      EXPECT_FP_EQ_WITH_EXCEPTION(zero, func(in.min_denormal, in.min_denormal),
                                  FE_UNDERFLOW | FE_INEXACT);
      EXPECT_MATH_ERRNO(ERANGE);
      EXPECT_FP_EQ_WITH_EXCEPTION(neg_min_denormal,
                                  func(in.neg_min_denormal, in.min_denormal),
                                  FE_UNDERFLOW | FE_INEXACT);
      EXPECT_MATH_ERRNO(ERANGE);
    }

    if (ForceRoundingMode r(RoundingMode::Upward); r.success) {
      EXPECT_FP_EQ_WITH_EXCEPTION(inf, func(max_normal, max_normal),
                                  FE_OVERFLOW | FE_INEXACT);
      EXPECT_MATH_ERRNO(ERANGE);
      EXPECT_FP_EQ_WITH_EXCEPTION(neg_max_normal,
                                  func(neg_max_normal, max_normal),
                                  FE_OVERFLOW | FE_INEXACT);

      EXPECT_FP_EQ_WITH_EXCEPTION(min_denormal,
                                  func(in.min_denormal, in.min_denormal),
                                  FE_UNDERFLOW | FE_INEXACT);
      EXPECT_MATH_ERRNO(ERANGE);
      EXPECT_FP_EQ_WITH_EXCEPTION(neg_zero,
                                  func(in.neg_min_denormal, in.min_denormal),
                                  FE_UNDERFLOW | FE_INEXACT);
      EXPECT_MATH_ERRNO(ERANGE);
    }
  }

  void test_inexact_results(MulFunc func) {
    InFPBits x_bits = InFPBits::one();
    x_bits.set_mantissa(InFPBits::SIG_MASK);
    InType x = x_bits.get_val();
    func(x, x);
    EXPECT_FP_EXCEPTION(FE_INEXACT);
  }
};

#define LIST_MUL_TESTS(OutType, InType, func)                                  \
  using LlvmLibcMulTest = MulTest<OutType, InType>;                            \
  TEST_F(LlvmLibcMulTest, SpecialNumbers) { test_special_numbers(&func); }     \
  TEST_F(LlvmLibcMulTest, InvalidOperations) {                                 \
    test_invalid_operations(&func);                                            \
  }                                                                            \
  TEST_F(LlvmLibcMulTest, RangeErrors) { test_range_errors(&func); }           \
  TEST_F(LlvmLibcMulTest, InexactResults) { test_inexact_results(&func); }

#endif // LLVM_LIBC_TEST_SRC_MATH_SMOKE_MULTEST_H
