//===-- Utility class to test different flavors of fma --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_TEST_SRC_MATH_FMATEST_H
#define LLVM_LIBC_TEST_SRC_MATH_FMATEST_H

#include "test/UnitTest/FEnvSafeTest.h"
#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"

template <typename OutType, typename InType = OutType>
class FmaTestTemplate : public LIBC_NAMESPACE::testing::FEnvSafeTest {

  struct OutConstants {
    DECLARE_SPECIAL_CONSTANTS(OutType)
  };

  struct InConstants {
    DECLARE_SPECIAL_CONSTANTS(InType)
  };

  using OutFPBits = typename OutConstants::FPBits;
  using OutStorageType = typename OutConstants::StorageType;
  using InFPBits = typename InConstants::FPBits;
  using InStorageType = typename InConstants::StorageType;

  static constexpr OutStorageType OUT_MIN_NORMAL_U =
      OutFPBits::min_normal().uintval();
  static constexpr InStorageType IN_MIN_NORMAL_U =
      InFPBits::min_normal().uintval();

  OutConstants out;
  InConstants in;

public:
  using FmaFunc = OutType (*)(InType, InType, InType);

  void test_special_numbers(FmaFunc func) {
    EXPECT_FP_EQ(out.zero, func(in.zero, in.zero, in.zero));
    EXPECT_FP_EQ(out.neg_zero, func(in.zero, in.neg_zero, in.neg_zero));
    EXPECT_FP_EQ(out.inf, func(in.inf, in.inf, in.zero));
    EXPECT_FP_EQ(out.neg_inf, func(in.neg_inf, in.inf, in.neg_inf));
    EXPECT_FP_EQ(out.aNaN, func(in.inf, in.zero, in.zero));
    EXPECT_FP_EQ(out.aNaN, func(in.inf, in.neg_inf, in.inf));
    EXPECT_FP_EQ(out.aNaN, func(in.aNaN, in.zero, in.inf));
    EXPECT_FP_EQ(out.aNaN, func(in.inf, in.neg_inf, in.aNaN));

    // Test underflow rounding up.
    EXPECT_FP_EQ(OutFPBits(OutStorageType(2)).get_val(),
                 func(OutType(0.5), out.min_denormal, out.min_denormal));

    if constexpr (sizeof(OutType) < sizeof(InType)) {
      EXPECT_FP_EQ(out.zero,
                   func(InType(0.5), in.min_denormal, in.min_denormal));
    }

    // Test underflow rounding down.
    OutType v = OutFPBits(static_cast<OutStorageType>(OUT_MIN_NORMAL_U +
                                                      OutStorageType(1)))
                    .get_val();
    EXPECT_FP_EQ(v, func(OutType(1) / OutType(OUT_MIN_NORMAL_U << 1), v,
                         out.min_normal));

    if constexpr (sizeof(OutType) < sizeof(InType)) {
      InType v = InFPBits(static_cast<InStorageType>(IN_MIN_NORMAL_U +
                                                     InStorageType(1)))
                     .get_val();
      EXPECT_FP_EQ(
          out.min_normal,
          func(InType(1) / InType(IN_MIN_NORMAL_U << 1), v, out.min_normal));
    }

    // Test overflow.
    OutType z = out.max_normal;
    EXPECT_FP_EQ_ALL_ROUNDING(OutType(0.75) * z, func(InType(1.75), z, -z));

    // Exact cancellation.
    EXPECT_FP_EQ_ROUNDING_NEAREST(
        out.zero, func(InType(3.0), InType(5.0), InType(-15.0)));
    EXPECT_FP_EQ_ROUNDING_UPWARD(out.zero,
                                 func(InType(3.0), InType(5.0), InType(-15.0)));
    EXPECT_FP_EQ_ROUNDING_TOWARD_ZERO(
        out.zero, func(InType(3.0), InType(5.0), InType(-15.0)));
    EXPECT_FP_EQ_ROUNDING_DOWNWARD(
        out.neg_zero, func(InType(3.0), InType(5.0), InType(-15.0)));

    EXPECT_FP_EQ_ROUNDING_NEAREST(
        out.zero, func(InType(-3.0), InType(5.0), InType(15.0)));
    EXPECT_FP_EQ_ROUNDING_UPWARD(out.zero,
                                 func(InType(-3.0), InType(5.0), InType(15.0)));
    EXPECT_FP_EQ_ROUNDING_TOWARD_ZERO(
        out.zero, func(InType(-3.0), InType(5.0), InType(15.0)));
    EXPECT_FP_EQ_ROUNDING_DOWNWARD(
        out.neg_zero, func(InType(-3.0), InType(5.0), InType(15.0)));
  }
};

#define LIST_FMA_TESTS(T, func)                                                \
  using LlvmLibcFmaTest = FmaTestTemplate<T>;                                  \
  TEST_F(LlvmLibcFmaTest, SpecialNumbers) { test_special_numbers(&func); }

#define LIST_NARROWING_FMA_TESTS(OutType, InType, func)                        \
  using LlvmLibcFmaTest = FmaTestTemplate<OutType, InType>;                    \
  TEST_F(LlvmLibcFmaTest, SpecialNumbers) { test_special_numbers(&func); }

#endif // LLVM_LIBC_TEST_SRC_MATH_FMATEST_H
