//===-- Utility class to test rsqrt[f|l] ------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_TEST_SRC_MATH_RSQRTTEST_H
#define LLVM_LIBC_TEST_SRC_MATH_RSQRTTEST_H

#include "test/UnitTest/FEnvSafeTest.h"
#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"
#include "utils/MPFRWrapper/MPFRUtils.h"

namespace mpfr = LIBC_NAMESPACE::testing::mpfr;

template <typename OutType, typename InType = OutType>
class RsqrtTest : public LIBC_NAMESPACE::testing::FEnvSafeTest {

  DECLARE_SPECIAL_CONSTANTS(InType)

  static constexpr StorageType HIDDEN_BIT =
      StorageType(1) << LIBC_NAMESPACE::fputil::FPBits<InType>::FRACTION_LEN;

public:
  using RsqrtFunc = OutType (*)(InType);

  // Subnormal inputs: probe both power-of-two mantissas and an even sampling
  // across the subnormal range.
  void test_denormal_values(RsqrtFunc func) {
    // Powers of two in the subnormal mantissa space.
    for (StorageType mant = 1; mant < HIDDEN_BIT; mant <<= 1) {
      FPBits denormal(zero);
      denormal.set_mantissa(mant);
      InType x = denormal.get_val();
      ASSERT_MPFR_MATCH_ALL_ROUNDING(mpfr::Operation::Rsqrt, x, func(x), 0.5);
    }

    // Even sampling across all subnormals.
    constexpr StorageType COUNT = 200'001;
    constexpr StorageType STEP = HIDDEN_BIT / COUNT;
    for (StorageType i = 0, v = 0; i <= COUNT; ++i, v += STEP) {
      InType x = FPBits(i).get_val();
      ASSERT_MPFR_MATCH_ALL_ROUNDING(mpfr::Operation::Rsqrt, x, func(x), 0.5);
    }
  }

  // Positive normal range sampling: skip NaNs and negative values.
  void test_normal_range(RsqrtFunc func) {
    constexpr StorageType COUNT = 200'001;
    constexpr StorageType STEP = STORAGE_MAX / COUNT;
    for (StorageType i = 0, v = 0; i <= COUNT; ++i, v += STEP) {
      FPBits x_bits(v);
      InType x = x_bits.get_val();
      if (x_bits.is_nan() || x_bits.is_neg())
        continue;
      ASSERT_MPFR_MATCH_ALL_ROUNDING(mpfr::Operation::Rsqrt, x, func(x), 0.5);
    }
  }
};

#define LIST_RSQRT_TESTS(T, func)                                              \
  using LlvmLibcRsqrtTest = RsqrtTest<T, T>;                                   \
  TEST_F(LlvmLibcRsqrtTest, DenormalValues) { test_denormal_values(&func); }   \
  TEST_F(LlvmLibcRsqrtTest, NormalRange) { test_normal_range(&func); }

#define LIST_NARROWING_RSQRT_TESTS(OutType, InType, func)                      \
  using LlvmLibcRsqrtTest = RsqrtTest<OutType, InType>;                        \
  TEST_F(LlvmLibcRsqrtTest, DenormalValues) { test_denormal_values(&func); }   \
  TEST_F(LlvmLibcRsqrtTest, NormalRange) { test_normal_range(&func); }

#endif // LLVM_LIBC_TEST_SRC_MATH_RSQRTTEST_H
