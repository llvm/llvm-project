//===-- Utility class to test different flavors of rint ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_TEST_SRC_MATH_RINTTEST_H
#define LLVM_LIBC_TEST_SRC_MATH_RINTTEST_H

#undef LIBC_MATH_USE_SYSTEM_FENV

#include "src/__support/CPP/algorithm.h"
#include "src/__support/FPUtil/FPBits.h"
#include "test/UnitTest/FEnvSafeTest.h"
#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"
#include "utils/MPFRWrapper/MPFRUtils.h"

#include "hdr/fenv_macros.h"
#include "hdr/math_macros.h"

namespace mpfr = LIBC_NAMESPACE::testing::mpfr;
using LIBC_NAMESPACE::Sign;

template <typename T>
class RIntTestTemplate : public LIBC_NAMESPACE::testing::FEnvSafeTest {
public:
  typedef T (*RIntFunc)(T);

private:
  using FPBits = LIBC_NAMESPACE::fputil::FPBits<T>;
  using FPTest = LIBC_NAMESPACE::testing::FPTest<T>;
  using StorageType = typename FPBits::StorageType;

  const T inf = FPBits::inf(Sign::POS).get_val();
  const T neg_inf = FPBits::inf(Sign::NEG).get_val();
  const T zero = FPBits::zero(Sign::POS).get_val();
  const T neg_zero = FPBits::zero(Sign::NEG).get_val();
  const T nan = FPBits::quiet_nan().get_val();

  static constexpr StorageType MIN_SUBNORMAL =
      FPBits::min_subnormal().uintval();
  static constexpr StorageType MAX_SUBNORMAL =
      FPBits::max_subnormal().uintval();
  static constexpr StorageType MIN_NORMAL = FPBits::min_normal().uintval();
  static constexpr StorageType MAX_NORMAL = FPBits::max_normal().uintval();

public:
  void testSpecialNumbers(RIntFunc func) {
    ASSERT_FP_EQ_ALL_ROUNDING(inf, func(inf));
    ASSERT_FP_EQ_ALL_ROUNDING(neg_inf, func(neg_inf));
    ASSERT_FP_EQ_ALL_ROUNDING(nan, func(nan));
    ASSERT_FP_EQ_ALL_ROUNDING(zero, func(zero));
    ASSERT_FP_EQ_ALL_ROUNDING(neg_zero, func(neg_zero));
  }

  void testRoundNumbers(RIntFunc func) {
    for (auto mpfr_mode : FPTest::ROUNDING_MODES) {
      ASSERT_FP_EQ_ROUNDING_MODE(mpfr::round(T(1.0), mpfr_mode), func(T(1.0)),
                                 mpfr_mode);
      ASSERT_FP_EQ_ROUNDING_MODE(mpfr::round(T(-1.0), mpfr_mode), func(T(-1.0)),
                                 mpfr_mode);
      ASSERT_FP_EQ_ROUNDING_MODE(mpfr::round(T(10.0), mpfr_mode), func(T(10.0)),
                                 mpfr_mode);
      ASSERT_FP_EQ_ROUNDING_MODE(mpfr::round(T(-10.0), mpfr_mode),
                                 func(T(-10.0)), mpfr_mode);
      ASSERT_FP_EQ_ROUNDING_MODE(mpfr::round(T(1234.0), mpfr_mode),
                                 func(T(1234.0)), mpfr_mode);
      ASSERT_FP_EQ_ROUNDING_MODE(mpfr::round(T(-1234.0), mpfr_mode),
                                 func(T(-1234.0)), mpfr_mode);
    }
  }

  void testFractions(RIntFunc func) {
    for (auto mpfr_mode : FPTest::ROUNDING_MODES) {
      ASSERT_FP_EQ_ROUNDING_MODE(mpfr::round(T(0.5), mpfr_mode), func(T(0.5)),
                                 mpfr_mode);
      ASSERT_FP_EQ_ROUNDING_MODE(mpfr::round(T(-0.5), mpfr_mode), func(T(-0.5)),
                                 mpfr_mode);
      ASSERT_FP_EQ_ROUNDING_MODE(mpfr::round(T(0.115), mpfr_mode),
                                 func(T(0.115)), mpfr_mode);
      ASSERT_FP_EQ_ROUNDING_MODE(mpfr::round(T(-0.115), mpfr_mode),
                                 func(T(-0.115)), mpfr_mode);
      ASSERT_FP_EQ_ROUNDING_MODE(mpfr::round(T(0.715), mpfr_mode),
                                 func(T(0.715)), mpfr_mode);
      ASSERT_FP_EQ_ROUNDING_MODE(mpfr::round(T(-0.715), mpfr_mode),
                                 func(T(-0.715)), mpfr_mode);
    }
  }

  void testSubnormalRange(RIntFunc func) {
    constexpr int COUNT = 1'231;
    constexpr StorageType STEP = LIBC_NAMESPACE::cpp::max(
        static_cast<StorageType>((MAX_SUBNORMAL - MIN_SUBNORMAL) / COUNT),
        StorageType(1));
    for (StorageType i = MIN_SUBNORMAL; i <= MAX_SUBNORMAL; i += STEP) {
      T x = FPBits(i).get_val();
      for (auto mpfr_mode : FPTest::ROUNDING_MODES) {
        ASSERT_FP_EQ_ROUNDING_MODE(mpfr::round(x, mpfr_mode), func(x),
                                   mpfr_mode);
      }
    }
  }

  void testNormalRange(RIntFunc func) {
    constexpr int COUNT = 1'231;
    constexpr StorageType STEP = LIBC_NAMESPACE::cpp::max(
        static_cast<StorageType>((MAX_NORMAL - MIN_NORMAL) / COUNT),
        StorageType(1));
    for (StorageType i = MIN_NORMAL; i <= MAX_NORMAL; i += STEP) {
      FPBits xbits(i);
      T x = xbits.get_val();
      // In normal range on x86 platforms, the long double implicit 1 bit can be
      // zero making the numbers NaN. We will skip them.
      if (xbits.is_nan())
        continue;

      for (auto mpfr_mode : FPTest::ROUNDING_MODES) {
        ASSERT_FP_EQ_ROUNDING_MODE(mpfr::round(x, mpfr_mode), func(x),
                                   mpfr_mode);
      }
    }
  }
};

#define LIST_RINT_TESTS(F, func)                                               \
  using LlvmLibcRIntTest = RIntTestTemplate<F>;                                \
  TEST_F(LlvmLibcRIntTest, specialNumbers) { testSpecialNumbers(&func); }      \
  TEST_F(LlvmLibcRIntTest, RoundNumbers) { testRoundNumbers(&func); }          \
  TEST_F(LlvmLibcRIntTest, Fractions) { testFractions(&func); }                \
  TEST_F(LlvmLibcRIntTest, SubnormalRange) { testSubnormalRange(&func); }      \
  TEST_F(LlvmLibcRIntTest, NormalRange) { testNormalRange(&func); }

#endif // LLVM_LIBC_TEST_SRC_MATH_RINTTEST_H
