//===-- Utility class to test fabs[f|l] -------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "utils/MPFRWrapper/MPFRUtils.h"
#include "utils/UnitTest/FPMatcher.h"
#include "utils/UnitTest/Test.h"

#include <math.h>

namespace mpfr = __llvm_libc::testing::mpfr;

template <typename T> class SqrtTest : public __llvm_libc::testing::Test {

  DECLARE_SPECIAL_CONSTANTS(T)

  static constexpr UIntType HiddenBit =
      UIntType(1) << __llvm_libc::fputil::MantissaWidth<T>::value;

public:
  typedef T (*SqrtFunc)(T);

  void testSpecialNumbers(SqrtFunc func) {
    ASSERT_FP_EQ(aNaN, func(aNaN));
    ASSERT_FP_EQ(inf, func(inf));
    ASSERT_FP_EQ(aNaN, func(negInf));
    ASSERT_FP_EQ(0.0, func(0.0));
    ASSERT_FP_EQ(-0.0, func(-0.0));
    ASSERT_FP_EQ(aNaN, func(T(-1.0)));
    ASSERT_FP_EQ(T(1.0), func(T(1.0)));
    ASSERT_FP_EQ(T(2.0), func(T(4.0)));
    ASSERT_FP_EQ(T(3.0), func(T(9.0)));
  }

  void testDenormalValues(SqrtFunc func) {
    for (UIntType mant = 1; mant < HiddenBit; mant <<= 1) {
      FPBits denormal(T(0.0));
      denormal.setMantissa(mant);

      ASSERT_MPFR_MATCH(mpfr::Operation::Sqrt, T(denormal), func(T(denormal)),
                        T(0.5));
    }

    constexpr UIntType count = 1'000'001;
    constexpr UIntType step = HiddenBit / count;
    for (UIntType i = 0, v = 0; i <= count; ++i, v += step) {
      T x = *reinterpret_cast<T *>(&v);
      ASSERT_MPFR_MATCH(mpfr::Operation::Sqrt, x, func(x), 0.5);
    }
  }

  void testNormalRange(SqrtFunc func) {
    constexpr UIntType count = 10'000'001;
    constexpr UIntType step = UIntType(-1) / count;
    for (UIntType i = 0, v = 0; i <= count; ++i, v += step) {
      T x = *reinterpret_cast<T *>(&v);
      if (isnan(x) || (x < 0)) {
        continue;
      }
      ASSERT_MPFR_MATCH(mpfr::Operation::Sqrt, x, func(x), 0.5);
    }
  }
};

#define LIST_SQRT_TESTS(T, func)                                               \
  using LlvmLibcSqrtTest = SqrtTest<T>;                                        \
  TEST_F(LlvmLibcSqrtTest, SpecialNumbers) { testSpecialNumbers(&func); }      \
  TEST_F(LlvmLibcSqrtTest, DenormalValues) { testDenormalValues(&func); }      \
  TEST_F(LlvmLibcSqrtTest, NormalRange) { testNormalRange(&func); }
