//===-- Utility class to test different flavors of ilogb --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_TEST_SRC_MATH_ILOGBTEST_H
#define LLVM_LIBC_TEST_SRC_MATH_ILOGBTEST_H

#include "utils/FPUtil/FPBits.h"
#include "utils/FPUtil/ManipulationFunctions.h"
#include "utils/UnitTest/Test.h"
#include <math.h>

#include <limits.h>

class ILogbTest : public __llvm_libc::testing::Test {
public:
  template <typename T> struct ILogbFunc { typedef int (*Func)(T); };

  template <typename T>
  void testSpecialNumbers(typename ILogbFunc<T>::Func func) {
    EXPECT_EQ(FP_ILOGB0, func(__llvm_libc::fputil::FPBits<T>::zero()));
    EXPECT_EQ(FP_ILOGB0, func(__llvm_libc::fputil::FPBits<T>::negZero()));

    EXPECT_EQ(FP_ILOGBNAN, func(__llvm_libc::fputil::FPBits<T>::buildNaN(1)));

    EXPECT_EQ(INT_MAX, func(__llvm_libc::fputil::FPBits<T>::inf()));
    EXPECT_EQ(INT_MAX, func(__llvm_libc::fputil::FPBits<T>::negInf()));
  }

  template <typename T> void testPowersOfTwo(typename ILogbFunc<T>::Func func) {
    EXPECT_EQ(0, func(T(1.0)));
    EXPECT_EQ(0, func(T(-1.0)));

    EXPECT_EQ(1, func(T(2.0)));
    EXPECT_EQ(1, func(T(-2.0)));

    EXPECT_EQ(2, func(T(4.0)));
    EXPECT_EQ(2, func(T(-4.0)));

    EXPECT_EQ(3, func(T(8.0)));
    EXPECT_EQ(3, func(-8.0));

    EXPECT_EQ(4, func(16.0));
    EXPECT_EQ(4, func(-16.0));

    EXPECT_EQ(5, func(32.0));
    EXPECT_EQ(5, func(-32.0));
  }

  template <typename T>
  void testSomeIntegers(typename ILogbFunc<T>::Func func) {
    EXPECT_EQ(1, func(T(3.0)));
    EXPECT_EQ(1, func(T(-3.0)));

    EXPECT_EQ(2, func(T(7.0)));
    EXPECT_EQ(2, func(T(-7.0)));

    EXPECT_EQ(3, func(T(10.0)));
    EXPECT_EQ(3, func(T(-10.0)));

    EXPECT_EQ(4, func(T(31.0)));
    EXPECT_EQ(4, func(-31.0));

    EXPECT_EQ(5, func(55.0));
    EXPECT_EQ(5, func(-55.0));
  }

  template <typename T>
  void testSubnormalRange(typename ILogbFunc<T>::Func func) {
    using FPBits = __llvm_libc::fputil::FPBits<T>;
    using UIntType = typename FPBits::UIntType;
    constexpr UIntType count = 1000001;
    constexpr UIntType step =
        (FPBits::maxSubnormal - FPBits::minSubnormal) / count;
    for (UIntType v = FPBits::minSubnormal; v <= FPBits::maxSubnormal;
         v += step) {
      T x = FPBits(v);
      if (isnan(x) || isinf(x) || x == 0.0)
        continue;

      int exponent;
      __llvm_libc::fputil::frexp(x, exponent);
      ASSERT_EQ(exponent, func(x) + 1);
    }
  }

  template <typename T> void testNormalRange(typename ILogbFunc<T>::Func func) {
    using FPBits = __llvm_libc::fputil::FPBits<T>;
    using UIntType = typename FPBits::UIntType;
    constexpr UIntType count = 1000001;
    constexpr UIntType step = (FPBits::maxNormal - FPBits::minNormal) / count;
    for (UIntType v = FPBits::minNormal; v <= FPBits::maxNormal; v += step) {
      T x = FPBits(v);
      if (isnan(x) || isinf(x) || x == 0.0)
        continue;

      int exponent;
      __llvm_libc::fputil::frexp(x, exponent);
      ASSERT_EQ(exponent, func(x) + 1);
    }
  }
};

#endif // LLVM_LIBC_TEST_SRC_MATH_ILOGBTEST_H
