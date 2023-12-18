//===-- Utility class to test different flavors of ilogb --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_TEST_SRC_MATH_ILOGBTEST_H
#define LLVM_LIBC_TEST_SRC_MATH_ILOGBTEST_H

#include "src/__support/FPUtil/FPBits.h"
#include "src/__support/FPUtil/ManipulationFunctions.h"
#include "test/UnitTest/Test.h"
#include <math.h>

#include <limits.h>

class LlvmLibcILogbTest : public LIBC_NAMESPACE::testing::Test {
public:
  template <typename T> struct ILogbFunc {
    typedef int (*Func)(T);
  };

  template <typename T>
  void test_special_numbers(typename ILogbFunc<T>::Func func) {
    using FPBits = LIBC_NAMESPACE::fputil::FPBits<T>;
    EXPECT_EQ(FP_ILOGB0, func(T(FPBits::zero())));
    EXPECT_EQ(FP_ILOGB0, func(T(FPBits::neg_zero())));
    EXPECT_EQ(FP_ILOGBNAN, func(T(FPBits::build_quiet_nan(1))));
    EXPECT_EQ(INT_MAX, func(T(FPBits::inf())));
    EXPECT_EQ(INT_MAX, func(T(FPBits::neg_inf())));
  }

  template <typename T>
  void test_powers_of_two(typename ILogbFunc<T>::Func func) {
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
  void test_some_integers(typename ILogbFunc<T>::Func func) {
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
  void test_subnormal_range(typename ILogbFunc<T>::Func func) {
    using FPBits = LIBC_NAMESPACE::fputil::FPBits<T>;
    using StorageType = typename FPBits::StorageType;
    constexpr StorageType COUNT = 10'001;
    constexpr StorageType STEP =
        (FPBits::MAX_SUBNORMAL - FPBits::MIN_SUBNORMAL) / COUNT;
    for (StorageType v = FPBits::MIN_SUBNORMAL; v <= FPBits::MAX_SUBNORMAL;
         v += STEP) {
      T x = T(FPBits(v));
      if (isnan(x) || isinf(x) || x == 0.0)
        continue;

      int exponent;
      LIBC_NAMESPACE::fputil::frexp(x, exponent);
      ASSERT_EQ(exponent, func(x) + 1);
    }
  }

  template <typename T>
  void test_normal_range(typename ILogbFunc<T>::Func func) {
    using FPBits = LIBC_NAMESPACE::fputil::FPBits<T>;
    using StorageType = typename FPBits::StorageType;
    constexpr StorageType COUNT = 10'001;
    constexpr StorageType STEP =
        (FPBits::MAX_NORMAL - FPBits::MIN_NORMAL) / COUNT;
    for (StorageType v = FPBits::MIN_NORMAL; v <= FPBits::MAX_NORMAL;
         v += STEP) {
      T x = T(FPBits(v));
      if (isnan(x) || isinf(x) || x == 0.0)
        continue;

      int exponent;
      LIBC_NAMESPACE::fputil::frexp(x, exponent);
      ASSERT_EQ(exponent, func(x) + 1);
    }
  }
};

#endif // LLVM_LIBC_TEST_SRC_MATH_ILOGBTEST_H
