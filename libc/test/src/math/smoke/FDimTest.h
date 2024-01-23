//===-- Utility class to test different flavors of fdim ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===---------------------------------------------------------------------===//

#include "src/__support/FPUtil/BasicOperations.h"
#include "src/__support/FPUtil/FPBits.h"
#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"
#include <math.h>

template <typename T>
class FDimTestTemplate : public LIBC_NAMESPACE::testing::Test {
public:
  using FuncPtr = T (*)(T, T);
  using FPBits = LIBC_NAMESPACE::fputil::FPBits<T>;
  using StorageType = typename FPBits::StorageType;
  using Sign = LIBC_NAMESPACE::fputil::Sign;

  const T inf = T(FPBits::inf(Sign::POS));
  const T neg_inf = T(FPBits::inf(Sign::NEG));
  const T zero = T(FPBits::zero(Sign::POS));
  const T neg_zero = T(FPBits::zero(Sign::NEG));
  const T nan = T(FPBits::build_quiet_nan(1));

  void test_na_n_arg(FuncPtr func) {
    EXPECT_FP_EQ(nan, func(nan, inf));
    EXPECT_FP_EQ(nan, func(neg_inf, nan));
    EXPECT_FP_EQ(nan, func(nan, zero));
    EXPECT_FP_EQ(nan, func(neg_zero, nan));
    EXPECT_FP_EQ(nan, func(nan, T(-1.2345)));
    EXPECT_FP_EQ(nan, func(T(1.2345), nan));
    EXPECT_FP_EQ(func(nan, nan), nan);
  }

  void test_inf_arg(FuncPtr func) {
    EXPECT_FP_EQ(zero, func(neg_inf, inf));
    EXPECT_FP_EQ(inf, func(inf, zero));
    EXPECT_FP_EQ(zero, func(neg_zero, inf));
    EXPECT_FP_EQ(inf, func(inf, T(1.2345)));
    EXPECT_FP_EQ(zero, func(T(-1.2345), inf));
  }

  void test_neg_inf_arg(FuncPtr func) {
    EXPECT_FP_EQ(inf, func(inf, neg_inf));
    EXPECT_FP_EQ(zero, func(neg_inf, zero));
    EXPECT_FP_EQ(inf, func(neg_zero, neg_inf));
    EXPECT_FP_EQ(zero, func(neg_inf, T(-1.2345)));
    EXPECT_FP_EQ(inf, func(T(1.2345), neg_inf));
  }

  void test_both_zero(FuncPtr func) {
    EXPECT_FP_EQ(zero, func(zero, zero));
    EXPECT_FP_EQ(zero, func(zero, neg_zero));
    EXPECT_FP_EQ(zero, func(neg_zero, zero));
    EXPECT_FP_EQ(zero, func(neg_zero, neg_zero));
  }

  void test_in_range(FuncPtr func) {
    constexpr StorageType STORAGE_MAX =
        LIBC_NAMESPACE::cpp::numeric_limits<StorageType>::max();
    constexpr StorageType COUNT = 100'001;
    constexpr StorageType STEP = STORAGE_MAX / COUNT;
    for (StorageType i = 0, v = 0, w = STORAGE_MAX; i <= COUNT;
         ++i, v += STEP, w -= STEP) {
      T x = T(FPBits(v)), y = T(FPBits(w));
      if (isnan(x) || isinf(x))
        continue;
      if (isnan(y) || isinf(y))
        continue;

      if (x > y) {
        EXPECT_FP_EQ(x - y, func(x, y));
      } else {
        EXPECT_FP_EQ(zero, func(x, y));
      }
    }
  }
};
