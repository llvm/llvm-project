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

template <typename T> class FAbsTest : public __llvm_libc::testing::Test {

  DECLARE_SPECIAL_CONSTANTS(T)

public:
  typedef T (*FabsFunc)(T);

  void testSpecialNumbers(FabsFunc func) {
    EXPECT_FP_EQ(aNaN, func(aNaN));

    EXPECT_FP_EQ(inf, func(inf));
    EXPECT_FP_EQ(inf, func(neg_inf));

    EXPECT_FP_EQ(zero, func(zero));
    EXPECT_FP_EQ(zero, func(neg_zero));
  }

  void testRange(FabsFunc func) {
    constexpr UIntType COUNT = 10000000;
    constexpr UIntType STEP = UIntType(-1) / COUNT;
    for (UIntType i = 0, v = 0; i <= COUNT; ++i, v += STEP) {
      T x = T(FPBits(v));
      if (isnan(x) || isinf(x))
        continue;
      ASSERT_MPFR_MATCH(mpfr::Operation::Abs, x, func(x), 0.0);
    }
  }
};

#define LIST_FABS_TESTS(T, func)                                               \
  using LlvmLibcFAbsTest = FAbsTest<T>;                                        \
  TEST_F(LlvmLibcFAbsTest, SpecialNumbers) { testSpecialNumbers(&func); }      \
  TEST_F(LlvmLibcFAbsTest, Range) { testRange(&func); }
