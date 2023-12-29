//===-- Utility class to test copysign[f|l] ---------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"

#include <math.h>

template <typename T>
class CopySignTest : public LIBC_NAMESPACE::testing::Test {

  DECLARE_SPECIAL_CONSTANTS(T)

public:
  typedef T (*CopySignFunc)(T, T);

  void testSpecialNumbers(CopySignFunc func) {
    EXPECT_FP_EQ(aNaN, func(aNaN, T(-1.0)));
    EXPECT_FP_EQ(aNaN, func(aNaN, T(1.0)));

    EXPECT_FP_EQ(neg_inf, func(inf, T(-1.0)));
    EXPECT_FP_EQ(inf, func(neg_inf, T(1.0)));

    EXPECT_FP_EQ(neg_zero, func(zero, T(-1.0)));
    EXPECT_FP_EQ(zero, func(neg_zero, T(1.0)));
  }

  void testRange(CopySignFunc func) {
    constexpr StorageType COUNT = 100'000;
    constexpr StorageType STEP = STORAGE_MAX / COUNT;
    for (StorageType i = 0, v = 0; i <= COUNT; ++i, v += STEP) {
      FPBits x_bits = FPBits(v);
      T x = T(v);
      if (x_bits.is_nan() || x_bits.is_inf())
        continue;

      T res1 = func(x, -x);
      ASSERT_FP_EQ(res1, -x);

      T res2 = func(x, x);
      ASSERT_FP_EQ(res2, x);
    }
  }
};

#define LIST_COPYSIGN_TESTS(T, func)                                           \
  using LlvmLibcCopySignTest = CopySignTest<T>;                                \
  TEST_F(LlvmLibcCopySignTest, SpecialNumbers) { testSpecialNumbers(&func); }  \
  TEST_F(LlvmLibcCopySignTest, Range) { testRange(&func); }
