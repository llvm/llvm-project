//===-- Unittests for shared math check exception functions ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "shared/math_check_exceptions.h"
#include "src/__support/CPP/array.h"
#include "src/__support/math/check/exp_exceptions.h"
#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"

template <typename T>
class CheckExpTest : public LIBC_NAMESPACE::testing::Test {

  DECLARE_SPECIAL_CONSTANTS(T)

public:
  void test() {
    using exp_exceptions = LIBC_NAMESPACE::shared::check::exp_exceptions<T>;
    using array = LIBC_NAMESPACE::cpp::array;
    using Bounds = LIBC_NAMESPACE::math::check::exp_internal::Bounds<T>;

    constexpr array<int, 4> ROUNDING_MODES[] = {FE_TONEAREST, FE_UPWARD,
                                                FE_DOWNWARD, FE_TOWARDZERO};
    for (auto rm : ROUNDING_MODES) {
      EXPECT_EQ(FE_INVALID, exp_exceptions(sNaN, rm));
      EXPECT_EQ(FE_INVALID, exp_exceptions(neg_sNaN, rm));
      EXPECT_EQ(0, exp_exceptions(aNaN, rm));
      EXPECT_EQ(0, exp_exceptions(neg_aNaN, rm));
      EXPECT_EQ(0, exp_exceptions(inf, rm));
      EXPECT_EQ(0, exp_exceptions(neg_inf, rm));
      EXPECT_EQ(0, exp_exceptions(zero, rm));
      EXPECT_EQ(0, exp_exceptions(neg_zero, rm));
      EXPECT_EQ(FE_OVERFLOW, exp_exceptions(max_normal, rm) & FE_OVERFLOW);
      EXPECT_EQ(FE_UNDERFLOW,
                exp_exceptions(neg_max_normal, rm) & FE_UNDERFLOW);
      EXPECT_EQ(FE_UNDERFLOW, exp_exceptions(Bounds::LOWER, rm) & FE_UNDERFLOW);
      EXPECT_EQ(0, exp_exceptions(T(1), rm) & (~FE_INEXACT));
    }

    EXPECT_EQ(FE_OVERFLOW,
              exp_exceptions(Bounds::UPPER, FE_TONEAREST) & FE_OVERFLOW);
    EXPECT_EQ(FE_OVERFLOW,
              exp_exceptions(Bounds::UPPER, FE_UPWARD) & FE_OVERFLOW);
    EXPECT_EQ(0, exp_exceptions(Bounds::UPPER, FE_DOWNWARD) & FE_OVERFLOW);
    EXPECT_EQ(0, exp_exceptions(Bounds::UPPER, FE_TOWARDZERO) & FE_OVERFLOW);
  }
};

using LlvmLibcCheckExpTestFloat = CheckExpTest<float>;
using LlvmLibcCheckExpTestDouble = CheckExpTest<double>;

TEST_F(LlvmLibcCheckExpTestFloat, CheckExpfExceptions) { test(); }
TEST_F(LlvmLibcCheckExpTestDouble, CheckExpExceptions) { test(); }
