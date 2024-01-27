//===-- Utility class to test different flavors of rint ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_TEST_SRC_MATH_RINTTEST_H
#define LLVM_LIBC_TEST_SRC_MATH_RINTTEST_H

#include "src/__support/FPUtil/FEnvImpl.h"
#include "src/__support/FPUtil/FPBits.h"
#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"

#include <fenv.h>
#include <math.h>
#include <stdio.h>

static constexpr int ROUNDING_MODES[4] = {FE_UPWARD, FE_DOWNWARD, FE_TOWARDZERO,
                                          FE_TONEAREST};

template <typename T>
class RIntTestTemplate : public LIBC_NAMESPACE::testing::Test {
public:
  typedef T (*RIntFunc)(T);

private:
  using FPBits = LIBC_NAMESPACE::fputil::FPBits<T>;
  using StorageType = typename FPBits::StorageType;
  using Sign = LIBC_NAMESPACE::fputil::Sign;

  const T inf = FPBits::inf(Sign::POS).get_val();
  const T neg_inf = FPBits::inf(Sign::NEG).get_val();
  const T zero = FPBits::zero(Sign::POS).get_val();
  const T neg_zero = FPBits::zero(Sign::NEG).get_val();
  const T nan = FPBits::build_quiet_nan().get_val();

public:
  void testSpecialNumbers(RIntFunc func) {
    for (int mode : ROUNDING_MODES) {
      LIBC_NAMESPACE::fputil::set_round(mode);
      ASSERT_FP_EQ(inf, func(inf));
      ASSERT_FP_EQ(neg_inf, func(neg_inf));
      ASSERT_FP_EQ(nan, func(nan));
      ASSERT_FP_EQ(zero, func(zero));
      ASSERT_FP_EQ(neg_zero, func(neg_zero));
    }
  }
};

#define LIST_RINT_TESTS(F, func)                                               \
  using LlvmLibcRIntTest = RIntTestTemplate<F>;                                \
  TEST_F(LlvmLibcRIntTest, specialNumbers) { testSpecialNumbers(&func); }

#endif // LLVM_LIBC_TEST_SRC_MATH_RINTTEST_H
