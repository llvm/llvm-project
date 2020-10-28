//===-- Unittests for floorf ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "include/math.h"
#include "src/math/floorf.h"
#include "utils/FPUtil/FPBits.h"
#include "utils/FPUtil/TestHelpers.h"
#include "utils/MPFRWrapper/MPFRUtils.h"
#include "utils/UnitTest/Test.h"

using FPBits = __llvm_libc::fputil::FPBits<float>;

namespace mpfr = __llvm_libc::testing::mpfr;

static const float zero = FPBits::zero();
static const float negZero = FPBits::negZero();
static const float nan = FPBits::buildNaN(1);
static const float inf = FPBits::inf();
static const float negInf = FPBits::negInf();

TEST(FloorfTest, SpecialNumbers) {
  EXPECT_FP_EQ(zero, __llvm_libc::floorf(zero));
  EXPECT_FP_EQ(negZero, __llvm_libc::floorf(negZero));

  EXPECT_FP_EQ(inf, __llvm_libc::floorf(inf));
  EXPECT_FP_EQ(negInf, __llvm_libc::floorf(negInf));

  ASSERT_NE(isnan(nan), 0);
  ASSERT_NE(isnan(__llvm_libc::floorf(nan)), 0);
}

TEST(FloorfTest, RoundedNumbers) {
  EXPECT_FP_EQ(1.0f, __llvm_libc::floorf(1.0f));
  EXPECT_FP_EQ(-1.0f, __llvm_libc::floorf(-1.0f));
  EXPECT_FP_EQ(10.0f, __llvm_libc::floorf(10.0f));
  EXPECT_FP_EQ(-10.0f, __llvm_libc::floorf(-10.0f));
  EXPECT_FP_EQ(1234.0f, __llvm_libc::floorf(1234.0f));
  EXPECT_FP_EQ(-1234.0f, __llvm_libc::floorf(-1234.0f));
}

TEST(FloorfTest, Fractions) {
  EXPECT_FP_EQ(0.0f, __llvm_libc::floorf(0.5f));
  EXPECT_FP_EQ(-1.0f, __llvm_libc::floorf(-0.5f));
  EXPECT_FP_EQ(0.0f, __llvm_libc::floorf(0.115f));
  EXPECT_FP_EQ(-1.0f, __llvm_libc::floorf(-0.115f));
  EXPECT_FP_EQ(0.0f, __llvm_libc::floorf(0.715f));
  EXPECT_FP_EQ(-1.0f, __llvm_libc::floorf(-0.715f));
  EXPECT_FP_EQ(1.0f, __llvm_libc::floorf(1.3f));
  EXPECT_FP_EQ(-2.0f, __llvm_libc::floorf(-1.3f));
  EXPECT_FP_EQ(1.0f, __llvm_libc::floorf(1.5f));
  EXPECT_FP_EQ(-2.0f, __llvm_libc::floorf(-1.5f));
  EXPECT_FP_EQ(1.0f, __llvm_libc::floorf(1.75f));
  EXPECT_FP_EQ(-2.0f, __llvm_libc::floorf(-1.75f));
  EXPECT_FP_EQ(10.0f, __llvm_libc::floorf(10.32f));
  EXPECT_FP_EQ(-11.0f, __llvm_libc::floorf(-10.32f));
  EXPECT_FP_EQ(10.0f, __llvm_libc::floorf(10.65f));
  EXPECT_FP_EQ(-11.0f, __llvm_libc::floorf(-10.65f));
  EXPECT_FP_EQ(1234.0f, __llvm_libc::floorf(1234.38f));
  EXPECT_FP_EQ(-1235.0f, __llvm_libc::floorf(-1234.38f));
  EXPECT_FP_EQ(1234.0f, __llvm_libc::floorf(1234.96f));
  EXPECT_FP_EQ(-1235.0f, __llvm_libc::floorf(-1234.96f));
}

TEST(FloorfTest, InFloatRange) {
  using UIntType = FPBits::UIntType;
  constexpr UIntType count = 1000000;
  constexpr UIntType step = UIntType(-1) / count;
  for (UIntType i = 0, v = 0; i <= count; ++i, v += step) {
    float x = FPBits(v);
    if (isnan(x) || isinf(x))
      continue;

    ASSERT_MPFR_MATCH(mpfr::Operation::Floor, x, __llvm_libc::floorf(x), 0.0);
  }
}
