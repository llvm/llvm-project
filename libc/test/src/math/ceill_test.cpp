//===-- Unittests for ceill -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "include/math.h"
#include "src/math/ceill.h"
#include "utils/FPUtil/FPBits.h"
#include "utils/MPFRWrapper/MPFRUtils.h"
#include "utils/UnitTest/Test.h"

using FPBits = __llvm_libc::fputil::FPBits<long double>;

namespace mpfr = __llvm_libc::testing::mpfr;

// Zero tolerance; As in, exact match with MPFR result.
static constexpr mpfr::Tolerance tolerance{mpfr::Tolerance::doublePrecision, 0,
                                           0};
TEST(CeillTest, SpecialNumbers) {
  ASSERT_TRUE(FPBits::zero() == __llvm_libc::ceill(FPBits::zero()));
  ASSERT_TRUE(FPBits::negZero() == __llvm_libc::ceill(FPBits::negZero()));

  ASSERT_TRUE(FPBits::inf() == __llvm_libc::ceill(FPBits::inf()));
  ASSERT_TRUE(FPBits::negInf() == __llvm_libc::ceill(FPBits::negInf()));

  long double nan = FPBits::buildNaN(1);
  ASSERT_NE(isnan(nan), 0);
  ASSERT_NE(isnan(__llvm_libc::ceill(nan)), 0);
}

TEST(CeillTest, RoundedNumbers) {
  ASSERT_TRUE(FPBits(1.0l) == __llvm_libc::ceill(1.0l));
  ASSERT_TRUE(FPBits(-1.0l) == __llvm_libc::ceill(-1.0l));
  ASSERT_TRUE(FPBits(10.0l) == __llvm_libc::ceill(10.0l));
  ASSERT_TRUE(FPBits(-10.0l) == __llvm_libc::ceill(-10.0l));
  ASSERT_TRUE(FPBits(1234.0l) == __llvm_libc::ceill(1234.0l));
  ASSERT_TRUE(FPBits(-1234.0l) == __llvm_libc::ceill(-1234.0l));
}

TEST(CeillTest, Fractions) {
  ASSERT_TRUE(FPBits(2.0l) == __llvm_libc::ceill(1.3l));
  ASSERT_TRUE(FPBits(-1.0l) == __llvm_libc::ceill(-1.3l));
  ASSERT_TRUE(FPBits(2.0l) == __llvm_libc::ceill(1.5l));
  ASSERT_TRUE(FPBits(-1.0l) == __llvm_libc::ceill(-1.5l));
  ASSERT_TRUE(FPBits(2.0l) == __llvm_libc::ceill(1.75l));
  ASSERT_TRUE(FPBits(-1.0l) == __llvm_libc::ceill(-1.75l));
  ASSERT_TRUE(FPBits(11.0l) == __llvm_libc::ceill(10.32l));
  ASSERT_TRUE(FPBits(-10.0l) == __llvm_libc::ceill(-10.32l));
  ASSERT_TRUE(FPBits(11.0l) == __llvm_libc::ceill(10.65l));
  ASSERT_TRUE(FPBits(-10.0l) == __llvm_libc::ceill(-10.65l));
  ASSERT_TRUE(FPBits(1235.0l) == __llvm_libc::ceill(1234.18l));
  ASSERT_TRUE(FPBits(-1234.0l) == __llvm_libc::ceill(-1234.18l));
  ASSERT_TRUE(FPBits(1235.0l) == __llvm_libc::ceill(1234.96l));
  ASSERT_TRUE(FPBits(-1234.0l) == __llvm_libc::ceill(-1234.96l));
}

TEST(CeillTest, InLongDoubleRange) {
  using UIntType = FPBits::UIntType;
  constexpr UIntType count = 10000000;
  constexpr UIntType step = UIntType(-1) / count;
  for (UIntType i = 0, v = 0; i <= count; ++i, v += step) {
    long double x = FPBits(v);
    if (isnan(x) || isinf(x))
      continue;

    ASSERT_MPFR_MATCH(mpfr::Operation::Ceil, x, __llvm_libc::ceill(x),
                      tolerance);
  }
}
