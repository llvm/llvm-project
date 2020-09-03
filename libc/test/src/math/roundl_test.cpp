//===-- Unittests for roundl ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "include/math.h"
#include "src/math/roundl.h"
#include "utils/FPUtil/FPBits.h"
#include "utils/MPFRWrapper/MPFRUtils.h"
#include "utils/UnitTest/Test.h"

using FPBits = __llvm_libc::fputil::FPBits<long double>;

namespace mpfr = __llvm_libc::testing::mpfr;

// Zero tolerance; As in, exact match with MPFR result.
static constexpr mpfr::Tolerance tolerance{mpfr::Tolerance::floatPrecision, 0,
                                           0};
TEST(RoundlTest, SpecialNumbers) {
  ASSERT_TRUE(FPBits::zero() == __llvm_libc::roundl(FPBits::zero()));
  ASSERT_TRUE(FPBits::negZero() == __llvm_libc::roundl(FPBits::negZero()));

  ASSERT_TRUE(FPBits::inf() == __llvm_libc::roundl(FPBits::inf()));
  ASSERT_TRUE(FPBits::negInf() == __llvm_libc::roundl(FPBits::negInf()));

  long double nan = FPBits::buildNaN(1);
  ASSERT_NE(isnan(nan), 0);
  ASSERT_NE(isnan(__llvm_libc::roundl(nan)), 0);
}

TEST(RoundlTest, RoundedNumbers) {
  ASSERT_TRUE(FPBits(1.0l) == __llvm_libc::roundl(1.0l));
  ASSERT_TRUE(FPBits(-1.0l) == __llvm_libc::roundl(-1.0l));
  ASSERT_TRUE(FPBits(10.0l) == __llvm_libc::roundl(10.0l));
  ASSERT_TRUE(FPBits(-10.0l) == __llvm_libc::roundl(-10.0l));
  ASSERT_TRUE(FPBits(1234.0l) == __llvm_libc::roundl(1234.0l));
  ASSERT_TRUE(FPBits(-1234.0l) == __llvm_libc::roundl(-1234.0l));
}

TEST(RoundlTest, Fractions) {
  ASSERT_TRUE(FPBits(1.0l) == __llvm_libc::roundl(1.3l));
  ASSERT_TRUE(FPBits(-1.0l) == __llvm_libc::roundl(-1.3l));
  ASSERT_TRUE(FPBits(2.0l) == __llvm_libc::roundl(1.5l));
  ASSERT_TRUE(FPBits(-2.0l) == __llvm_libc::roundl(-1.5l));
  ASSERT_TRUE(FPBits(2.0l) == __llvm_libc::roundl(1.75l));
  ASSERT_TRUE(FPBits(-2.0l) == __llvm_libc::roundl(-1.75l));
  ASSERT_TRUE(FPBits(10.0l) == __llvm_libc::roundl(10.32l));
  ASSERT_TRUE(FPBits(-10.0l) == __llvm_libc::roundl(-10.32l));
  ASSERT_TRUE(FPBits(11.0l) == __llvm_libc::roundl(10.65l));
  ASSERT_TRUE(FPBits(-11.0l) == __llvm_libc::roundl(-10.65l));
  ASSERT_TRUE(FPBits(1234.0l) == __llvm_libc::roundl(1234.38l));
  ASSERT_TRUE(FPBits(-1234.0l) == __llvm_libc::roundl(-1234.38l));
  ASSERT_TRUE(FPBits(1235.0l) == __llvm_libc::roundl(1234.96l));
  ASSERT_TRUE(FPBits(-1235.0l) == __llvm_libc::roundl(-1234.96l));
}

TEST(RoundlTest, InLongDoubleRange) {
  using UIntType = FPBits::UIntType;
  constexpr UIntType count = 10000000;
  constexpr UIntType step = UIntType(-1) / count;
  for (UIntType i = 0, v = 0; i <= count; ++i, v += step) {
    long double x = FPBits(v);
    if (isnan(x) || isinf(x))
      continue;

    ASSERT_MPFR_MATCH(mpfr::Operation::Round, x, __llvm_libc::roundl(x),
                      tolerance);
  }
}
