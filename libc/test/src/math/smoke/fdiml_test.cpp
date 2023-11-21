//===-- Unittests for fdiml -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "FDimTest.h"

#include "src/__support/FPUtil/FPBits.h"
#include "src/math/fdiml.h"
#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"
#include <math.h>

using LlvmLibcFDimTest = FDimTestTemplate<long double>;

TEST_F(LlvmLibcFDimTest, NaNArg_fdiml) {
  test_na_n_arg(&LIBC_NAMESPACE::fdiml);
}

TEST_F(LlvmLibcFDimTest, InfArg_fdiml) { test_inf_arg(&LIBC_NAMESPACE::fdiml); }

TEST_F(LlvmLibcFDimTest, NegInfArg_fdiml) {
  test_neg_inf_arg(&LIBC_NAMESPACE::fdiml);
}

TEST_F(LlvmLibcFDimTest, BothZero_fdiml) {
  test_both_zero(&LIBC_NAMESPACE::fdiml);
}

TEST_F(LlvmLibcFDimTest, InLongDoubleRange_fdiml) {
  test_in_range(&LIBC_NAMESPACE::fdiml);
}
