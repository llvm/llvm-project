//===-- Unittests for fdiml -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "FDimTest.h"

#include "hdr/math_macros.h"
#include "src/__support/FPUtil/FPBits.h"
#include "src/math/fdiml.h"
#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"

using LlvmLibcFdimlTest = FDimTestTemplate<long double>;

TEST_F(LlvmLibcFdimlTest, NaNArg_fdiml) {
  test_na_n_arg(&LIBC_NAMESPACE::fdiml);
}

TEST_F(LlvmLibcFdimlTest, InfArg_fdiml) {
  test_inf_arg(&LIBC_NAMESPACE::fdiml);
}

TEST_F(LlvmLibcFdimlTest, NegInfArg_fdiml) {
  test_neg_inf_arg(&LIBC_NAMESPACE::fdiml);
}

TEST_F(LlvmLibcFdimlTest, BothZero_fdiml) {
  test_both_zero(&LIBC_NAMESPACE::fdiml);
}

TEST_F(LlvmLibcFdimlTest, InLongDoubleRange_fdiml) {
  test_in_range(&LIBC_NAMESPACE::fdiml);
}
