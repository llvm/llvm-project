//===-- Unittests for fdim ------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "FDimTest.h"

#include "src/__support/FPUtil/FPBits.h"
#include "src/math/fdim.h"
#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"
#include <math.h>

using LlvmLibcFDimTest = FDimTestTemplate<double>;

TEST_F(LlvmLibcFDimTest, NaNArg_fdim) { test_na_n_arg(&LIBC_NAMESPACE::fdim); }

TEST_F(LlvmLibcFDimTest, InfArg_fdim) { test_inf_arg(&LIBC_NAMESPACE::fdim); }

TEST_F(LlvmLibcFDimTest, NegInfArg_fdim) {
  test_neg_inf_arg(&LIBC_NAMESPACE::fdim);
}

TEST_F(LlvmLibcFDimTest, BothZero_fdim) {
  test_both_zero(&LIBC_NAMESPACE::fdim);
}

TEST_F(LlvmLibcFDimTest, InDoubleRange_fdim) {
  test_in_range(&LIBC_NAMESPACE::fdim);
}
