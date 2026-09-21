//===-- Unittests for fdim ------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "FDimTest.h"

#include "hdr/math_macros.h"
#include "src/__support/FPUtil/FPBits.h"
#include "src/math/fdim.h"
#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"

using LlvmLibcFdimTest = FDimTestTemplate<double>;

TEST_F(LlvmLibcFdimTest, NaNArg_fdim) { test_na_n_arg(&LIBC_NAMESPACE::fdim); }

TEST_F(LlvmLibcFdimTest, InfArg_fdim) { test_inf_arg(&LIBC_NAMESPACE::fdim); }

TEST_F(LlvmLibcFdimTest, NegInfArg_fdim) {
  test_neg_inf_arg(&LIBC_NAMESPACE::fdim);
}

TEST_F(LlvmLibcFdimTest, BothZero_fdim) {
  test_both_zero(&LIBC_NAMESPACE::fdim);
}

TEST_F(LlvmLibcFdimTest, InDoubleRange_fdim) {
  test_in_range(&LIBC_NAMESPACE::fdim);
}
