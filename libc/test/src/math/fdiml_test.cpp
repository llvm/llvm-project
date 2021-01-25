//===-- Unittests for fdiml -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "FDimTest.h"

#include "src/math/fdiml.h"
#include "utils/FPUtil/FPBits.h"
#include "utils/FPUtil/TestHelpers.h"
#include "utils/UnitTest/Test.h"
#include <math.h>

using LlvmLibcFDimTest = FDimTestTemplate<long double>;

TEST_F(LlvmLibcFDimTest, NaNArg_fdiml) { testNaNArg(&__llvm_libc::fdiml); }

TEST_F(LlvmLibcFDimTest, InfArg_fdiml) { testInfArg(&__llvm_libc::fdiml); }

TEST_F(LlvmLibcFDimTest, NegInfArg_fdiml) {
  testNegInfArg(&__llvm_libc::fdiml);
}

TEST_F(LlvmLibcFDimTest, BothZero_fdiml) { testBothZero(&__llvm_libc::fdiml); }

TEST_F(LlvmLibcFDimTest, InLongDoubleRange_fdiml) {
  testInRange(&__llvm_libc::fdiml);
}
