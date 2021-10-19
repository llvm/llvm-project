//===-- Unittests for fdimf -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "FDimTest.h"

#include "src/__support/FPUtil/FPBits.h"
#include "src/__support/FPUtil/TestHelpers.h"
#include "src/math/fdimf.h"
#include "utils/UnitTest/Test.h"
#include <math.h>

using LlvmLibcFDimTest = FDimTestTemplate<float>;

TEST_F(LlvmLibcFDimTest, NaNArg_fdimf) { testNaNArg(&__llvm_libc::fdimf); }

TEST_F(LlvmLibcFDimTest, InfArg_fdimf) { testInfArg(&__llvm_libc::fdimf); }

TEST_F(LlvmLibcFDimTest, NegInfArg_fdimf) {
  testNegInfArg(&__llvm_libc::fdimf);
}

TEST_F(LlvmLibcFDimTest, BothZero_fdimf) { testBothZero(&__llvm_libc::fdimf); }

TEST_F(LlvmLibcFDimTest, InFloatRange_fdimf) {
  testInRange(&__llvm_libc::fdimf);
}
