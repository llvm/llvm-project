//===-- Unittests for fdimf -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "FDimTest.h"

#include "include/math.h"
#include "src/math/fdimf.h"
#include "utils/FPUtil/FPBits.h"
#include "utils/FPUtil/TestHelpers.h"
#include "utils/UnitTest/Test.h"

using FDimTest = FDimTestTemplate<float>;

TEST_F(FDimTest, NaNArg_fdimf) { testNaNArg(&__llvm_libc::fdimf); }

TEST_F(FDimTest, InfArg_fdimf) { testInfArg(&__llvm_libc::fdimf); }

TEST_F(FDimTest, NegInfArg_fdimf) { testNegInfArg(&__llvm_libc::fdimf); }

TEST_F(FDimTest, BothZero_fdimf) { testBothZero(&__llvm_libc::fdimf); }

TEST_F(FDimTest, InFloatRange_fdimf) { testInRange(&__llvm_libc::fdimf); }
