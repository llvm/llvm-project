//===-- Unittests for fdim ------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "FDimTest.h"

#include "src/math/fdim.h"
#include "utils/FPUtil/FPBits.h"
#include "utils/FPUtil/TestHelpers.h"
#include "utils/UnitTest/Test.h"
#include <math.h>

using FDimTest = FDimTestTemplate<double>;

TEST_F(FDimTest, NaNArg_fdim) { testNaNArg(&__llvm_libc::fdim); }

TEST_F(FDimTest, InfArg_fdim) { testInfArg(&__llvm_libc::fdim); }

TEST_F(FDimTest, NegInfArg_fdim) { testNegInfArg(&__llvm_libc::fdim); }

TEST_F(FDimTest, BothZero_fdim) { testBothZero(&__llvm_libc::fdim); }

TEST_F(FDimTest, InDoubleRange_fdim) { testInRange(&__llvm_libc::fdim); }
