//===-- Unittests for ilogbf ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ILogbTest.h"

#include "src/math/ilogbf.h"
#include "utils/FPUtil/FPBits.h"
#include "utils/FPUtil/ManipulationFunctions.h"
#include "utils/FPUtil/TestHelpers.h"
#include "utils/UnitTest/Test.h"
#include <math.h>

TEST_F(ILogbTest, SpecialNumbers_ilogbf) {
  testSpecialNumbers<float>(&__llvm_libc::ilogbf);
}

TEST_F(ILogbTest, PowersOfTwo_ilogbf) {
  testPowersOfTwo<float>(&__llvm_libc::ilogbf);
}

TEST_F(ILogbTest, SomeIntegers_ilogbf) {
  testSomeIntegers<float>(&__llvm_libc::ilogbf);
}

TEST_F(ILogbTest, SubnormalRange_ilogbf) {
  testSubnormalRange<float>(&__llvm_libc::ilogbf);
}

TEST_F(ILogbTest, NormalRange_ilogbf) {
  testNormalRange<float>(&__llvm_libc::ilogbf);
}
