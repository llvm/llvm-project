//===-- Unittests for ilogb -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ILogbTest.h"

#include "src/math/ilogb.h"
#include "utils/FPUtil/FPBits.h"
#include "utils/FPUtil/ManipulationFunctions.h"
#include "utils/FPUtil/TestHelpers.h"
#include "utils/UnitTest/Test.h"
#include <math.h>

TEST_F(ILogbTest, SpecialNumbers_ilogb) {
  testSpecialNumbers<double>(&__llvm_libc::ilogb);
}

TEST_F(ILogbTest, PowersOfTwo_ilogb) {
  testPowersOfTwo<double>(&__llvm_libc::ilogb);
}

TEST_F(ILogbTest, SomeIntegers_ilogb) {
  testSomeIntegers<double>(&__llvm_libc::ilogb);
}

TEST_F(ILogbTest, SubnormalRange_ilogb) {
  testSubnormalRange<double>(&__llvm_libc::ilogb);
}

TEST_F(ILogbTest, NormalRange_ilogb) {
  testNormalRange<double>(&__llvm_libc::ilogb);
}
