//===-- Unittests for ilogbl ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ILogbTest.h"

#include "include/math.h"
#include "src/math/ilogbl.h"
#include "utils/FPUtil/FPBits.h"
#include "utils/FPUtil/ManipulationFunctions.h"
#include "utils/FPUtil/TestHelpers.h"
#include "utils/UnitTest/Test.h"

using RunContext = __llvm_libc::testing::RunContext;

TEST_F(ILogbTest, SpecialNumbers_ilogbl) {
  testSpecialNumbers<long double>(&__llvm_libc::ilogbl);
}

TEST_F(ILogbTest, PowersOfTwo_ilogbl) {
  testPowersOfTwo<long double>(&__llvm_libc::ilogbl);
}

TEST_F(ILogbTest, SomeIntegers_ilogbl) {
  testSomeIntegers<long double>(&__llvm_libc::ilogbl);
}

TEST_F(ILogbTest, SubnormalRange_ilogbl) {
  testSubnormalRange<long double>(&__llvm_libc::ilogbl);
}

TEST_F(ILogbTest, NormalRange_ilogbl) {
  testNormalRange<long double>(&__llvm_libc::ilogbl);
}
