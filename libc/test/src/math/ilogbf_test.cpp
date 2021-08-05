//===-- Unittests for ilogbf ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ILogbTest.h"

#include "src/__support/FPUtil/FPBits.h"
#include "src/__support/FPUtil/ManipulationFunctions.h"
#include "src/__support/FPUtil/TestHelpers.h"
#include "src/math/ilogbf.h"
#include "utils/UnitTest/Test.h"
#include <math.h>

TEST_F(LlvmLibcILogbTest, SpecialNumbers_ilogbf) {
  testSpecialNumbers<float>(&__llvm_libc::ilogbf);
}

TEST_F(LlvmLibcILogbTest, PowersOfTwo_ilogbf) {
  testPowersOfTwo<float>(&__llvm_libc::ilogbf);
}

TEST_F(LlvmLibcILogbTest, SomeIntegers_ilogbf) {
  testSomeIntegers<float>(&__llvm_libc::ilogbf);
}

TEST_F(LlvmLibcILogbTest, SubnormalRange_ilogbf) {
  testSubnormalRange<float>(&__llvm_libc::ilogbf);
}

TEST_F(LlvmLibcILogbTest, NormalRange_ilogbf) {
  testNormalRange<float>(&__llvm_libc::ilogbf);
}
