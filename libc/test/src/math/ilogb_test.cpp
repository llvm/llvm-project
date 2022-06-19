//===-- Unittests for ilogb -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ILogbTest.h"

#include "src/__support/FPUtil/FPBits.h"
#include "src/__support/FPUtil/ManipulationFunctions.h"
#include "src/math/ilogb.h"
#include "utils/UnitTest/FPMatcher.h"
#include "utils/UnitTest/Test.h"
#include <math.h>

TEST_F(LlvmLibcILogbTest, SpecialNumbers_ilogb) {
  test_special_numbers<double>(&__llvm_libc::ilogb);
}

TEST_F(LlvmLibcILogbTest, PowersOfTwo_ilogb) {
  test_powers_of_two<double>(&__llvm_libc::ilogb);
}

TEST_F(LlvmLibcILogbTest, SomeIntegers_ilogb) {
  test_some_integers<double>(&__llvm_libc::ilogb);
}

TEST_F(LlvmLibcILogbTest, SubnormalRange_ilogb) {
  test_subnormal_range<double>(&__llvm_libc::ilogb);
}

TEST_F(LlvmLibcILogbTest, NormalRange_ilogb) {
  test_normal_range<double>(&__llvm_libc::ilogb);
}
