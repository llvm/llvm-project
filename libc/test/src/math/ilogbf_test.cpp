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
#include "src/math/ilogbf.h"
#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"
#include <math.h>

TEST_F(LlvmLibcILogbTest, SpecialNumbers_ilogbf) {
  test_special_numbers<float>(&__llvm_libc::ilogbf);
}

TEST_F(LlvmLibcILogbTest, PowersOfTwo_ilogbf) {
  test_powers_of_two<float>(&__llvm_libc::ilogbf);
}

TEST_F(LlvmLibcILogbTest, SomeIntegers_ilogbf) {
  test_some_integers<float>(&__llvm_libc::ilogbf);
}

TEST_F(LlvmLibcILogbTest, SubnormalRange_ilogbf) {
  test_subnormal_range<float>(&__llvm_libc::ilogbf);
}

TEST_F(LlvmLibcILogbTest, NormalRange_ilogbf) {
  test_normal_range<float>(&__llvm_libc::ilogbf);
}
