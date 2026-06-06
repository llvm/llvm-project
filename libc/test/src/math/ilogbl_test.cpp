//===-- Unittests for ilogbl ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ILogbTest.h"

#include "hdr/math_macros.h"
#include "src/__support/FPUtil/FPBits.h"
#include "src/__support/FPUtil/ManipulationFunctions.h"
#include "src/math/ilogbl.h"
#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"

TEST_F(LlvmLibcILogbTest, SpecialNumbers_ilogbl) {
  test_special_numbers<long double>(&LIBC_NAMESPACE::ilogbl);
}

TEST_F(LlvmLibcILogbTest, PowersOfTwo_ilogbl) {
  test_powers_of_two<long double>(&LIBC_NAMESPACE::ilogbl);
}

TEST_F(LlvmLibcILogbTest, SomeIntegers_ilogbl) {
  test_some_integers<long double>(&LIBC_NAMESPACE::ilogbl);
}

TEST_F(LlvmLibcILogbTest, SubnormalRange_ilogbl) {
  test_subnormal_range<long double>(&LIBC_NAMESPACE::ilogbl);
}

TEST_F(LlvmLibcILogbTest, NormalRange_ilogbl) {
  test_normal_range<long double>(&LIBC_NAMESPACE::ilogbl);
}
