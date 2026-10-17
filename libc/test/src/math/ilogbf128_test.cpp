//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unittests for the ilogbf128 function.
///
//===----------------------------------------------------------------------===//

#include "ILogbTest.h"

#include "hdr/math_macros.h"
#include "src/__support/FPUtil/FPBits.h"
#include "src/__support/FPUtil/ManipulationFunctions.h"
#include "src/__support/FPUtil/float128.h"
#include "src/math/ilogbf128.h"
#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"

#ifndef LIBC_TYPES_HAS_NATIVE_FLOAT128
using float128 = LIBC_NAMESPACE::fputil::Float128;
#endif // LIBC_TYPES_HAS_NATIVE_FLOAT128

TEST_F(LlvmLibcILogbTest, SpecialNumbers_ilogbf128) {
  test_special_numbers<float128>(&LIBC_NAMESPACE::ilogbf128);
}

TEST_F(LlvmLibcILogbTest, PowersOfTwo_ilogbf128) {
  test_powers_of_two<float128>(&LIBC_NAMESPACE::ilogbf128);
}

TEST_F(LlvmLibcILogbTest, SomeIntegers_ilogbf128) {
  test_some_integers<float128>(&LIBC_NAMESPACE::ilogbf128);
}

TEST_F(LlvmLibcILogbTest, SubnormalRange_ilogbf128) {
  test_subnormal_range<float128>(&LIBC_NAMESPACE::ilogbf128);
}

TEST_F(LlvmLibcILogbTest, NormalRange_ilogbf128) {
  test_normal_range<float128>(&LIBC_NAMESPACE::ilogbf128);
}
