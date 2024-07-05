//===-- Unittests for stdbit ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDSList-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "test/UnitTest/Test.h"

/*
 * The intent of this test is validate that the generic math macros operate as
 * intended
 */

// #include "stdbit_stub.h"

#include "include/llvm-libc-macros/generic-math-macros.h"

TEST(LlvmLibcGenericMath, TypeGenericMacroMathIsfinite) {
  EXPECT_EQ(isfinite(3.14), 0);
  EXPECT_EQ(isfinite(3.14 / 0.0), 1);
}

/*
TEST(LlvmLibcGenericMath, TypeGenericMacroMathIsinf) {

}

TEST(LlvmLibcGenericMath, TypeGenericMacroMathIsnan) {

}


TEST(LlvmLibcGenericMath, TypeGenericMacroMathSignbit) {

}*/
