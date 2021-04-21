//===-- Unittests for fma ------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "FmaTest.h"

#include "src/math/fma.h"

using LlvmLibcFmaTest = FmaTestTemplate<double>;

TEST_F(LlvmLibcFmaTest, SpecialNumbers) {
  testSpecialNumbers(&__llvm_libc::fma);
}

TEST_F(LlvmLibcFmaTest, SubnormalRange) {
  testSubnormalRange(&__llvm_libc::fma);
}

TEST_F(LlvmLibcFmaTest, NormalRange) { testNormalRange(&__llvm_libc::fma); }
