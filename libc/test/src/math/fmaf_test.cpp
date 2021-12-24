//===-- Unittests for fmaf ------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "FmaTest.h"

#include "src/math/fmaf.h"

using LlvmLibcFmaTest = FmaTestTemplate<float>;

TEST_F(LlvmLibcFmaTest, SpecialNumbers) {
  test_special_numbers(&__llvm_libc::fmaf);
}

TEST_F(LlvmLibcFmaTest, SubnormalRange) {
  test_subnormal_range(&__llvm_libc::fmaf);
}

TEST_F(LlvmLibcFmaTest, NormalRange) { test_normal_range(&__llvm_libc::fmaf); }
