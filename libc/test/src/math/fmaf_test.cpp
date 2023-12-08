//===-- Unittests for fmaf ------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "FmaTest.h"

#include "src/math/fmaf.h"

using LlvmLibcFmafTest = FmaTestTemplate<float>;

TEST_F(LlvmLibcFmafTest, SpecialNumbers) {
  test_special_numbers(&LIBC_NAMESPACE::fmaf);
}

TEST_F(LlvmLibcFmafTest, SubnormalRange) {
  test_subnormal_range(&LIBC_NAMESPACE::fmaf);
}

TEST_F(LlvmLibcFmafTest, NormalRange) {
  test_normal_range(&LIBC_NAMESPACE::fmaf);
}
