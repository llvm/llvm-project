//===-- Unittests for f16fmaf ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "FmaTest.h"

#include "src/math/f16fmaf.h"

using LlvmLibcF16fmafTest = FmaTestTemplate<float16, float>;

TEST_F(LlvmLibcF16fmafTest, SubnormalRange) {
  test_subnormal_range(&LIBC_NAMESPACE::f16fmaf);
}

TEST_F(LlvmLibcF16fmafTest, NormalRange) {
  test_normal_range(&LIBC_NAMESPACE::f16fmaf);
}
