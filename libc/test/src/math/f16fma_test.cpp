//===-- Unittests for f16fma ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "FmaTest.h"

#include "src/math/f16fma.h"

using LlvmLibcF16fmaTest = FmaTestTemplate<float16, double>;

TEST_F(LlvmLibcF16fmaTest, SubnormalRange) {
  test_subnormal_range(&LIBC_NAMESPACE::f16fma);
}

TEST_F(LlvmLibcF16fmaTest, NormalRange) {
  test_normal_range(&LIBC_NAMESPACE::f16fma);
}
