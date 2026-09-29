//===-- Unittests for hypot -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "HypotTest.h"

#include "src/math/hypot.h"

using LlvmLibcHypotTest = HypotTestTemplate<double>;

TEST_F(LlvmLibcHypotTest, SpecialNumbers) {
  test_special_numbers(&LIBC_NAMESPACE::hypot);

#ifndef LIBC_MATH_HAS_SKIP_ACCURATE_PASS
  // Test denormal inputs.
  EXPECT_FP_EQ(
      0x0.c0bf7399534e3p-1022,
      LIBC_NAMESPACE::hypot(0x0.2c2671b3c16b3p-1022, 0x0.bb9f8fecba9adp-1022));
  EXPECT_FP_EQ(
      0x1.6a09e667f3bcbp-1022,
      LIBC_NAMESPACE::hypot(0x0.fffffffffffffp-1022, 0x0.fffffffffffffp-1022));
  EXPECT_FP_EQ(
      0x0.bffffb1b06483p-1022,
      LIBC_NAMESPACE::hypot(0x0.603e52daf0bfdp-1022, -0x0.a622d0a9a433bp-1022));
#endif // LIBC_MATH_HAS_SKIP_ACCURATE_PASS
}
