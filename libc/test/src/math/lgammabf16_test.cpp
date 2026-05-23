//===-- Exhaustive test for lgammabf16 ------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/FPUtil/bfloat16.h"
#include "src/math/lgammabf16.h"
#include "test/src/math/exhaustive/exhaustive_test.h"

namespace mpfr = LIBC_NAMESPACE::testing::mpfr;

using LlvmLibcLgammabf16ExhaustiveTest =
    LlvmLibcUnaryOpExhaustiveMathTest<LIBC_NAMESPACE::fputil::BFloat16,
                                      mpfr::Operation::Lgamma,
                                      LIBC_NAMESPACE::lgammabf16>;

TEST_F(LlvmLibcLgammabf16ExhaustiveTest, PositiveRange) {
  test_full_range_all_roundings(0x0000U, 0x7F80U);
}

TEST_F(LlvmLibcLgammabf16ExhaustiveTest, NegativeRange) {
  test_full_range_all_roundings(0x8000U, 0xFF80U);
}
