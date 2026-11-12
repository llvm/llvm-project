//===-- Unittests for stdc_has_single_bit_ull -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/CPP/limits.h"
#include "src/stdbit/stdc_has_single_bit_ull.h"
#include "test/UnitTest/Test.h"

TEST(LlvmLibcStdcHasSingleBitUllTest, Zero) {
  EXPECT_EQ(LIBC_NAMESPACE::stdc_has_single_bit_ull(0U), false);
}

TEST(LlvmLibcStdcHasSingleBitUllTest, OneHot) {
  for (unsigned i = 0U; i != ULLONG_WIDTH; ++i)
    EXPECT_EQ(LIBC_NAMESPACE::stdc_has_single_bit_ull(1ULL << i), true);
}
