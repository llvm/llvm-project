//===-- Unittests for stdc_bit_ceil_ui ------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/CPP/limits.h"
#include "src/stdbit/stdc_bit_ceil_ui.h"
#include "test/UnitTest/Test.h"

TEST(LlvmLibcStdcBitceilUiTest, Zero) {
  EXPECT_EQ(LIBC_NAMESPACE::stdc_bit_ceil_ui(0U), 1U);
}

TEST(LlvmLibcStdcBitceilUiTest, Ones) {
  for (unsigned i = 0U; i != UINT_WIDTH; ++i)
    EXPECT_EQ(LIBC_NAMESPACE::stdc_bit_ceil_ui(1U << i), 1U << i);
}

TEST(LlvmLibcStdcBitceilUiTest, OneLessThanPowsTwo) {
  for (unsigned i = 2U; i != UINT_WIDTH; ++i)
    EXPECT_EQ(LIBC_NAMESPACE::stdc_bit_ceil_ui((1U << i) - 1), 1U << i);
}

TEST(LlvmLibcStdcBitceilUiTest, OneMoreThanPowsTwo) {
  for (unsigned i = 1U; i != UINT_WIDTH - 1; ++i)
    EXPECT_EQ(LIBC_NAMESPACE::stdc_bit_ceil_ui((1U << i) + 1), 1U << (i + 1));
}
