//===-- Unittests for stdc_bit_floor_ui -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/CPP/limits.h"
#include "src/stdbit/stdc_bit_floor_ui.h"
#include "test/UnitTest/Test.h"

TEST(LlvmLibcStdcBitfloorUiTest, Zero) {
  EXPECT_EQ(LIBC_NAMESPACE::stdc_bit_floor_ui(0U), 0U);
}

TEST(LlvmLibcStdcBitfloorUiTest, Ones) {
  for (unsigned i = 0U; i != INT_WIDTH; ++i)
    EXPECT_EQ(LIBC_NAMESPACE::stdc_bit_floor_ui(UINT_MAX >> i),
              1U << (UINT_WIDTH - i - 1));
}
