//===-- Unittests for stdc_bit_width_ui -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/CPP/limits.h"
#include "src/stdbit/stdc_bit_width_ui.h"
#include "test/UnitTest/Test.h"

TEST(LlvmLibcStdcBitWidthUiTest, Zero) {
  EXPECT_EQ(LIBC_NAMESPACE::stdc_bit_width_ui(0U), 0U);
}

TEST(LlvmLibcStdcBitWidthUiTest, Ones) {
  for (unsigned i = 0U; i != UINT_WIDTH; ++i)
    EXPECT_EQ(LIBC_NAMESPACE::stdc_bit_width_ui(UINT_MAX >> i), UINT_WIDTH - i);
}
