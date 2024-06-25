//===-- Unittests for stdc_trailing_zeros_ui ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/CPP/limits.h"
#include "src/stdbit/stdc_trailing_zeros_ui.h"
#include "test/UnitTest/Test.h"

TEST(LlvmLibcStdcTrailingZerosUiTest, Zero) {
  EXPECT_EQ(LIBC_NAMESPACE::stdc_trailing_zeros_ui(0U),
            static_cast<unsigned>(UINT_WIDTH));
}

TEST(LlvmLibcStdcTrailingZerosUiTest, OneHot) {
  for (unsigned i = 0U; i != UINT_WIDTH; ++i)
    EXPECT_EQ(LIBC_NAMESPACE::stdc_trailing_zeros_ui(1U << i), i);
}
