//===-- Unittests for stdc_first_leading_zero_ui --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/CPP/limits.h"
#include "src/stdbit/stdc_first_leading_zero_ui.h"
#include "test/UnitTest/Test.h"

TEST(LlvmLibcStdcFirstLeadingZeroUiTest, ALL) {
  EXPECT_EQ(LIBC_NAMESPACE::stdc_first_leading_zero_ui(UINT_MAX), 0U);
}

TEST(LlvmLibcStdcFirstLeadingZeroUiTest, ZeroHot) {
  for (unsigned i = 0U; i != UINT_WIDTH; ++i)
    EXPECT_EQ(LIBC_NAMESPACE::stdc_first_leading_zero_ui(~(1U << i)),
              UINT_WIDTH - i);
}
