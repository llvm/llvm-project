//===-- Unittests for stdc_first_trailing_zero_uc -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/CPP/limits.h"
#include "src/stdbit/stdc_first_trailing_zero_uc.h"
#include "test/UnitTest/Test.h"

TEST(LlvmLibcStdcFirstTrailingZeroUcTest, ALL) {
  EXPECT_EQ(LIBC_NAMESPACE::stdc_first_trailing_zero_uc(UCHAR_MAX), 0U);
}

TEST(LlvmLibcStdcFirstTrailingZeroUcTest, ZeroHot) {
  for (unsigned i = 0U; i != UCHAR_WIDTH; ++i)
    EXPECT_EQ(LIBC_NAMESPACE::stdc_first_trailing_zero_uc(~(1U << i)), i + 1);
}
