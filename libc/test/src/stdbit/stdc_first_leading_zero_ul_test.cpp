//===-- Unittests for stdc_first_leading_zero_ul --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/CPP/limits.h"
#include "src/stdbit/stdc_first_leading_zero_ul.h"
#include "test/UnitTest/Test.h"

TEST(LlvmLibcStdcFirstLeadingZeroUlTest, ALL) {
  EXPECT_EQ(LIBC_NAMESPACE::stdc_first_leading_zero_ul(ULONG_MAX), 0U);
}

TEST(LlvmLibcStdcFirstLeadingZeroUlTest, ZeroHot) {
  for (unsigned i = 0U; i != ULONG_WIDTH; ++i)
    EXPECT_EQ(LIBC_NAMESPACE::stdc_first_leading_zero_ul(~(1UL << i)),
              ULONG_WIDTH - i);
}
