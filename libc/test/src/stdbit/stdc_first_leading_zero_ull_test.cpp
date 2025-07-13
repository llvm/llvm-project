//===-- Unittests for stdc_first_leading_zero_ull -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/CPP/limits.h"
#include "src/stdbit/stdc_first_leading_zero_ull.h"
#include "test/UnitTest/Test.h"

TEST(LlvmLibcStdcFirstLeadingZeroUllTest, ALL) {
  EXPECT_EQ(LIBC_NAMESPACE::stdc_first_leading_zero_ull(ULLONG_MAX), 0U);
}

TEST(LlvmLibcStdcFirstLeadingZeroUllTest, ZeroHot) {
  for (unsigned i = 0U; i != ULLONG_WIDTH; ++i)
    EXPECT_EQ(LIBC_NAMESPACE::stdc_first_leading_zero_ull(~(1ULL << i)),
              ULLONG_WIDTH - i);
}
