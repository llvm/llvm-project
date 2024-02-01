//===-- Unittests for stdc_leading_ones_us --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/CPP/limits.h"
#include "src/stdbit/stdc_leading_ones_us.h"
#include "test/UnitTest/Test.h"

TEST(LlvmLibcStdcLeadingOnesUsTest, All) {
  EXPECT_EQ(LIBC_NAMESPACE::stdc_leading_ones_us(USHRT_MAX),
            static_cast<unsigned short>(USHRT_WIDTH));
}

TEST(LlvmLibcStdcLeadingOnesUsTest, ZeroHot) {
  for (unsigned i = 0U; i != USHRT_WIDTH; ++i)
    EXPECT_EQ(LIBC_NAMESPACE::stdc_leading_ones_us(~(1U << i)),
              static_cast<unsigned short>(USHRT_WIDTH - i - 1));
}
