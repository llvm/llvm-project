//===-- Unittests for stdc_leading_zeros_us -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/CPP/limits.h"
#include "src/stdbit/stdc_leading_zeros_us.h"
#include "test/UnitTest/Test.h"

#define LZ(x) LIBC_NAMESPACE::stdc_leading_zeros_us((x))

TEST(LlvmLibcStdcLeadingZerosUsTest, Zero) {
  EXPECT_EQ(LZ(0U), static_cast<unsigned short>(USHRT_WIDTH));
}

TEST(LlvmLibcStdcLeadingZerosUsTest, OneHot) {
  for (unsigned i = 0U; i != USHRT_WIDTH; ++i)
    EXPECT_EQ(LZ(1U << i), static_cast<unsigned short>(USHRT_WIDTH - i - 1));
}
