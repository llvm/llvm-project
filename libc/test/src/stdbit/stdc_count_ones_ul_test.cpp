//===-- Unittests for stdc_count_ones_ul ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/CPP/limits.h"
#include "src/stdbit/stdc_count_ones_ul.h"
#include "test/UnitTest/Test.h"

TEST(LlvmLibcStdcCountOnesUlTest, Zero) {
  EXPECT_EQ(LIBC_NAMESPACE::stdc_count_ones_ul(0UL), 0U);
}

TEST(LlvmLibcStdcCountOnesUlTest, Ones) {
  for (unsigned i = 0U; i != ULONG_WIDTH; ++i)
    EXPECT_EQ(LIBC_NAMESPACE::stdc_count_ones_ul(ULONG_MAX >> i),
              ULONG_WIDTH - i);
}
