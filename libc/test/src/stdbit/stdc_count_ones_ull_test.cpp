//===-- Unittests for stdc_count_ones_ull ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/CPP/limits.h"
#include "src/stdbit/stdc_count_ones_ull.h"
#include "test/UnitTest/Test.h"

TEST(LlvmLibcStdcCountOnesUllTest, Zero) {
  EXPECT_EQ(LIBC_NAMESPACE::stdc_count_ones_ull(0ULL), 0U);
}

TEST(LlvmLibcStdcCountOnesUllTest, Ones) {
  for (unsigned i = 0U; i != ULLONG_WIDTH; ++i)
    EXPECT_EQ(LIBC_NAMESPACE::stdc_count_ones_ull(ULLONG_MAX >> i),
              ULLONG_WIDTH - i);
}
