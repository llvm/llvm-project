//===-- Unittests for stdc_bit_floor_ull ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/CPP/limits.h"
#include "src/stdbit/stdc_bit_floor_ull.h"
#include "test/UnitTest/Test.h"

TEST(LlvmLibcStdcBitfloorUllTest, Zero) {
  EXPECT_EQ(LIBC_NAMESPACE::stdc_bit_floor_ull(0ULL), 0ULL);
}

TEST(LlvmLibcStdcBitfloorUllTest, Ones) {
  for (unsigned i = 0U; i != ULLONG_WIDTH; ++i)
    EXPECT_EQ(LIBC_NAMESPACE::stdc_bit_floor_ull(ULLONG_MAX >> i),
              1ULL << (ULLONG_WIDTH - i - 1));
}
