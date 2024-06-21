//===-- Unittests for stdc_bit_ceil_ull -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/CPP/limits.h"
#include "src/stdbit/stdc_bit_ceil_ull.h"
#include "test/UnitTest/Test.h"

TEST(LlvmLibcStdcBitceilUllTest, Zero) {
  EXPECT_EQ(LIBC_NAMESPACE::stdc_bit_ceil_ull(0ULL), 1ULL);
}

TEST(LlvmLibcStdcBitceilUllTest, Ones) {
  for (unsigned i = 0U; i != ULLONG_WIDTH; ++i)
    EXPECT_EQ(LIBC_NAMESPACE::stdc_bit_ceil_ull(1ULL << i), 1ULL << i);
}

TEST(LlvmLibcStdcBitceilUllTest, OneLessThanPowsTwo) {
  for (unsigned i = 2U; i != ULLONG_WIDTH; ++i)
    EXPECT_EQ(LIBC_NAMESPACE::stdc_bit_ceil_ull((1ULL << i) - 1), 1ULL << i);
}

TEST(LlvmLibcStdcBitceilUllTest, OneMoreThanPowsTwo) {
  for (unsigned i = 1U; i != ULLONG_WIDTH - 1; ++i)
    EXPECT_EQ(LIBC_NAMESPACE::stdc_bit_ceil_ull((1ULL << i) + 1),
              1ULL << (i + 1));
}
