//===-- Unittests for stdc_bit_ceil_ul ------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/CPP/limits.h"
#include "src/stdbit/stdc_bit_ceil_ul.h"
#include "test/UnitTest/Test.h"

TEST(LlvmLibcStdcBitceilUlTest, Zero) {
  EXPECT_EQ(LIBC_NAMESPACE::stdc_bit_ceil_ul(0UL), 1UL);
}

TEST(LlvmLibcStdcBitceilUlTest, Ones) {
  for (unsigned i = 0U; i != ULONG_WIDTH; ++i)
    EXPECT_EQ(LIBC_NAMESPACE::stdc_bit_ceil_ul(1UL << i), 1UL << i);
}

TEST(LlvmLibcStdcBitceilUlTest, OneLessThanPowsTwo) {
  for (unsigned i = 2U; i != ULONG_WIDTH; ++i)
    EXPECT_EQ(LIBC_NAMESPACE::stdc_bit_ceil_ul((1UL << i) - 1), 1UL << i);
}

TEST(LlvmLibcStdcBitceilUlTest, OneMoreThanPowsTwo) {
  for (unsigned i = 1U; i != ULONG_WIDTH - 1; ++i)
    EXPECT_EQ(LIBC_NAMESPACE::stdc_bit_ceil_ul((1UL << i) + 1), 1UL << (i + 1));
}
