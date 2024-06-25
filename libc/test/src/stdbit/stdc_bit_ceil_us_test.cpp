//===-- Unittests for stdc_bit_ceil_us ------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/CPP/limits.h"
#include "src/stdbit/stdc_bit_ceil_us.h"
#include "test/UnitTest/Test.h"

TEST(LlvmLibcStdcBitceilUsTest, Zero) {
  EXPECT_EQ(LIBC_NAMESPACE::stdc_bit_ceil_us(0U),
            static_cast<unsigned short>(1));
}

TEST(LlvmLibcStdcBitceilUsTest, Ones) {
  for (unsigned i = 0U; i != USHRT_WIDTH; ++i)
    EXPECT_EQ(LIBC_NAMESPACE::stdc_bit_ceil_us(1U << i),
              static_cast<unsigned short>(1U << i));
}

TEST(LlvmLibcStdcBitceilUsTest, OneLessThanPowsTwo) {
  for (unsigned i = 2U; i != USHRT_WIDTH; ++i)
    EXPECT_EQ(LIBC_NAMESPACE::stdc_bit_ceil_us((1U << i) - 1),
              static_cast<unsigned short>(1U << i));
}

TEST(LlvmLibcStdcBitceilUsTest, OneMoreThanPowsTwo) {
  for (unsigned i = 1U; i != USHRT_WIDTH - 1; ++i)
    EXPECT_EQ(LIBC_NAMESPACE::stdc_bit_ceil_us((1U << i) + 1),
              static_cast<unsigned short>(1U << (i + 1)));
}
