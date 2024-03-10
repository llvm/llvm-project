//===-- Unittests for stdc_bit_ceil_uc ------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/CPP/limits.h"
#include "src/stdbit/stdc_bit_ceil_uc.h"
#include "test/UnitTest/Test.h"

TEST(LlvmLibcStdcBitceilUcTest, Zero) {
  EXPECT_EQ(LIBC_NAMESPACE::stdc_bit_ceil_uc(0U),
            static_cast<unsigned char>(1));
}

TEST(LlvmLibcStdcBitceilUcTest, Ones) {
  for (unsigned i = 0U; i != UCHAR_WIDTH; ++i)
    EXPECT_EQ(LIBC_NAMESPACE::stdc_bit_ceil_uc(1U << i),
              static_cast<unsigned char>(1U << i));
}

TEST(LlvmLibcStdcBitceilUcTest, OneLessThanPowsTwo) {
  for (unsigned i = 2U; i != UCHAR_WIDTH; ++i)
    EXPECT_EQ(LIBC_NAMESPACE::stdc_bit_ceil_uc((1U << i) - 1),
              static_cast<unsigned char>(1U << i));
}

TEST(LlvmLibcStdcBitceilUcTest, OneMoreThanPowsTwo) {
  for (unsigned i = 1U; i != UCHAR_WIDTH - 1; ++i)
    EXPECT_EQ(LIBC_NAMESPACE::stdc_bit_ceil_uc((1U << i) + 1),
              static_cast<unsigned char>(1U << (i + 1)));
}
