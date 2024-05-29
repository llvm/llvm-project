//===-- Unittests for stdc_bit_floor_uc -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/CPP/limits.h"
#include "src/stdbit/stdc_bit_floor_uc.h"
#include "test/UnitTest/Test.h"

TEST(LlvmLibcStdcBitfloorUcTest, Zero) {
  EXPECT_EQ(LIBC_NAMESPACE::stdc_bit_floor_uc(0U),
            static_cast<unsigned char>(0));
}

TEST(LlvmLibcStdcBitfloorUcTest, Ones) {
  for (unsigned i = 0U; i != UCHAR_WIDTH; ++i)
    EXPECT_EQ(LIBC_NAMESPACE::stdc_bit_floor_uc(UCHAR_MAX >> i),
              static_cast<unsigned char>(1 << (UCHAR_WIDTH - i - 1)));
}
