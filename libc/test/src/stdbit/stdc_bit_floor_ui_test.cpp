//===-- Unittests for stdc_bit_floor_ui -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/CPP/limits.h"
#include "src/stdbit/stdc_bit_floor_ui.h"
#include "test/UnitTest/Test.h"

TEST(LlvmLibcStdcBitfloorUiTest, Zero) {
  EXPECT_EQ(LIBC_NAMESPACE::stdc_bit_floor_ui(0U), 0U);
}

TEST(LlvmLibcStdcBitfloorUiTest, Ones) {
  for (unsigned i = 0U; i != LIBC_NAMESPACE::cpp::numeric_limits<int>::digits;
       ++i)
    EXPECT_EQ(
        LIBC_NAMESPACE::stdc_bit_floor_ui(
            LIBC_NAMESPACE::cpp::numeric_limits<unsigned int>::max() >> i),
        1U << (LIBC_NAMESPACE::cpp::numeric_limits<unsigned int>::digits - i -
               1));
}
