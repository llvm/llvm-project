//===-- Unittests for stdc_trailing_zeros_us ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/CPP/limits.h"
#include "src/stdbit/stdc_trailing_zeros_us.h"
#include "test/UnitTest/Test.h"

TEST(LlvmLibcStdcTrailingZerosUsTest, Zero) {
  EXPECT_EQ(LIBC_NAMESPACE::stdc_trailing_zeros_us(0U),
            static_cast<unsigned>(USHRT_WIDTH));
}

TEST(LlvmLibcStdcTrailingZerosUsTest, OneHot) {
  for (unsigned i = 0U; i != USHRT_WIDTH; ++i)
    EXPECT_EQ(LIBC_NAMESPACE::stdc_trailing_zeros_us(1U << i), i);
}
