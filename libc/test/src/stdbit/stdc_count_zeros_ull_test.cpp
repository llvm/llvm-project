//===-- Unittests for stdc_count_zeros_ull --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/CPP/limits.h"
#include "src/stdbit/stdc_count_zeros_ull.h"
#include "test/UnitTest/Test.h"

TEST(LlvmLibcStdcCountZerosUllTest, Zero) {
  EXPECT_EQ(LIBC_NAMESPACE::stdc_count_zeros_ull(0U),
            static_cast<unsigned>(ULLONG_WIDTH));
}

TEST(LlvmLibcStdcCountZerosUllTest, Ones) {
  for (unsigned i = 0U; i != ULLONG_WIDTH; ++i)
    EXPECT_EQ(LIBC_NAMESPACE::stdc_count_zeros_ull(ULLONG_MAX >> i), i);
}
