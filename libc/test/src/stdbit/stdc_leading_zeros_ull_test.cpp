//===-- Unittests for stdc_leading_zeros_ull ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/CPP/limits.h"
#include "src/stdbit/stdc_leading_zeros_ull.h"
#include "test/UnitTest/Test.h"
#include <stddef.h>

TEST(LlvmLibcStdcLeadingZerosUllTest, Zero) {
  EXPECT_EQ(LIBC_NAMESPACE::stdc_leading_zeros_ull(0ULL),
            static_cast<unsigned>(ULLONG_WIDTH));
}

TEST(LlvmLibcStdcLeadingZerosUllTest, OneHot) {
  for (unsigned i = 0U; i != ULLONG_WIDTH; ++i)
    EXPECT_EQ(LIBC_NAMESPACE::stdc_leading_zeros_ull(1ULL << i),
              ULLONG_WIDTH - i - 1U);
}
