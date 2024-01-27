//===-- Unittests for stdc_leading_zeros_ull
//-------------------------------===//
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

#define LZ(x) LIBC_NAMESPACE::stdc_leading_zeros_ull((x))

TEST(LlvmLibcStdcLeadingZerosUllTest, Zero) {
  EXPECT_EQ(LZ(0ULL), static_cast<unsigned long long>(ULLONG_WIDTH));
}

TEST(LlvmLibcStdcLeadingZerosUllTest, OneHot) {
  for (unsigned i = 0U; i != ULLONG_WIDTH; ++i)
    EXPECT_EQ(LZ(1ULL << i),
              static_cast<unsigned long long>(ULLONG_WIDTH - i - 1));
}
