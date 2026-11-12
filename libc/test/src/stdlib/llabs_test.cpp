//===-- Unittests for llabs -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/stdlib/llabs.h"
#include "test/UnitTest/Test.h"

TEST(LlvmLibcLlabsTest, Zero) { EXPECT_EQ(LIBC_NAMESPACE::llabs(0ll), 0ll); }

TEST(LlvmLibcLlabsTest, Positive) {
  EXPECT_EQ(LIBC_NAMESPACE::llabs(1ll), 1ll);
}

TEST(LlvmLibcLlabsTest, Negative) {
  EXPECT_EQ(LIBC_NAMESPACE::llabs(-1ll), 1ll);
}
