//===-- Unittests for stdckdint -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDSList-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "test/UnitTest/Test.h"

#include "include/llvm-libc-macros/stdckdint-macros.h"

TEST(LlvmLibcStdCkdIntTest, Add) {
  int result;
  ASSERT_FALSE(ckd_add(&result, 1, 2));
  ASSERT_EQ(result, 3);
  ASSERT_TRUE(ckd_add(&result, INT_MAX, 1));
}

TEST(LlvmLibcStdCkdIntTest, Sub) {
  int result;
  ASSERT_FALSE(ckd_sub(&result, 3, 2));
  ASSERT_EQ(result, 1);
  ASSERT_TRUE(ckd_sub(&result, INT_MIN, 1));
}

TEST(LlvmLibcStdCkdIntTest, Mul) {
  int result;
  ASSERT_FALSE(ckd_mul(&result, 2, 3));
  ASSERT_EQ(result, 6);
  ASSERT_TRUE(ckd_mul(&result, INT_MAX, 2));
}
