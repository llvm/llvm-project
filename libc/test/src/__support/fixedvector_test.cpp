//===-- Unittests for FixedVector -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/fixedvector.h"
#include "utils/UnitTest/Test.h"

TEST(LlvmLibcFixedVectorTest, PushAndPop) {
  __llvm_libc::FixedVector<int, 20> fixed_vector;
  ASSERT_TRUE(fixed_vector.empty());
  for (int i = 0; i < 20; i++)
    ASSERT_TRUE(fixed_vector.push_back(i));
  ASSERT_FALSE(fixed_vector.empty());
  ASSERT_FALSE(fixed_vector.push_back(123));
  for (int i = 20; i > 0; --i) {
    ASSERT_EQ(fixed_vector.back(), i - 1);
    ASSERT_TRUE(fixed_vector.pop_back());
  }
  ASSERT_FALSE(fixed_vector.pop_back());
  ASSERT_TRUE(fixed_vector.empty());
}

TEST(LlvmLibcFixedVectorTest, Reset) {
  __llvm_libc::FixedVector<int, 20> fixed_vector;
  ASSERT_TRUE(fixed_vector.empty());
  for (int i = 0; i < 20; i++)
    ASSERT_TRUE(fixed_vector.push_back(i));
  ASSERT_FALSE(fixed_vector.empty());
  fixed_vector.reset();
  ASSERT_TRUE(fixed_vector.empty());
}

TEST(LlvmLibcFixedVectorTest, Destroy) {
  __llvm_libc::FixedVector<int, 20> fixed_vector;
  ASSERT_TRUE(fixed_vector.empty());
  for (int i = 0; i < 20; i++)
    ASSERT_TRUE(fixed_vector.push_back(i));
  ASSERT_FALSE(fixed_vector.empty());
  __llvm_libc::FixedVector<int, 20>::destroy(&fixed_vector);
  ASSERT_TRUE(fixed_vector.empty());
}
