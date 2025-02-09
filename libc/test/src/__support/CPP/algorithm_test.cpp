//===-- Unittests for Algorithm -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/CPP/algorithm.h"
#include "src/__support/CPP/array.h"
#include "src/__support/macros/config.h"
#include "test/UnitTest/Test.h"

// TODO(https://github.com/llvm/llvm-project/issues/94066): Add unittests for
// the remaining algorithm functions.
namespace LIBC_NAMESPACE_DECL {
namespace cpp {

TEST(LlvmLibcAlgorithmTest, FindIfNot) {
  array<int, 4> nums{1, 2, 3, 4};
  EXPECT_EQ(find_if_not(nums.begin(), nums.end(), [](int i) { return i == 0; }),
            nums.begin());
  EXPECT_EQ(find_if_not(nums.begin(), nums.end(), [](int i) { return i == 1; }),
            nums.begin() + 1);
  EXPECT_EQ(find_if_not(nums.begin(), nums.end(), [](int i) { return i < 4; }),
            nums.begin() + 3);
  EXPECT_EQ(find_if_not(nums.begin(), nums.end(), [](int i) { return i < 5; }),
            nums.end());

  EXPECT_EQ(
      find_if_not(nums.begin() + 1, nums.end(), [](int i) { return i == 0; }),
      nums.begin() + 1);
  EXPECT_EQ(
      find_if_not(nums.begin(), nums.begin(), [](int i) { return i == 0; }),
      nums.begin());
}

TEST(LlvmLibcAlgorithmTest, AllOf) {
  array<int, 4> nums{1, 2, 3, 4};
  EXPECT_TRUE(all_of(nums.begin(), nums.end(), [](int i) { return i < 5; }));
  EXPECT_FALSE(all_of(nums.begin(), nums.end(), [](int i) { return i < 4; }));
  EXPECT_TRUE(
      all_of(nums.begin(), nums.begin() + 3, [](int i) { return i < 4; }));
  EXPECT_TRUE(
      all_of(nums.begin() + 1, nums.end(), [](int i) { return i > 1; }));
  EXPECT_TRUE(all_of(nums.begin(), nums.begin(), [](int i) { return i < 0; }));
}

} // namespace cpp
} // namespace LIBC_NAMESPACE_DECL
