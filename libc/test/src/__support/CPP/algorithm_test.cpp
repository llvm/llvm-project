//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unit tests for algorithm.
///
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

TEST(LlvmLibcAlgorithmTest, MinMax) {
  EXPECT_EQ(min(1, 2), 1);
  EXPECT_EQ(min(2, 1), 1);
  EXPECT_EQ(min(-5, -3), -5);
  EXPECT_EQ(min(42, 42), 42);

  EXPECT_EQ(max(1, 2), 2);
  EXPECT_EQ(max(2, 1), 2);
  EXPECT_EQ(max(-5, -3), -3);
  EXPECT_EQ(max(42, 42), 42);

  constexpr int a = min(10, 20);
  static_assert(a == 10);
  constexpr int b = max(10, 20);
  static_assert(b == 20);
}

TEST(LlvmLibcAlgorithmTest, Clamp) {
  EXPECT_EQ(clamp(5, 0, 10), 5);
  EXPECT_EQ(clamp(-5, 0, 10), 0);
  EXPECT_EQ(clamp(15, 0, 10), 10);
  EXPECT_EQ(clamp(0, 0, 10), 0);
  EXPECT_EQ(clamp(10, 0, 10), 10);

  // Custom comparator (greater)
  auto greater = [](int x, int y) { return x > y; };
  EXPECT_EQ(clamp(5, 10, 0, greater), 5);
  EXPECT_EQ(clamp(15, 10, 0, greater), 10);
  EXPECT_EQ(clamp(-5, 10, 0, greater), 0);

  // Constexpr check
  constexpr int c1 = clamp(5, 0, 10);
  static_assert(c1 == 5);
  constexpr int c2 = clamp(-5, 0, 10);
  static_assert(c2 == 0);
  constexpr int c3 = clamp(15, 0, 10);
  static_assert(c3 == 10);
}

} // namespace cpp
} // namespace LIBC_NAMESPACE_DECL
