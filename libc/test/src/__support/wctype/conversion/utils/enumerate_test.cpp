//===-- Unittests for slice - wctype conversion utils ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "src/__support/integer_literals.h"
#include "src/__support/wctype/conversion/utils/enumerate.h"
#include "src/__support/wctype/conversion/utils/utils.h"
#include "test/UnitTest/Test.h"

namespace LIBC_NAMESPACE_DECL {

namespace wctype_internal {

namespace conversion_utils {

TEST(LlvmLibcEnumerateTest, EnumeratesArrayElements) {
  cpp::array<int, 4> arr{10, 20, 30, 40};

  size_t expected_index = 0;
  for (auto [index, value] : enumerate(arr)) {
    EXPECT_EQ(index, expected_index);
    EXPECT_EQ(value, arr[expected_index]);
    ++expected_index;
  }

  EXPECT_EQ(expected_index, arr.size());
}

TEST(LlvmLibcEnumerateTest, EnumeratesRange) {
  Range r(3, 8); // 3,4,5,6,7

  size_t expected_index = 0;
  int expected_value = 3;

  for (auto [index, value] : enumerate(r)) {
    EXPECT_EQ(index, expected_index);
    EXPECT_EQ(value, expected_value);
    ++expected_index;
    ++expected_value;
  }

  EXPECT_EQ(expected_index, 5_u64);
}

TEST(LlvmLibcEnumerateTest, EnumerateAllowsMutation) {
  cpp::array<int, 3> arr{1, 2, 3};

  for (auto [index, value] : enumerate(arr)) {
    value += static_cast<int>(index);
  }

  EXPECT_EQ(arr[0], 1);
  EXPECT_EQ(arr[1], 3);
  EXPECT_EQ(arr[2], 5);
}

TEST(LlvmLibcEnumerateTest, ConstIterable) {
  const cpp::array<int, 3> arr{7, 8, 9};

  size_t expected_index = 0;
  for (const auto &[index, value] : enumerate(arr)) {
    EXPECT_EQ(index, expected_index++);
    EXPECT_EQ(value, arr[index]);
  }
}

} // namespace conversion_utils

} // namespace wctype_internal

} // namespace LIBC_NAMESPACE_DECL
