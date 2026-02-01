//===-- Unittests for slice - wctype conversion utils ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/wctype/conversion/utils/slice.h"
#include "test/UnitTest/Test.h"

namespace LIBC_NAMESPACE_DECL {

using wctype_internal::conversion_utils::Ordering;
using wctype_internal::conversion_utils::Slice;

struct IntComparator {
  int target;
  constexpr Ordering operator()(int value) const {
    if (value < target)
      return Ordering::Less;
    if (value > target)
      return Ordering::Greater;
    return Ordering::Equal;
  }
};

TEST(LlvmLibcSliceBinarySearchTest, EmptySlice) {
  Slice<int> s;

  auto result = s.binary_search_by(IntComparator{10});

  EXPECT_EQ(result.error(), static_cast<size_t>(0));
  EXPECT_FALSE(result.has_value());
}

TEST(LlvmLibcSliceBinarySearchTest, SingleElementFound) {
  int data[] = {42};
  Slice<int> s(data, 1);

  auto result = s.binary_search_by(IntComparator{42});

  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(*result, static_cast<size_t>(0));
}

TEST(LlvmLibcSliceBinarySearchTest, SingleElementNotFound) {
  int data[] = {42};
  Slice<int> s(data, 1);

  auto result = s.binary_search_by(IntComparator{10});

  EXPECT_FALSE(result.has_value());
  EXPECT_EQ(result.error(), static_cast<size_t>(0));
}

TEST(LlvmLibcSliceBinarySearchTest, FoundInMiddle) {
  int data[] = {1, 3, 5, 7, 9};
  Slice<int> s(data, 5);

  auto result = s.binary_search_by(IntComparator{7});

  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(*result, static_cast<size_t>(3));
}

TEST(LlvmLibcSliceBinarySearchTest, InsertPositionMiddle) {
  int data[] = {1, 3, 5, 7, 9};
  Slice<int> s(data, 5);

  auto result = s.binary_search_by(IntComparator{6});

  EXPECT_FALSE(result.has_value());
  EXPECT_EQ(result.error(), static_cast<size_t>(3));
}

TEST(LlvmLibcSliceBinarySearchTest, InsertAtBeginning) {
  int data[] = {10, 20, 30};
  Slice<int> s(data, 3);

  auto result = s.binary_search_by(IntComparator{5});

  EXPECT_FALSE(result.has_value());
  EXPECT_EQ(result.error(), static_cast<size_t>(0));
}

TEST(LlvmLibcSliceBinarySearchTest, InsertAtEnd) {
  int data[] = {10, 20, 30};
  Slice<int> s(data, 3);

  auto result = s.binary_search_by(IntComparator{40});

  EXPECT_FALSE(result.has_value());
  EXPECT_EQ(result.error(), static_cast<size_t>(3));
}

TEST(LlvmLibcSliceBinarySearchTest, DuplicateValues) {
  int data[] = {1, 2, 2, 2, 3};
  Slice<int> s(data, 5);

  auto result = s.binary_search_by(IntComparator{2});

  ASSERT_TRUE(result.has_value());
  EXPECT_GE(*result, static_cast<size_t>(1));
  EXPECT_LE(*result, static_cast<size_t>(3));
}

TEST(LlvmLibcSliceRangeTest, BasicRange) {
  int data[] = {0, 1, 2, 3, 4};
  Slice<int> s(data, 5);

  Slice<int> sub = s.slice_form_range(1, 4);

  ASSERT_EQ(sub.size(), static_cast<size_t>(3));
  EXPECT_EQ(sub[0], 1);
  EXPECT_EQ(sub[1], 2);
  EXPECT_EQ(sub[2], 3);
}

TEST(LlvmLibcSliceContainsTest, ContainsElement) {
  int data[] = {5, 10, 15};
  Slice<int> s(data, 3);

  EXPECT_TRUE(s.contains(10));
  EXPECT_FALSE(s.contains(7));
}

TEST(LlvmLibcSliceCopyTest, SameSizeCopy) {
  int src_data[] = {1, 2, 3};
  int dst_data[] = {0, 0, 0};

  Slice<int> src(src_data, 3);
  Slice<int> dst(dst_data, 3);

  dst.copy_from_slice(src);

  EXPECT_EQ(dst[0], 1);
  EXPECT_EQ(dst[1], 2);
  EXPECT_EQ(dst[2], 3);
}

TEST(LlvmLibcSliceCopyTest, SourceLargerThanDestination) {
  int src_data[] = {1, 2, 3, 4};
  int dst_data[] = {0, 0};

  Slice<int> src(src_data, 4);
  Slice<int> dst(dst_data, 2);

  dst.copy_from_slice(src);

  EXPECT_EQ(dst[0], 1);
  EXPECT_EQ(dst[1], 2);
}

TEST(LlvmLibcSliceCopyTest, DestinationLargerThanSource) {
  int src_data[] = {1, 2};
  int dst_data[] = {0, 0, 0};

  Slice<int> src(src_data, 2);
  Slice<int> dst(dst_data, 3);

  dst.copy_from_slice(src);

  EXPECT_EQ(dst[0], 1);
  EXPECT_EQ(dst[1], 2);
  EXPECT_EQ(dst[2], 0);
}

} // namespace LIBC_NAMESPACE_DECL
