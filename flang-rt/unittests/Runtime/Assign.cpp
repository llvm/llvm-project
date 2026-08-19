//===-- unittests/Runtime/Assign.cpp ------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Runtime/assign.h"
#include "tools.h"
#include "gtest/gtest.h"
#include <vector>

using namespace Fortran::runtime;
using Fortran::common::TypeCategory;

TEST(Assign, RTNAME(CopyInAssign)) {
  // contiguous -> contiguous copy in
  auto intArray{MakeArray<TypeCategory::Integer, 1>(
      std::vector<int>{2, 3}, std::vector<int>{1, 2, 3, 4, 5, 6}, sizeof(int))};
  StaticDescriptor<2> staticIntResult;
  Descriptor &intResult{staticIntResult.descriptor()};

  RTNAME(CopyInAssign(intResult, *intArray));
  ASSERT_TRUE(intResult.IsAllocated());
  ASSERT_TRUE(intResult.IsContiguous());
  ASSERT_EQ(intResult.type(), intArray->type());
  ASSERT_EQ(intResult.ElementBytes(), sizeof(int));
  EXPECT_EQ(intResult.GetDimension(0).LowerBound(), 1);
  EXPECT_EQ(intResult.GetDimension(0).Extent(), 2);
  EXPECT_EQ(intResult.GetDimension(1).LowerBound(), 1);
  EXPECT_EQ(intResult.GetDimension(1).Extent(), 3);
  int expected[6] = {1, 2, 3, 4, 5, 6};
  EXPECT_EQ(
      std::memcmp(intResult.OffsetElement<int>(0), expected, 6 * sizeof(int)),
      0);
  intResult.Destroy();

  // discontiguous -> contiguous rank-1 copy in
  intArray = MakeArray<TypeCategory::Integer, 1>(std::vector<int>{8},
      std::vector<int>{1, 2, 3, 4, 5, 6, 7, 8}, sizeof(int));
  StaticDescriptor<1> staticIntResultStrided;
  Descriptor &intResultStrided{staticIntResultStrided.descriptor()};
  // Treat the descriptor as a strided array of 4
  intArray->GetDimension(0).SetByteStride(sizeof(int) * 2);
  intArray->GetDimension(0).SetExtent(4);
  RTNAME(CopyInAssign(intResultStrided, *intArray));

  int expectedStrided[4] = {1, 3, 5, 7};
  EXPECT_EQ(std::memcmp(intResultStrided.OffsetElement<int>(0), expectedStrided,
                4 * sizeof(int)),
      0);

  intResultStrided.Destroy();
}

TEST(AssignSimple, AliasedReverseStride) {
  // Test aliasing detection with reverse-stride copy: a(5:1:-1) = a(1:5)
  // This exercises the MayAlias() detection and temporary buffer path.
  // Without temp buffer, the element-wise copy would corrupt data by
  // overwriting source elements before they're read.

  // Create backing storage as a C++ array
  int data[5] = {1, 2, 3, 4, 5};
  constexpr int elementBytes = sizeof(int);
  TypeCode intType{TypeCategory::Integer, 4};

  // Create source descriptor: forward view (1:5)
  StaticDescriptor<1> staticSource;
  Descriptor &source{staticSource.descriptor()};
  SubscriptValue extent[1]{5};
  source.Establish(intType, elementBytes, data, 1, extent);
  source.GetDimension(0).SetLowerBound(1);

  // Create dest descriptor: reverse view (5:1:-1) of same memory
  StaticDescriptor<1> staticDest;
  Descriptor &dest{staticDest.descriptor()};
  dest.Establish(
      intType, elementBytes, &data[4], 1, extent); // Start at last element
  dest.GetDimension(0).SetLowerBound(1);
  dest.GetDimension(0).SetByteStride(-elementBytes); // Negative stride

  RTNAME(AssignSimple)(dest, source, __FILE__, __LINE__);

  // Verify reverse copy succeeded.
  // The backing array should now be [5,4,3,2,1] (reversed from [1,2,3,4,5])
  int expected[5] = {5, 4, 3, 2, 1};
  EXPECT_EQ(std::memcmp(data, expected, 5 * sizeof(int)), 0);
}

TEST(AssignSimple, ReallocateUnallocated) {
  // Test allocatable reallocation from unallocated state
  StaticDescriptor<1> staticDest;
  Descriptor &dest{staticDest.descriptor()};
  dest.Establish(TypeCode{TypeCategory::Integer, 4}, sizeof(int), nullptr, 1,
      nullptr, CFI_attribute_allocatable);
  dest.GetDimension(0).SetBounds(1, 0);
  // dest is now unallocated

  auto source{MakeArray<TypeCategory::Integer, 4>(
      std::vector<int>{4}, std::vector<int>{10, 20, 30, 40}, sizeof(int))};

  EXPECT_FALSE(dest.IsAllocated());

  RTNAME(AssignSimple)(dest, *source, __FILE__, __LINE__);

  // Verify dest is now allocated with correct shape and data
  EXPECT_TRUE(dest.IsAllocated());
  EXPECT_EQ(dest.rank(), 1);
  EXPECT_EQ(dest.GetDimension(0).LowerBound(), 1);
  EXPECT_EQ(dest.GetDimension(0).Extent(), 4);
  EXPECT_EQ(dest.Elements(), 4U);

  int expected[4] = {10, 20, 30, 40};
  EXPECT_EQ(
      std::memcmp(dest.OffsetElement<int>(0), expected, 4 * sizeof(int)), 0);

  // Verify source unchanged
  EXPECT_EQ(
      std::memcmp(source->OffsetElement<int>(0), expected, 4 * sizeof(int)), 0);

  dest.Destroy();
  source->Destroy();
}

TEST(AssignSimple, ReallocateShapeMismatch) {
  // Test allocatable reallocation when shape (extent) differs
  auto dest{MakeArray<TypeCategory::Integer, 4>(
      std::vector<int>{3}, std::vector<int>{1, 2, 3}, sizeof(int))};

  auto source{MakeArray<TypeCategory::Integer, 4>(
      std::vector<int>{5}, std::vector<int>{10, 20, 30, 40, 50}, sizeof(int))};

  EXPECT_TRUE(dest->IsAllocated());
  EXPECT_EQ(dest->GetDimension(0).Extent(), 3);

  RTNAME(AssignSimple)(*dest, *source, __FILE__, __LINE__);

  // Verify dest was reallocated with new extent matching source
  EXPECT_TRUE(dest->IsAllocated());
  EXPECT_EQ(dest->rank(), 1);
  EXPECT_EQ(dest->GetDimension(0).LowerBound(), 1);
  EXPECT_EQ(dest->GetDimension(0).Extent(), 5);
  EXPECT_EQ(dest->Elements(), 5U);

  int expected[5] = {10, 20, 30, 40, 50};
  EXPECT_EQ(
      std::memcmp(dest->OffsetElement<int>(0), expected, 5 * sizeof(int)), 0);

  // Verify source unchanged
  EXPECT_EQ(
      std::memcmp(source->OffsetElement<int>(0), expected, 5 * sizeof(int)), 0);

  dest->Destroy();
  source->Destroy();
}

TEST(AssignSimple, NonContiguousToContiguous) {
  // Test non-contiguous source (strided) to contiguous destination
  // Pattern: take every other element from an 8-element array
  auto source{MakeArray<TypeCategory::Integer, 4>(std::vector<int>{8},
      std::vector<int>{1, 2, 3, 4, 5, 6, 7, 8}, sizeof(int))};

  // Make source non-contiguous: stride=2*sizeof(int), extent=4
  // This gives us elements [1, 3, 5, 7] from the backing array
  source->GetDimension(0).SetByteStride(sizeof(int) * 2);
  source->GetDimension(0).SetExtent(4);
  EXPECT_FALSE(source->IsContiguous());

  auto dest{MakeArray<TypeCategory::Integer, 4>(
      std::vector<int>{4}, std::vector<int>{0, 0, 0, 0}, sizeof(int))};
  EXPECT_TRUE(dest->IsContiguous());

  RTNAME(AssignSimple)(*dest, *source, __FILE__, __LINE__);

  // Verify dest has strided elements from source
  int expected[4] = {1, 3, 5, 7};
  EXPECT_EQ(
      std::memcmp(dest->OffsetElement<int>(0), expected, 4 * sizeof(int)), 0);
  EXPECT_TRUE(dest->IsContiguous());

  dest->Destroy();
  source->Destroy();
}

TEST(AssignSimple, ZeroSizeArray) {
  // Test zero-size array edge case
  auto source{MakeArray<TypeCategory::Integer, 4>(
      std::vector<int>{0}, std::vector<int>{}, sizeof(int))};

  auto dest{MakeArray<TypeCategory::Integer, 4>(
      std::vector<int>{0}, std::vector<int>{}, sizeof(int))};

  EXPECT_EQ(source->Elements(), 0U);
  EXPECT_EQ(dest->Elements(), 0U);

  // Should not crash with zero-size arrays
  RTNAME(AssignSimple)(*dest, *source, __FILE__, __LINE__);

  // Verify both still have 0 elements
  EXPECT_EQ(dest->Elements(), 0U);
  EXPECT_EQ(source->Elements(), 0U);

  dest->Destroy();
  source->Destroy();
}

TEST(AssignSimple, AliasedOverlappingSection) {
  // Test aliasing with overlapping array sections: a(3:7) = a(1:5)
  // This is a classic case where the destination partially overlaps the source.
  // Without a temporary buffer, elements would be corrupted as the copy
  // progresses.
  //
  // Example:
  // Initial:  [1, 2, 3, 4, 5, 6, 7, 8]
  // a(3:7) = a(1:5) should produce [1, 2, 1, 2, 3, 4, 5, 8]

  int data[8] = {1, 2, 3, 4, 5, 6, 7, 8};
  constexpr int elementBytes = sizeof(int);
  TypeCode intType{TypeCategory::Integer, 4};

  // Source descriptor: a(1:5) - elements at indices 0-4
  StaticDescriptor<1> staticSource;
  Descriptor &source{staticSource.descriptor()};
  SubscriptValue extent[1]{5};
  source.Establish(intType, elementBytes, data, 1, extent);
  source.GetDimension(0).SetLowerBound(1);

  // Dest descriptor: a(3:7) - elements at indices 2-6 (same backing array)
  StaticDescriptor<1> staticDest;
  Descriptor &dest{staticDest.descriptor()};
  dest.Establish(intType, elementBytes, &data[2], 1, extent);
  dest.GetDimension(0).SetLowerBound(1);

  RTNAME(AssignSimple)(dest, source, __FILE__, __LINE__);

  // Expected result: [1, 2, 1, 2, 3, 4, 5, 8]
  // Positions 3-7 (indices 2-6) should now contain values from positions 1-5
  int expected[8] = {1, 2, 1, 2, 3, 4, 5, 8};
  EXPECT_EQ(std::memcmp(data, expected, 8 * sizeof(int)), 0);
}

TEST(AssignSimple, AliasedTwoDimensionalReverse) {
  // Test aliasing in 2D array with column reversal: a(:, 2:1:-1) = a(:, 1:2)
  // This tests that aliasing detection works across multiple dimensions.
  //
  // Initial array (3x2, column-major):
  //   Column 1  Column 2
  //   [1]       [4]
  //   [2]       [5]
  //   [3]       [6]
  //
  // After a(:, 2:1:-1) = a(:, 1:2), should be:
  //   [4]  [1]
  //   [5]  [2]
  //   [6]  [3]
  //
  // Backing storage (column-major): [1,2,3,4,5,6] -> [4,5,6,1,2,3]

  int data[6] = {1, 2, 3, 4, 5, 6};
  constexpr int elementBytes = sizeof(int);
  TypeCode intType{TypeCategory::Integer, 4};

  // Source descriptor: a(:, 1:2) - all rows, columns 1-2 (forward)
  StaticDescriptor<2> staticSource;
  Descriptor &source{staticSource.descriptor()};
  SubscriptValue extent[2]{3, 2}; // 3 rows, 2 columns
  source.Establish(intType, elementBytes, data, 2, extent);
  source.GetDimension(0).SetLowerBound(1);
  source.GetDimension(0).SetByteStride(elementBytes); // Rows are contiguous
  source.GetDimension(1).SetLowerBound(1);
  source.GetDimension(1).SetByteStride(3 * elementBytes); // Column stride

  // Dest descriptor: a(:, 2:1:-1) - all rows, columns 2-1 (reverse)
  StaticDescriptor<2> staticDest;
  Descriptor &dest{staticDest.descriptor()};
  dest.Establish(
      intType, elementBytes, &data[3], 2, extent); // Start at column 2
  dest.GetDimension(0).SetLowerBound(1);
  dest.GetDimension(0).SetByteStride(elementBytes);
  dest.GetDimension(1).SetLowerBound(1);
  dest.GetDimension(1).SetByteStride(-3 * elementBytes); // Negative stride

  RTNAME(AssignSimple)(dest, source, __FILE__, __LINE__);

  // Expected: columns swapped
  // Column-major storage: [4,5,6,1,2,3]
  int expected[6] = {4, 5, 6, 1, 2, 3};
  EXPECT_EQ(std::memcmp(data, expected, 6 * sizeof(int)), 0);
}

TEST(AssignSimple, AliasedReallocatableSelfAssign) {
  // Test aliasing when LHS is allocatable and gets reallocated during a
  // self-assignment with a different shape: a = a(1:3)
  //
  // This is tricky because:
  // 1. Aliasing is detected (LHS and RHS point to same memory)
  // 2. Shapes differ, so reallocation is needed
  // 3. Deallocating LHS would free RHS memory
  // 4. Temp buffer must be created BEFORE deallocation

  // Initial array: [10, 20, 30, 40, 50]
  auto dest{MakeArray<TypeCategory::Integer, 4>(
      std::vector<int>{5}, std::vector<int>{10, 20, 30, 40, 50}, sizeof(int))};

  // Create source descriptor pointing to first 3 elements of dest
  StaticDescriptor<1> staticSource;
  Descriptor &source{staticSource.descriptor()};
  SubscriptValue extent[1]{3};
  source.Establish(TypeCode{TypeCategory::Integer, 4}, sizeof(int),
      dest->OffsetElement(), 1, extent);
  source.GetDimension(0).SetLowerBound(1);

  EXPECT_TRUE(dest->IsAllocated());
  EXPECT_EQ(dest->GetDimension(0).Extent(), 5);

  // Self-assign with different shape: dest = dest(1:3)
  RTNAME(AssignSimple)(*dest, source, __FILE__, __LINE__);

  // Verify dest was reallocated to size 3 with correct values
  EXPECT_TRUE(dest->IsAllocated());
  EXPECT_EQ(dest->GetDimension(0).Extent(), 3);

  int expected[3] = {10, 20, 30};
  EXPECT_EQ(
      std::memcmp(dest->OffsetElement<int>(0), expected, 3 * sizeof(int)), 0);

  dest->Destroy();
}

TEST(AssignSimple, AliasedNonContiguousToNonContiguous) {
  // Test aliasing where both LHS and RHS are non-contiguous strided views
  // a(6:2:-2) = a(1:5:2)
  //
  // This ensures the temporary buffer path works correctly when BOTH sides
  // are non-contiguous, requiring element-wise copy in both directions.
  //
  // Initial: [1, 2, 3, 4, 5, 6, 7, 8]
  // Source: a(1:5:2) = indices [0, 2, 4] = [1, 3, 5]
  // Dest: a(6:2:-2) = indices [5, 3, 1] = [6, 4, 2] (reverse)
  //
  // After assignment: [1, 5, 3, 3, 5, 1, 7, 8]

  int data[8] = {1, 2, 3, 4, 5, 6, 7, 8};
  constexpr int elementBytes = sizeof(int);
  TypeCode intType{TypeCategory::Integer, 4};

  // Source: a(1:5:2) - indices [0, 2, 4] forward, stride 2
  StaticDescriptor<1> staticSource;
  Descriptor &source{staticSource.descriptor()};
  SubscriptValue extent[1]{3};
  source.Establish(intType, elementBytes, &data[0], 1, extent);
  source.GetDimension(0).SetLowerBound(1);
  source.GetDimension(0).SetByteStride(2 * elementBytes);
  EXPECT_FALSE(source.IsContiguous());

  // Dest: a(6:2:-2) - indices [5, 3, 1] reverse, stride -2
  StaticDescriptor<1> staticDest;
  Descriptor &dest{staticDest.descriptor()};
  dest.Establish(
      intType, elementBytes, &data[5], 1, extent); // Start at index 5
  dest.GetDimension(0).SetLowerBound(1);
  dest.GetDimension(0).SetByteStride(-2 * elementBytes);
  EXPECT_FALSE(dest.IsContiguous());

  RTNAME(AssignSimple)(dest, source, __FILE__, __LINE__);

  // Expected: dest positions [5,3,1] get source values [1,3,5]
  // Result: [1, 5, 3, 3, 5, 1, 7, 8]
  int expected[8] = {1, 5, 3, 3, 5, 1, 7, 8};
  EXPECT_EQ(std::memcmp(data, expected, 8 * sizeof(int)), 0);
}
