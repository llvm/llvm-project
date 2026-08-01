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
  EXPECT_EQ(dest.Elements(), 4);

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
  EXPECT_EQ(dest->Elements(), 5);

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

  EXPECT_EQ(source->Elements(), 0);
  EXPECT_EQ(dest->Elements(), 0);

  // Should not crash with zero-size arrays
  RTNAME(AssignSimple)(*dest, *source, __FILE__, __LINE__);

  // Verify both still have 0 elements
  EXPECT_EQ(dest->Elements(), 0);
  EXPECT_EQ(source->Elements(), 0);

  dest->Destroy();
  source->Destroy();
}
