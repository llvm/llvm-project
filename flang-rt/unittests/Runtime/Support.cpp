//===-- unittests/Runtime/Support.cpp ---------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Runtime/support.h"
#include "tools.h"
#include "gtest/gtest.h"
#include "flang-rt/runtime/descriptor.h"

using namespace Fortran::runtime;
using Fortran::common::TypeCategory;
TEST(CopyAndUpdateDescriptor, Basic) {
  auto x{MakeArray<TypeCategory::Integer, 4>(
      std::vector<int>{2, 3}, std::vector<std::int32_t>{0, 1, 2, 3, 4, 5})};
  x->GetDimension(0).SetLowerBound(11);
  x->GetDimension(1).SetLowerBound(12);

  StaticDescriptor<2, false> statDesc;
  Descriptor &result{statDesc.descriptor()};

  RTNAME(CopyAndUpdateDescriptor)
  (result, *x, nullptr, CFI_attribute_pointer, LowerBoundModifier::Preserve);
  ASSERT_EQ(result.rank(), 2);
  EXPECT_EQ(result.raw().base_addr, x->raw().base_addr);
  EXPECT_TRUE(result.IsPointer());
  EXPECT_EQ(result.GetDimension(0).Extent(), x->GetDimension(0).Extent());
  EXPECT_EQ(
      result.GetDimension(0).LowerBound(), x->GetDimension(0).LowerBound());
  EXPECT_EQ(result.GetDimension(1).Extent(), x->GetDimension(1).Extent());
  EXPECT_EQ(
      result.GetDimension(1).LowerBound(), x->GetDimension(1).LowerBound());

  RTNAME(CopyAndUpdateDescriptor)
  (result, *x, nullptr, CFI_attribute_allocatable,
      LowerBoundModifier::SetToZeroes);
  ASSERT_EQ(result.rank(), 2);
  EXPECT_EQ(result.raw().base_addr, x->raw().base_addr);
  EXPECT_TRUE(result.IsAllocatable());
  EXPECT_EQ(result.GetDimension(0).Extent(), x->GetDimension(0).Extent());
  EXPECT_EQ(result.GetDimension(0).LowerBound(), 0);
  EXPECT_EQ(result.GetDimension(1).Extent(), x->GetDimension(1).Extent());
  EXPECT_EQ(result.GetDimension(1).LowerBound(), 0);

  RTNAME(CopyAndUpdateDescriptor)
  (result, *x, nullptr, CFI_attribute_other, LowerBoundModifier::SetToOnes);
  ASSERT_EQ(result.rank(), 2);
  EXPECT_EQ(result.raw().base_addr, x->raw().base_addr);
  EXPECT_FALSE(result.IsAllocatable());
  EXPECT_FALSE(result.IsPointer());
  EXPECT_EQ(result.GetDimension(0).Extent(), x->GetDimension(0).Extent());
  EXPECT_EQ(result.GetDimension(0).LowerBound(), 1);
  EXPECT_EQ(result.GetDimension(1).Extent(), x->GetDimension(1).Extent());
  EXPECT_EQ(result.GetDimension(1).LowerBound(), 1);
}

TEST(IsAssumedSize, Basic) {
  auto x{MakeArray<TypeCategory::Integer, 4>(
      std::vector<int>{2, 3}, std::vector<std::int32_t>{0, 1, 2, 3, 4, 5})};
  EXPECT_FALSE(RTNAME(IsAssumedSize)(*x));
  x->GetDimension(1).SetExtent(-1);
  EXPECT_TRUE(RTNAME(IsAssumedSize)(*x));
  auto scalar{MakeArray<TypeCategory::Integer, 4>(
      std::vector<int>{}, std::vector<std::int32_t>{0})};
  EXPECT_FALSE(RTNAME(IsAssumedSize)(*scalar));
}

TEST(DescriptorBytesFor, Basic) {
  for (size_t i = 0; i < Fortran::common::TypeCategory_enumSize; ++i) {
    auto tc{static_cast<TypeCategory>(i)};
    if (tc == TypeCategory::Derived)
      continue;

    auto b{Descriptor::BytesFor(tc, 4)};
    EXPECT_GT(b, 0U);
  }
}

TEST(IsContiguous, Basic) {
  // ARRAY  1 3 5
  //        2 4 6
  auto array{MakeArray<TypeCategory::Integer, 4>(
      std::vector<int>{2, 3}, std::vector<std::int32_t>{1, 2, 3, 4, 5, 6})};
  StaticDescriptor<2> sectionStaticDesc;
  Descriptor &section{sectionStaticDesc.descriptor()};
  section.Establish(array->type(), array->ElementBytes(),
      /*p=*/nullptr, /*rank=*/2);
  static const SubscriptValue lbs[]{1, 1}, ubs[]{2, 3}, strides[]{1, 2};
  const auto error{
      CFI_section(&section.raw(), &array->raw(), lbs, ubs, strides)};
  ASSERT_EQ(error, 0) << "CFI_section failed for array: " << error;

  EXPECT_TRUE(RTNAME(IsContiguous)(*array));
  EXPECT_FALSE(RTNAME(IsContiguous)(section));
  EXPECT_TRUE(RTNAME(IsContiguousUpTo)(section, 1));
  EXPECT_FALSE(RTNAME(IsContiguousUpTo)(section, 2));
}

TEST(DescriptorGetBaseAddress, Basic) {
  auto array{MakeArray<TypeCategory::Integer, 4>(
      std::vector<int>{2, 3}, std::vector<std::int32_t>{0, 1, 2, 3, 4, 5})};
  void *baseAddr = RTNAME(DescriptorGetBaseAddress)(*array);
  EXPECT_NE(baseAddr, nullptr);
  EXPECT_EQ(baseAddr, array->raw().base_addr);
}

TEST(DescriptorGetDataSizeInBytes, Basic) {
  // Test with a 2x3 integer*4 array
  auto int4Array{MakeArray<TypeCategory::Integer, 4>({2, 3})};
  EXPECT_EQ(RTNAME(DescriptorGetDataSizeInBytes)(*int4Array),
      6 * sizeof(std::int32_t));
  // Test with a 1D, 5-element real*8 array
  auto real8Array{MakeArray<TypeCategory::Real, 8>({5})};
  EXPECT_EQ(
      RTNAME(DescriptorGetDataSizeInBytes)(*real8Array), 5 * sizeof(double));
  // Test with a scalar logical*1
  auto logical1Scalar{MakeArray<TypeCategory::Logical, 1>({})};
  EXPECT_EQ(
      RTNAME(DescriptorGetDataSizeInBytes)(*logical1Scalar), 1 * sizeof(bool));
}
