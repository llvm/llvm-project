//===-- flang/unittests/Runtime/Support.cpp ----------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Runtime/support.h"
#include "gtest/gtest.h"
#include "tools.h"
#include "flang/Runtime/descriptor.h"

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
