//===-- flang/unittests/RuntimeGTest/Inquiry.cpp -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Runtime/inquiry.h"
#include "gtest/gtest.h"
#include "tools.h"
#include "flang/Runtime/type-code.h"

using namespace Fortran::runtime;
using Fortran::common::TypeCategory;

TEST(Inquiry, Lbound) {
  // ARRAY  1 3 5
  //        2 4 6
  auto array{MakeArray<TypeCategory::Integer, 4>(
      std::vector<int>{2, 3}, std::vector<std::int32_t>{1, 2, 3, 4, 5, 6})};
  array->GetDimension(0).SetLowerBound(0);
  array->GetDimension(1).SetLowerBound(-1);

  EXPECT_EQ(RTNAME(LboundDim)(*array, 1, __FILE__, __LINE__), std::int64_t{0});
  EXPECT_EQ(RTNAME(LboundDim)(*array, 2, __FILE__, __LINE__), std::int64_t{-1});
}

TEST(Inquiry, Ubound) {
  // ARRAY  1 3 5
  //        2 4 6
  auto array{MakeArray<TypeCategory::Integer, 4>(
      std::vector<int>{2, 3}, std::vector<std::int32_t>{1, 2, 3, 4, 5, 6})};
  array->GetDimension(0).SetLowerBound(1000);
  array->GetDimension(1).SetLowerBound(1);
  StaticDescriptor<2, true> statDesc;

  int intValue{1};
  SubscriptValue extent[]{2};
  Descriptor &result{statDesc.descriptor()};
  result.Establish(TypeCategory::Integer, /*KIND=*/4,
      static_cast<void *>(&intValue), 1, extent, CFI_attribute_pointer);
  RTNAME(Ubound)(result, *array, /*KIND=*/4, __FILE__, __LINE__);
  EXPECT_EQ(result.rank(), 1);
  EXPECT_EQ(result.type().raw(), (TypeCode{TypeCategory::Integer, 4}.raw()));
  // The lower bound of UBOUND's result array is always 1
  EXPECT_EQ(result.GetDimension(0).LowerBound(), 1);
  EXPECT_EQ(result.GetDimension(0).Extent(), 2);
  EXPECT_EQ(*result.ZeroBasedIndexedElement<std::int32_t>(0), 1001);
  EXPECT_EQ(*result.ZeroBasedIndexedElement<std::int32_t>(1), 3);
  result.Destroy();

  result = statDesc.descriptor();
  result.Establish(TypeCategory::Integer, /*KIND=*/1,
      static_cast<void *>(&intValue), 1, extent, CFI_attribute_pointer);
  RTNAME(Ubound)(result, *array, /*KIND=*/1, __FILE__, __LINE__);
  EXPECT_EQ(result.rank(), 1);
  EXPECT_EQ(result.type().raw(), (TypeCode{TypeCategory::Integer, 1}.raw()));
  // The lower bound of UBOUND's result array is always 1
  EXPECT_EQ(result.GetDimension(0).LowerBound(), 1);
  EXPECT_EQ(result.GetDimension(0).Extent(), 2);
  EXPECT_EQ(*result.ZeroBasedIndexedElement<std::int8_t>(0), -23);
  EXPECT_EQ(*result.ZeroBasedIndexedElement<std::int8_t>(1), 3);
  result.Destroy();
}

TEST(Inquiry, Size) {
  // ARRAY  1 3 5
  //        2 4 6
  auto array{MakeArray<TypeCategory::Integer, 4>(
      std::vector<int>{2, 3}, std::vector<std::int32_t>{1, 2, 3, 4, 5, 6})};
  array->GetDimension(0).SetLowerBound(0); // shouldn't matter
  array->GetDimension(1).SetLowerBound(-1);

  EXPECT_EQ(RTNAME(SizeDim)(*array, 1, __FILE__, __LINE__), std::int64_t{2});
  EXPECT_EQ(RTNAME(SizeDim)(*array, 2, __FILE__, __LINE__), std::int64_t{3});
  EXPECT_EQ(RTNAME(Size)(*array, __FILE__, __LINE__), std::int64_t{6});
}
