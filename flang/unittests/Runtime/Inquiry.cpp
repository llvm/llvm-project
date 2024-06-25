//===-- flang/unittests/Runtime/Inquiry.cpp -------------------------------===//
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

TEST(Inquiry, LboundDim) {
  // ARRAY  1 3 5
  //        2 4 6
  auto array{MakeArray<TypeCategory::Integer, 4>(
      std::vector<int>{2, 3}, std::vector<std::int32_t>{1, 2, 3, 4, 5, 6})};
  array->GetDimension(0).SetLowerBound(0);
  array->GetDimension(1).SetLowerBound(-1);

  EXPECT_EQ(RTNAME(LboundDim)(*array, 1, __FILE__, __LINE__), std::int64_t{0});
  EXPECT_EQ(RTNAME(LboundDim)(*array, 2, __FILE__, __LINE__), std::int64_t{-1});
}

TEST(Inquiry, Lbound) {
  // ARRAY  1 3 5
  //        2 4 6
  auto array{MakeArray<TypeCategory::Integer, 4>(
      std::vector<int>{2, 3}, std::vector<std::int32_t>{1, 2, 3, 4, 5, 6})};
  array->GetDimension(0).SetLowerBound(0);
  array->GetDimension(1).SetLowerBound(-1);

  // LBOUND(ARRAY, KIND=1)
  auto int8Result{
      MakeArray<TypeCategory::Integer, 1>(std::vector<int>{array->rank()},
          std::vector<std::int8_t>(array->rank(), 0))};
  RTNAME(Lbound)
  (int8Result->raw().base_addr, *array, /*KIND=*/1, __FILE__, __LINE__);
  EXPECT_EQ(*int8Result->ZeroBasedIndexedElement<std::int8_t>(0), 0);
  EXPECT_EQ(*int8Result->ZeroBasedIndexedElement<std::int8_t>(1), -1);

  // LBOUND(ARRAY, KIND=4)
  auto int32Result{
      MakeArray<TypeCategory::Integer, 4>(std::vector<int>{array->rank()},
          std::vector<std::int32_t>(array->rank(), 0))};
  RTNAME(Lbound)
  (int32Result->raw().base_addr, *array, /*KIND=*/4, __FILE__, __LINE__);
  EXPECT_EQ(*int32Result->ZeroBasedIndexedElement<std::int32_t>(0), 0);
  EXPECT_EQ(*int32Result->ZeroBasedIndexedElement<std::int32_t>(1), -1);

  // LBOUND(ARRAY, KIND=8)
  auto int64Result{
      MakeArray<TypeCategory::Integer, 8>(std::vector<int>{array->rank()},
          std::vector<std::int64_t>(array->rank(), 0))};
  RTNAME(Lbound)
  (int64Result->raw().base_addr, *array, /*KIND=*/8, __FILE__, __LINE__);
  EXPECT_EQ(*int64Result->ZeroBasedIndexedElement<std::int64_t>(0), 0);
  EXPECT_EQ(*int64Result->ZeroBasedIndexedElement<std::int64_t>(1), -1);
}

TEST(Inquiry, Ubound) {
  // ARRAY  1 3 5
  //        2 4 6
  auto array{MakeArray<TypeCategory::Integer, 4>(
      std::vector<int>{2, 3}, std::vector<std::int32_t>{1, 2, 3, 4, 5, 6})};
  array->GetDimension(0).SetLowerBound(1000);
  array->GetDimension(1).SetLowerBound(1);

  // UBOUND(ARRAY, KIND=1)
  auto int8Result{
      MakeArray<TypeCategory::Integer, 1>(std::vector<int>{array->rank()},
          std::vector<std::int8_t>(array->rank(), 0))};
  RTNAME(Ubound)
  (int8Result->raw().base_addr, *array, /*KIND=*/1, __FILE__, __LINE__);
  EXPECT_EQ(*int8Result->ZeroBasedIndexedElement<std::int8_t>(0), -23);
  EXPECT_EQ(*int8Result->ZeroBasedIndexedElement<std::int8_t>(1), 3);

  // UBOUND(ARRAY, KIND=4)
  auto int32Result{
      MakeArray<TypeCategory::Integer, 4>(std::vector<int>{array->rank()},
          std::vector<std::int32_t>(array->rank(), 0))};
  RTNAME(Ubound)
  (int32Result->raw().base_addr, *array, /*KIND=*/4, __FILE__, __LINE__);
  EXPECT_EQ(*int32Result->ZeroBasedIndexedElement<std::int32_t>(0), 1001);
  EXPECT_EQ(*int32Result->ZeroBasedIndexedElement<std::int32_t>(1), 3);

  // UBOUND(ARRAY, KIND=8)
  auto int64Result{
      MakeArray<TypeCategory::Integer, 8>(std::vector<int>{array->rank()},
          std::vector<std::int64_t>(array->rank(), 0))};
  RTNAME(Ubound)
  (int64Result->raw().base_addr, *array, /*KIND=*/8, __FILE__, __LINE__);
  EXPECT_EQ(*int64Result->ZeroBasedIndexedElement<std::int64_t>(0), 1001);
  EXPECT_EQ(*int64Result->ZeroBasedIndexedElement<std::int64_t>(1), 3);
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

TEST(Inquiry, Shape) {
  // ARRAY  1 3 5
  //        2 4 6
  auto array{MakeArray<TypeCategory::Integer, 4>(
      std::vector<int>{2, 3}, std::vector<std::int32_t>{1, 2, 3, 4, 5, 6})};

  // SHAPE(ARRAY, KIND=1)
  auto int8Result{
      MakeArray<TypeCategory::Integer, 1>(std::vector<int>{array->rank()},
          std::vector<std::int8_t>(array->rank(), 0))};
  RTNAME(Shape)
  (int8Result->raw().base_addr, *array, /*KIND=*/1, __FILE__, __LINE__);
  EXPECT_EQ(*int8Result->ZeroBasedIndexedElement<std::int8_t>(0), 2);
  EXPECT_EQ(*int8Result->ZeroBasedIndexedElement<std::int8_t>(1), 3);

  // SHAPE(ARRAY, KIND=4)
  auto int32Result{
      MakeArray<TypeCategory::Integer, 4>(std::vector<int>{array->rank()},
          std::vector<std::int32_t>(array->rank(), 0))};
  RTNAME(Shape)
  (int32Result->raw().base_addr, *array, /*KIND=*/4, __FILE__, __LINE__);
  EXPECT_EQ(*int32Result->ZeroBasedIndexedElement<std::int32_t>(0), 2);
  EXPECT_EQ(*int32Result->ZeroBasedIndexedElement<std::int32_t>(1), 3);

  // SHAPE(ARRAY, KIND=8)
  auto int64Result{
      MakeArray<TypeCategory::Integer, 8>(std::vector<int>{array->rank()},
          std::vector<std::int64_t>(array->rank(), 0))};
  RTNAME(Shape)
  (int64Result->raw().base_addr, *array, /*KIND=*/8, __FILE__, __LINE__);
  EXPECT_EQ(*int64Result->ZeroBasedIndexedElement<std::int64_t>(0), 2);
  EXPECT_EQ(*int64Result->ZeroBasedIndexedElement<std::int64_t>(1), 3);
}
