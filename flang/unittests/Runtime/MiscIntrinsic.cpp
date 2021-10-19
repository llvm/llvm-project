//===-- flang/unittests/RuntimeGTest/MiscIntrinsic.cpp ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "gtest/gtest.h"
#include "tools.h"
#include "flang/Runtime//misc-intrinsic.h"
#include "flang/Runtime/allocatable.h"
#include "flang/Runtime/cpp-type.h"
#include "flang/Runtime/descriptor.h"

using namespace Fortran::runtime;

// TRANSFER examples from Fortran 2018

TEST(MiscIntrinsic, TransferScalar) {
  StaticDescriptor<2, true, 2> staticDesc[2];
  auto &result{staticDesc[0].descriptor()};
  auto source{MakeArray<TypeCategory::Integer, 4>(
      std::vector<int>{}, std::vector<std::int32_t>{1082130432})};
  auto &mold{staticDesc[1].descriptor()};
  mold.Establish(TypeCategory::Real, 4, nullptr, 0);
  RTNAME(Transfer)(result, *source, mold, __FILE__, __LINE__);
  EXPECT_EQ(result.rank(), 0);
  EXPECT_EQ(result.type().raw(), (TypeCode{TypeCategory::Real, 4}.raw()));
  EXPECT_EQ(*result.OffsetElement<float>(), 4.0);
  result.Destroy();
}

TEST(MiscIntrinsic, TransferMold) {
  StaticDescriptor<2, true, 2> staticDesc[2];
  auto &result{staticDesc[0].descriptor()};
  auto source{MakeArray<TypeCategory::Real, 4>(
      std::vector<int>{3}, std::vector<float>{1.1F, 2.2F, 3.3F})};
  auto &mold{staticDesc[1].descriptor()};
  SubscriptValue extent[1]{1};
  mold.Establish(TypeCategory::Complex, 4, nullptr, 1, extent);
  RTNAME(Transfer)(result, *source, mold, __FILE__, __LINE__);
  EXPECT_EQ(result.rank(), 1);
  EXPECT_EQ(result.GetDimension(0).LowerBound(), 1);
  EXPECT_EQ(result.GetDimension(0).Extent(), 2);
  EXPECT_EQ(result.type().raw(), (TypeCode{TypeCategory::Complex, 4}.raw()));
  EXPECT_EQ(result.OffsetElement<float>()[0], 1.1F);
  EXPECT_EQ(result.OffsetElement<float>()[1], 2.2F);
  EXPECT_EQ(result.OffsetElement<float>()[2], 3.3F);
  EXPECT_EQ(result.OffsetElement<float>()[3], 0.0F);
  result.Destroy();
}

TEST(MiscIntrinsic, TransferSize) {
  StaticDescriptor<2, true, 2> staticDesc[2];
  auto &result{staticDesc[0].descriptor()};
  auto source{MakeArray<TypeCategory::Real, 4>(
      std::vector<int>{3}, std::vector<float>{1.1F, 2.2F, 3.3F})};
  auto &mold{staticDesc[1].descriptor()};
  SubscriptValue extent[1]{1};
  mold.Establish(TypeCategory::Complex, 4, nullptr, 1, extent);
  RTNAME(TransferSize)(result, *source, mold, __FILE__, __LINE__, 1);
  EXPECT_EQ(result.rank(), 1);
  EXPECT_EQ(result.GetDimension(0).LowerBound(), 1);
  EXPECT_EQ(result.GetDimension(0).Extent(), 1);
  EXPECT_EQ(result.type().raw(), (TypeCode{TypeCategory::Complex, 4}.raw()));
  EXPECT_EQ(result.OffsetElement<float>()[0], 1.1F);
  EXPECT_EQ(result.OffsetElement<float>()[1], 2.2F);
  result.Destroy();
}
