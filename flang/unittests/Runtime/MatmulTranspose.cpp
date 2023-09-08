//===-- flang/unittests/Runtime/MatmulTranspose.cpp -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "gtest/gtest.h"
#include "tools.h"
#include "flang/Runtime/allocatable.h"
#include "flang/Runtime/cpp-type.h"
#include "flang/Runtime/descriptor.h"
#include "flang/Runtime/matmul-transpose.h"
#include "flang/Runtime/type-code.h"

using namespace Fortran::runtime;
using Fortran::common::TypeCategory;

TEST(MatmulTranspose, Basic) {
  // X 0 1     Y 6  9     Z 6  7  8    M 0 0 1 1    V -1 -2
  //   2 3       7 10       9 10 11      0 1 0 1
  //   4 5       8 11

  auto x{MakeArray<TypeCategory::Integer, 4>(
      std::vector<int>{3, 2}, std::vector<std::int32_t>{0, 2, 4, 1, 3, 5})};
  auto y{MakeArray<TypeCategory::Integer, 2>(
      std::vector<int>{3, 2}, std::vector<std::int16_t>{6, 7, 8, 9, 10, 11})};
  auto z{MakeArray<TypeCategory::Integer, 2>(
      std::vector<int>{2, 3}, std::vector<std::int16_t>{6, 9, 7, 10, 8, 11})};
  auto m{MakeArray<TypeCategory::Integer, 2>(std::vector<int>{2, 4},
      std::vector<std::int16_t>{0, 0, 0, 1, 1, 0, 1, 1})};
  auto v{MakeArray<TypeCategory::Integer, 8>(
      std::vector<int>{2}, std::vector<std::int64_t>{-1, -2})};
  // X2  0  1     Y2 -1 -1     Z2  6  7  8
  //     2  3         6  9         9 10 11
  //     4  5         7 10        -1 -1 -1
  //    -1 -1         8 11
  auto x2{MakeArray<TypeCategory::Integer, 4>(std::vector<int>{4, 2},
      std::vector<std::int32_t>{0, 2, 4, -1, 1, 3, 5, -1})};
  auto y2{MakeArray<TypeCategory::Integer, 2>(std::vector<int>{4, 2},
      std::vector<std::int16_t>{-1, 6, 7, 8, -1, 9, 10, 11})};
  auto z2{MakeArray<TypeCategory::Integer, 2>(std::vector<int>{3, 3},
      std::vector<std::int16_t>{6, 9, -1, 7, 10, -1, 8, 11, -1})};

  StaticDescriptor<2, true> statDesc;
  Descriptor &result{statDesc.descriptor()};

  RTNAME(MatmulTranspose)(result, *x, *y, __FILE__, __LINE__);
  ASSERT_EQ(result.rank(), 2);
  EXPECT_EQ(result.GetDimension(0).LowerBound(), 1);
  EXPECT_EQ(result.GetDimension(0).Extent(), 2);
  EXPECT_EQ(result.GetDimension(1).LowerBound(), 1);
  EXPECT_EQ(result.GetDimension(1).Extent(), 2);
  ASSERT_EQ(result.type(), (TypeCode{TypeCategory::Integer, 4}));
  EXPECT_EQ(*result.ZeroBasedIndexedElement<std::int32_t>(0), 46);
  EXPECT_EQ(*result.ZeroBasedIndexedElement<std::int32_t>(1), 67);
  EXPECT_EQ(*result.ZeroBasedIndexedElement<std::int32_t>(2), 64);
  EXPECT_EQ(*result.ZeroBasedIndexedElement<std::int32_t>(3), 94);

  std::memset(
      result.raw().base_addr, 0, result.Elements() * result.ElementBytes());
  result.GetDimension(0).SetLowerBound(0);
  result.GetDimension(1).SetLowerBound(2);
  RTNAME(MatmulTransposeDirect)(result, *x, *y, __FILE__, __LINE__);
  EXPECT_EQ(*result.ZeroBasedIndexedElement<std::int32_t>(0), 46);
  EXPECT_EQ(*result.ZeroBasedIndexedElement<std::int32_t>(1), 67);
  EXPECT_EQ(*result.ZeroBasedIndexedElement<std::int32_t>(2), 64);
  EXPECT_EQ(*result.ZeroBasedIndexedElement<std::int32_t>(3), 94);
  result.Destroy();

  RTNAME(MatmulTranspose)(result, *z, *v, __FILE__, __LINE__);
  ASSERT_EQ(result.rank(), 1);
  EXPECT_EQ(result.GetDimension(0).LowerBound(), 1);
  EXPECT_EQ(result.GetDimension(0).Extent(), 3);
  ASSERT_EQ(result.type(), (TypeCode{TypeCategory::Integer, 8}));
  EXPECT_EQ(*result.ZeroBasedIndexedElement<std::int64_t>(0), -24);
  EXPECT_EQ(*result.ZeroBasedIndexedElement<std::int64_t>(1), -27);
  EXPECT_EQ(*result.ZeroBasedIndexedElement<std::int64_t>(2), -30);
  result.Destroy();

  RTNAME(MatmulTranspose)(result, *m, *z, __FILE__, __LINE__);
  ASSERT_EQ(result.rank(), 2);
  ASSERT_EQ(result.GetDimension(0).LowerBound(), 1);
  ASSERT_EQ(result.GetDimension(0).UpperBound(), 4);
  ASSERT_EQ(result.GetDimension(1).LowerBound(), 1);
  ASSERT_EQ(result.GetDimension(1).UpperBound(), 3);
  ASSERT_EQ(result.type(), (TypeCode{TypeCategory::Integer, 2}));
  EXPECT_EQ(*result.ZeroBasedIndexedElement<std::int16_t>(0), 0);
  EXPECT_EQ(*result.ZeroBasedIndexedElement<std::int16_t>(1), 9);
  EXPECT_EQ(*result.ZeroBasedIndexedElement<std::int16_t>(2), 6);
  EXPECT_EQ(*result.ZeroBasedIndexedElement<std::int16_t>(3), 15);
  EXPECT_EQ(*result.ZeroBasedIndexedElement<std::int16_t>(4), 0);
  EXPECT_EQ(*result.ZeroBasedIndexedElement<std::int16_t>(5), 10);
  EXPECT_EQ(*result.ZeroBasedIndexedElement<std::int16_t>(6), 7);
  EXPECT_EQ(*result.ZeroBasedIndexedElement<std::int16_t>(7), 17);
  EXPECT_EQ(*result.ZeroBasedIndexedElement<std::int16_t>(8), 0);
  EXPECT_EQ(*result.ZeroBasedIndexedElement<std::int16_t>(9), 11);
  EXPECT_EQ(*result.ZeroBasedIndexedElement<std::int16_t>(10), 8);
  EXPECT_EQ(*result.ZeroBasedIndexedElement<std::int16_t>(11), 19);
  result.Destroy();

  // Test non-contiguous sections.
  static constexpr int sectionRank{2};
  StaticDescriptor<sectionRank> sectionStaticDescriptorX2;
  Descriptor &sectionX2{sectionStaticDescriptorX2.descriptor()};
  sectionX2.Establish(x2->type(), x2->ElementBytes(),
      /*p=*/nullptr, /*rank=*/sectionRank);
  static const SubscriptValue lowersX2[]{1, 1}, uppersX2[]{3, 2};
  // Section of X2:
  //   +-----+
  //   | 0  1|
  //   | 2  3|
  //   | 4  5|
  //   +-----+
  //    -1 -1
  const auto errorX2{CFI_section(
      &sectionX2.raw(), &x2->raw(), lowersX2, uppersX2, /*strides=*/nullptr)};
  ASSERT_EQ(errorX2, 0) << "CFI_section failed for X2: " << errorX2;

  StaticDescriptor<sectionRank> sectionStaticDescriptorY2;
  Descriptor &sectionY2{sectionStaticDescriptorY2.descriptor()};
  sectionY2.Establish(y2->type(), y2->ElementBytes(),
      /*p=*/nullptr, /*rank=*/sectionRank);
  static const SubscriptValue lowersY2[]{2, 1};
  // Section of Y2:
  //    -1 -1
  //   +-----+
  //   | 6  0|
  //   | 7 10|
  //   | 8 11|
  //   +-----+
  const auto errorY2{CFI_section(&sectionY2.raw(), &y2->raw(), lowersY2,
      /*uppers=*/nullptr, /*strides=*/nullptr)};
  ASSERT_EQ(errorY2, 0) << "CFI_section failed for Y2: " << errorY2;

  StaticDescriptor<sectionRank> sectionStaticDescriptorZ2;
  Descriptor &sectionZ2{sectionStaticDescriptorZ2.descriptor()};
  sectionZ2.Establish(z2->type(), z2->ElementBytes(),
      /*p=*/nullptr, /*rank=*/sectionRank);
  static const SubscriptValue lowersZ2[]{1, 1}, uppersZ2[]{2, 3};
  // Section of Z2:
  //   +--------+
  //   | 6  7  8|
  //   | 9 10 11|
  //   +--------+
  //    -1 -1 -1
  const auto errorZ2{CFI_section(
      &sectionZ2.raw(), &z2->raw(), lowersZ2, uppersZ2, /*strides=*/nullptr)};
  ASSERT_EQ(errorZ2, 0) << "CFI_section failed for Z2: " << errorZ2;

  RTNAME(MatmulTranspose)(result, sectionX2, *y, __FILE__, __LINE__);
  ASSERT_EQ(result.rank(), 2);
  EXPECT_EQ(result.GetDimension(0).LowerBound(), 1);
  EXPECT_EQ(result.GetDimension(0).Extent(), 2);
  EXPECT_EQ(result.GetDimension(1).LowerBound(), 1);
  EXPECT_EQ(result.GetDimension(1).Extent(), 2);
  ASSERT_EQ(result.type(), (TypeCode{TypeCategory::Integer, 4}));
  EXPECT_EQ(*result.ZeroBasedIndexedElement<std::int32_t>(0), 46);
  EXPECT_EQ(*result.ZeroBasedIndexedElement<std::int32_t>(1), 67);
  EXPECT_EQ(*result.ZeroBasedIndexedElement<std::int32_t>(2), 64);
  EXPECT_EQ(*result.ZeroBasedIndexedElement<std::int32_t>(3), 94);
  result.Destroy();

  RTNAME(MatmulTranspose)(result, *x, sectionY2, __FILE__, __LINE__);
  ASSERT_EQ(result.rank(), 2);
  EXPECT_EQ(result.GetDimension(0).LowerBound(), 1);
  EXPECT_EQ(result.GetDimension(0).Extent(), 2);
  EXPECT_EQ(result.GetDimension(1).LowerBound(), 1);
  EXPECT_EQ(result.GetDimension(1).Extent(), 2);
  ASSERT_EQ(result.type(), (TypeCode{TypeCategory::Integer, 4}));
  EXPECT_EQ(*result.ZeroBasedIndexedElement<std::int32_t>(0), 46);
  EXPECT_EQ(*result.ZeroBasedIndexedElement<std::int32_t>(1), 67);
  EXPECT_EQ(*result.ZeroBasedIndexedElement<std::int32_t>(2), 64);
  EXPECT_EQ(*result.ZeroBasedIndexedElement<std::int32_t>(3), 94);
  result.Destroy();

  RTNAME(MatmulTranspose)(result, sectionX2, sectionY2, __FILE__, __LINE__);
  ASSERT_EQ(result.rank(), 2);
  EXPECT_EQ(result.GetDimension(0).LowerBound(), 1);
  EXPECT_EQ(result.GetDimension(0).Extent(), 2);
  EXPECT_EQ(result.GetDimension(1).LowerBound(), 1);
  EXPECT_EQ(result.GetDimension(1).Extent(), 2);
  ASSERT_EQ(result.type(), (TypeCode{TypeCategory::Integer, 4}));
  EXPECT_EQ(*result.ZeroBasedIndexedElement<std::int32_t>(0), 46);
  EXPECT_EQ(*result.ZeroBasedIndexedElement<std::int32_t>(1), 67);
  EXPECT_EQ(*result.ZeroBasedIndexedElement<std::int32_t>(2), 64);
  EXPECT_EQ(*result.ZeroBasedIndexedElement<std::int32_t>(3), 94);
  result.Destroy();

  RTNAME(MatmulTranspose)(result, sectionZ2, *v, __FILE__, __LINE__);
  ASSERT_EQ(result.rank(), 1);
  EXPECT_EQ(result.GetDimension(0).LowerBound(), 1);
  EXPECT_EQ(result.GetDimension(0).Extent(), 3);
  ASSERT_EQ(result.type(), (TypeCode{TypeCategory::Integer, 8}));
  EXPECT_EQ(*result.ZeroBasedIndexedElement<std::int64_t>(0), -24);
  EXPECT_EQ(*result.ZeroBasedIndexedElement<std::int64_t>(1), -27);
  EXPECT_EQ(*result.ZeroBasedIndexedElement<std::int64_t>(2), -30);
  result.Destroy();

  // X F F    Y F T    V T F T
  //   T F      F T
  //   T T      F F
  auto xLog{MakeArray<TypeCategory::Logical, 1>(std::vector<int>{3, 2},
      std::vector<std::uint8_t>{false, true, true, false, false, true})};
  auto yLog{MakeArray<TypeCategory::Logical, 2>(std::vector<int>{3, 2},
      std::vector<std::uint16_t>{false, false, false, true, true, false})};
  auto vLog{MakeArray<TypeCategory::Logical, 1>(
      std::vector<int>{3}, std::vector<std::uint16_t>{true, false, true})};
  RTNAME(MatmulTranspose)(result, *xLog, *yLog, __FILE__, __LINE__);
  ASSERT_EQ(result.rank(), 2);
  EXPECT_EQ(result.GetDimension(0).LowerBound(), 1);
  EXPECT_EQ(result.GetDimension(0).Extent(), 2);
  EXPECT_EQ(result.GetDimension(1).LowerBound(), 1);
  EXPECT_EQ(result.GetDimension(1).Extent(), 2);
  ASSERT_EQ(result.type(), (TypeCode{TypeCategory::Logical, 2}));
  EXPECT_FALSE(
      static_cast<bool>(*result.ZeroBasedIndexedElement<std::uint16_t>(0)));
  EXPECT_FALSE(
      static_cast<bool>(*result.ZeroBasedIndexedElement<std::uint16_t>(1)));
  EXPECT_TRUE(
      static_cast<bool>(*result.ZeroBasedIndexedElement<std::uint16_t>(2)));
  EXPECT_FALSE(
      static_cast<bool>(*result.ZeroBasedIndexedElement<std::uint16_t>(3)));

  RTNAME(MatmulTranspose)(result, *yLog, *vLog, __FILE__, __LINE__);
  ASSERT_EQ(result.rank(), 1);
  EXPECT_EQ(result.GetDimension(0).LowerBound(), 1);
  EXPECT_EQ(result.GetDimension(0).Extent(), 2);
  ASSERT_EQ(result.type(), (TypeCode{TypeCategory::Logical, 2}));
  EXPECT_FALSE(
      static_cast<bool>(*result.ZeroBasedIndexedElement<std::uint16_t>(0)));
  EXPECT_TRUE(
      static_cast<bool>(*result.ZeroBasedIndexedElement<std::uint16_t>(1)));
}
