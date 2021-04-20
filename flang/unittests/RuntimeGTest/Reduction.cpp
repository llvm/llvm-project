//===-- flang/unittests/RuntimeGTest/Reductions.cpp -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../../runtime/reduction.h"
#include "gtest/gtest.h"
#include "tools.h"
#include "../../runtime/allocatable.h"
#include "../../runtime/cpp-type.h"
#include "../../runtime/descriptor.h"
#include "../../runtime/type-code.h"
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

using namespace Fortran::runtime;
using Fortran::common::TypeCategory;

TEST(Reductions, SumInt4) {
  auto array{MakeArray<TypeCategory::Integer, 4>(
      std::vector<int>{2, 3}, std::vector<std::int32_t>{1, 2, 3, 4, 5, 6})};
  std::int32_t sum{RTNAME(SumInteger4)(*array, __FILE__, __LINE__)};
  EXPECT_EQ(sum, 21) << sum;
}

TEST(Reductions, DimMaskProductInt4) {
  std::vector<int> shape{2, 3};
  auto array{MakeArray<TypeCategory::Integer, 4>(
      shape, std::vector<std::int32_t>{1, 2, 3, 4, 5, 6})};
  auto mask{MakeArray<TypeCategory::Logical, 1>(
      shape, std::vector<bool>{true, false, false, true, true, true})};
  StaticDescriptor<1> statDesc;
  Descriptor &prod{statDesc.descriptor()};
  RTNAME(ProductDim)(prod, *array, 1, __FILE__, __LINE__, &*mask);
  EXPECT_EQ(prod.rank(), 1);
  EXPECT_EQ(prod.GetDimension(0).LowerBound(), 1);
  EXPECT_EQ(prod.GetDimension(0).Extent(), 3);
  EXPECT_EQ(*prod.ZeroBasedIndexedElement<std::int32_t>(0), 1);
  EXPECT_EQ(*prod.ZeroBasedIndexedElement<std::int32_t>(1), 4);
  EXPECT_EQ(*prod.ZeroBasedIndexedElement<std::int32_t>(2), 30);
  EXPECT_EQ(RTNAME(SumInteger4)(prod, __FILE__, __LINE__), 35);
  prod.Destroy();
}

TEST(Reductions, DoubleMaxMin) {
  std::vector<int> shape{3, 4, 2}; // rows, columns, planes
  //   0  -3   6  -9     12 -15  18 -21
  //  -1   4  -7  10    -13  16 -19  22
  //   2  -5   8 -11     14 -17  20  22   <- note last two are equal to test
  //   BACK=
  auto array{MakeArray<TypeCategory::Real, 8>(shape,
      std::vector<double>{0, -1, 2, -3, 4, -5, 6, -7, 8, -9, 10, -11, 12, -13,
          14, -15, 16, -17, 18, -19, 20, -21, 22, 22})};
  EXPECT_EQ(RTNAME(MaxvalReal8)(*array, __FILE__, __LINE__), 22.0);
  EXPECT_EQ(RTNAME(MinvalReal8)(*array, __FILE__, __LINE__), -21.0);
  StaticDescriptor<2> statDesc;
  Descriptor &loc{statDesc.descriptor()};
  RTNAME(Maxloc)
  (loc, *array, /*KIND=*/8, __FILE__, __LINE__, /*MASK=*/nullptr,
      /*BACK=*/false);
  EXPECT_EQ(loc.rank(), 1);
  EXPECT_EQ(loc.type().raw(), (TypeCode{TypeCategory::Integer, 8}.raw()));
  EXPECT_EQ(loc.GetDimension(0).LowerBound(), 1);
  EXPECT_EQ(loc.GetDimension(0).Extent(), 3);
  EXPECT_EQ(
      *array->Element<double>(loc.ZeroBasedIndexedElement<SubscriptValue>(0)),
      22.0);
  EXPECT_EQ(*loc.ZeroBasedIndexedElement<std::int64_t>(0), 2);
  EXPECT_EQ(*loc.ZeroBasedIndexedElement<std::int64_t>(1), 4);
  EXPECT_EQ(*loc.ZeroBasedIndexedElement<std::int64_t>(2), 2);
  loc.Destroy();
  RTNAME(Maxloc)
  (loc, *array, /*KIND=*/8, __FILE__, __LINE__, /*MASK=*/nullptr,
      /*BACK=*/true);
  EXPECT_EQ(loc.rank(), 1);
  EXPECT_EQ(loc.type().raw(), (TypeCode{TypeCategory::Integer, 8}.raw()));
  EXPECT_EQ(loc.GetDimension(0).LowerBound(), 1);
  EXPECT_EQ(loc.GetDimension(0).Extent(), 3);
  EXPECT_EQ(
      *array->Element<double>(loc.ZeroBasedIndexedElement<SubscriptValue>(0)),
      22.0);
  EXPECT_EQ(*loc.ZeroBasedIndexedElement<std::int64_t>(0), 3);
  EXPECT_EQ(*loc.ZeroBasedIndexedElement<std::int64_t>(1), 4);
  EXPECT_EQ(*loc.ZeroBasedIndexedElement<std::int64_t>(2), 2);
  loc.Destroy();
  RTNAME(MinlocDim)
  (loc, *array, /*KIND=*/2, /*DIM=*/1, __FILE__, __LINE__, /*MASK=*/nullptr,
      /*BACK=*/false);
  EXPECT_EQ(loc.rank(), 2);
  EXPECT_EQ(loc.type().raw(), (TypeCode{TypeCategory::Integer, 2}.raw()));
  EXPECT_EQ(loc.GetDimension(0).LowerBound(), 1);
  EXPECT_EQ(loc.GetDimension(0).Extent(), 4);
  EXPECT_EQ(loc.GetDimension(1).LowerBound(), 1);
  EXPECT_EQ(loc.GetDimension(1).Extent(), 2);
  EXPECT_EQ(*loc.ZeroBasedIndexedElement<std::int16_t>(0), 2); // -1
  EXPECT_EQ(*loc.ZeroBasedIndexedElement<std::int16_t>(1), 3); // -5
  EXPECT_EQ(*loc.ZeroBasedIndexedElement<std::int16_t>(2), 2); // -2
  EXPECT_EQ(*loc.ZeroBasedIndexedElement<std::int16_t>(3), 3); // -11
  EXPECT_EQ(*loc.ZeroBasedIndexedElement<std::int16_t>(4), 2); // -13
  EXPECT_EQ(*loc.ZeroBasedIndexedElement<std::int16_t>(5), 3); // -17
  EXPECT_EQ(*loc.ZeroBasedIndexedElement<std::int16_t>(6), 2); // -19
  EXPECT_EQ(*loc.ZeroBasedIndexedElement<std::int16_t>(7), 1); // -21
  loc.Destroy();
  auto mask{MakeArray<TypeCategory::Logical, 1>(shape,
      std::vector<bool>{false, false, false, false, false, true, false, true,
          false, false, true, true, true, false, false, true, false, true, true,
          true, false, true, true, true})};
  RTNAME(MaxlocDim)
  (loc, *array, /*KIND=*/2, /*DIM=*/3, __FILE__, __LINE__, /*MASK=*/&*mask,
      false);
  EXPECT_EQ(loc.rank(), 2);
  EXPECT_EQ(loc.type().raw(), (TypeCode{TypeCategory::Integer, 2}.raw()));
  EXPECT_EQ(loc.GetDimension(0).LowerBound(), 1);
  EXPECT_EQ(loc.GetDimension(0).Extent(), 3);
  EXPECT_EQ(loc.GetDimension(1).LowerBound(), 1);
  EXPECT_EQ(loc.GetDimension(1).Extent(), 4);
  EXPECT_EQ(*loc.ZeroBasedIndexedElement<std::int16_t>(0), 2); // 12
  EXPECT_EQ(*loc.ZeroBasedIndexedElement<std::int16_t>(1), 0);
  EXPECT_EQ(*loc.ZeroBasedIndexedElement<std::int16_t>(2), 0);
  EXPECT_EQ(*loc.ZeroBasedIndexedElement<std::int16_t>(3), 2); // -15
  EXPECT_EQ(*loc.ZeroBasedIndexedElement<std::int16_t>(4), 0);
  EXPECT_EQ(*loc.ZeroBasedIndexedElement<std::int16_t>(5), 1); // -5
  EXPECT_EQ(*loc.ZeroBasedIndexedElement<std::int16_t>(6), 2); // 18
  EXPECT_EQ(*loc.ZeroBasedIndexedElement<std::int16_t>(7), 1); // -7
  EXPECT_EQ(*loc.ZeroBasedIndexedElement<std::int16_t>(8), 0);
  EXPECT_EQ(*loc.ZeroBasedIndexedElement<std::int16_t>(9), 2); // -21
  EXPECT_EQ(*loc.ZeroBasedIndexedElement<std::int16_t>(10), 2); // 22
  EXPECT_EQ(*loc.ZeroBasedIndexedElement<std::int16_t>(11), 2); // 22
  loc.Destroy();
}

TEST(Reductions, Character) {
  std::vector<int> shape{2, 3};
  auto array{MakeArray<TypeCategory::Character, 1>(shape,
      std::vector<std::string>{"abc", "def", "ghi", "jkl", "mno", "abc"}, 3)};
  StaticDescriptor<1> statDesc[2];
  Descriptor &res{statDesc[0].descriptor()};
  RTNAME(MaxvalCharacter)(res, *array, __FILE__, __LINE__);
  EXPECT_EQ(res.rank(), 0);
  EXPECT_EQ(res.type().raw(), (TypeCode{TypeCategory::Character, 1}.raw()));
  EXPECT_EQ(std::memcmp(res.OffsetElement<char>(), "mno", 3), 0);
  res.Destroy();
  RTNAME(MinvalCharacter)(res, *array, __FILE__, __LINE__);
  EXPECT_EQ(res.rank(), 0);
  EXPECT_EQ(res.type().raw(), (TypeCode{TypeCategory::Character, 1}.raw()));
  EXPECT_EQ(std::memcmp(res.OffsetElement<char>(), "abc", 3), 0);
  res.Destroy();
  RTNAME(Maxloc)
  (res, *array, /*KIND=*/4, __FILE__, __LINE__, /*MASK=*/nullptr,
      /*BACK=*/false);
  EXPECT_EQ(res.rank(), 1);
  EXPECT_EQ(res.type().raw(), (TypeCode{TypeCategory::Integer, 4}.raw()));
  EXPECT_EQ(res.GetDimension(0).LowerBound(), 1);
  EXPECT_EQ(res.GetDimension(0).Extent(), 2);
  EXPECT_EQ(*res.ZeroBasedIndexedElement<std::int32_t>(0), 1);
  EXPECT_EQ(*res.ZeroBasedIndexedElement<std::int32_t>(1), 3);
  res.Destroy();
  auto mask{MakeArray<TypeCategory::Logical, 1>(
      shape, std::vector<bool>{false, true, false, true, false, true})};
  RTNAME(Maxloc)
  (res, *array, /*KIND=*/4, __FILE__, __LINE__, /*MASK=*/&*mask,
      /*BACK=*/false);
  EXPECT_EQ(res.rank(), 1);
  EXPECT_EQ(res.type().raw(), (TypeCode{TypeCategory::Integer, 4}.raw()));
  EXPECT_EQ(res.GetDimension(0).LowerBound(), 1);
  EXPECT_EQ(res.GetDimension(0).Extent(), 2);
  EXPECT_EQ(*res.ZeroBasedIndexedElement<std::int32_t>(0), 2);
  EXPECT_EQ(*res.ZeroBasedIndexedElement<std::int32_t>(1), 2);
  res.Destroy();
  RTNAME(Minloc)
  (res, *array, /*KIND=*/4, __FILE__, __LINE__, /*MASK=*/nullptr,
      /*BACK=*/false);
  EXPECT_EQ(res.rank(), 1);
  EXPECT_EQ(res.type().raw(), (TypeCode{TypeCategory::Integer, 4}.raw()));
  EXPECT_EQ(res.GetDimension(0).LowerBound(), 1);
  EXPECT_EQ(res.GetDimension(0).Extent(), 2);
  EXPECT_EQ(*res.ZeroBasedIndexedElement<std::int32_t>(0), 1);
  EXPECT_EQ(*res.ZeroBasedIndexedElement<std::int32_t>(1), 1);
  res.Destroy();
  RTNAME(Minloc)
  (res, *array, /*KIND=*/4, __FILE__, __LINE__, /*MASK=*/nullptr,
      /*BACK=*/true);
  EXPECT_EQ(res.rank(), 1);
  EXPECT_EQ(res.type().raw(), (TypeCode{TypeCategory::Integer, 4}.raw()));
  EXPECT_EQ(res.GetDimension(0).LowerBound(), 1);
  EXPECT_EQ(res.GetDimension(0).Extent(), 2);
  EXPECT_EQ(*res.ZeroBasedIndexedElement<std::int32_t>(0), 2);
  EXPECT_EQ(*res.ZeroBasedIndexedElement<std::int32_t>(1), 3);
  res.Destroy();
  RTNAME(Minloc)
  (res, *array, /*KIND=*/4, __FILE__, __LINE__, /*MASK=*/&*mask,
      /*BACK=*/true);
  EXPECT_EQ(res.rank(), 1);
  EXPECT_EQ(res.type().raw(), (TypeCode{TypeCategory::Integer, 4}.raw()));
  EXPECT_EQ(res.GetDimension(0).LowerBound(), 1);
  EXPECT_EQ(res.GetDimension(0).Extent(), 2);
  EXPECT_EQ(*res.ZeroBasedIndexedElement<std::int32_t>(0), 2);
  EXPECT_EQ(*res.ZeroBasedIndexedElement<std::int32_t>(1), 3);
  res.Destroy();
  static const char targetChar[]{"abc"};
  Descriptor &target{statDesc[1].descriptor()};
  target.Establish(1, std::strlen(targetChar),
      const_cast<void *>(static_cast<const void *>(&targetChar)), 0, nullptr,
      CFI_attribute_pointer);
  RTNAME(Findloc)
  (res, *array, target, /*KIND=*/4, __FILE__, __LINE__, nullptr,
      /*BACK=*/false);
  EXPECT_EQ(res.rank(), 1);
  EXPECT_EQ(res.type().raw(), (TypeCode{TypeCategory::Integer, 4}.raw()));
  EXPECT_EQ(res.GetDimension(0).LowerBound(), 1);
  EXPECT_EQ(res.GetDimension(0).Extent(), 2);
  EXPECT_EQ(*res.ZeroBasedIndexedElement<std::int32_t>(0), 1);
  EXPECT_EQ(*res.ZeroBasedIndexedElement<std::int32_t>(1), 1);
  res.Destroy();
  RTNAME(Findloc)
  (res, *array, target, /*KIND=*/4, __FILE__, __LINE__, nullptr, /*BACK=*/true);
  EXPECT_EQ(res.rank(), 1);
  EXPECT_EQ(res.type().raw(), (TypeCode{TypeCategory::Integer, 4}.raw()));
  EXPECT_EQ(res.GetDimension(0).LowerBound(), 1);
  EXPECT_EQ(res.GetDimension(0).Extent(), 2);
  EXPECT_EQ(*res.ZeroBasedIndexedElement<std::int32_t>(0), 2);
  EXPECT_EQ(*res.ZeroBasedIndexedElement<std::int32_t>(1), 3);
  res.Destroy();
}

TEST(Reductions, Logical) {
  std::vector<int> shape{2, 2};
  auto array{MakeArray<TypeCategory::Logical, 4>(
      shape, std::vector<std::int32_t>{false, false, true, true})};
  ASSERT_EQ(array->ElementBytes(), std::size_t{4});
  EXPECT_EQ(RTNAME(All)(*array, __FILE__, __LINE__), false);
  EXPECT_EQ(RTNAME(Any)(*array, __FILE__, __LINE__), true);
  EXPECT_EQ(RTNAME(Parity)(*array, __FILE__, __LINE__), false);
  EXPECT_EQ(RTNAME(Count)(*array, __FILE__, __LINE__), 2);
  StaticDescriptor<2> statDesc[2];
  Descriptor &res{statDesc[0].descriptor()};
  RTNAME(AllDim)(res, *array, /*DIM=*/1, __FILE__, __LINE__);
  EXPECT_EQ(res.rank(), 1);
  EXPECT_EQ(res.type().raw(), (TypeCode{TypeCategory::Logical, 4}.raw()));
  EXPECT_EQ(res.GetDimension(0).LowerBound(), 1);
  EXPECT_EQ(res.GetDimension(0).Extent(), 2);
  EXPECT_EQ(*res.ZeroBasedIndexedElement<std::int32_t>(0), 0);
  EXPECT_EQ(*res.ZeroBasedIndexedElement<std::int32_t>(1), 1);
  res.Destroy();
  RTNAME(AllDim)(res, *array, /*DIM=*/2, __FILE__, __LINE__);
  EXPECT_EQ(res.rank(), 1);
  EXPECT_EQ(res.type().raw(), (TypeCode{TypeCategory::Logical, 4}.raw()));
  EXPECT_EQ(res.GetDimension(0).LowerBound(), 1);
  EXPECT_EQ(res.GetDimension(0).Extent(), 2);
  EXPECT_EQ(*res.ZeroBasedIndexedElement<std::int32_t>(0), 0);
  EXPECT_EQ(*res.ZeroBasedIndexedElement<std::int32_t>(1), 0);
  res.Destroy();
  RTNAME(AnyDim)(res, *array, /*DIM=*/1, __FILE__, __LINE__);
  EXPECT_EQ(res.rank(), 1);
  EXPECT_EQ(res.type().raw(), (TypeCode{TypeCategory::Logical, 4}.raw()));
  EXPECT_EQ(res.GetDimension(0).LowerBound(), 1);
  EXPECT_EQ(res.GetDimension(0).Extent(), 2);
  EXPECT_EQ(*res.ZeroBasedIndexedElement<std::int32_t>(0), 0);
  EXPECT_EQ(*res.ZeroBasedIndexedElement<std::int32_t>(1), 1);
  res.Destroy();
  RTNAME(AnyDim)(res, *array, /*DIM=*/2, __FILE__, __LINE__);
  EXPECT_EQ(res.rank(), 1);
  EXPECT_EQ(res.type().raw(), (TypeCode{TypeCategory::Logical, 4}.raw()));
  EXPECT_EQ(res.GetDimension(0).LowerBound(), 1);
  EXPECT_EQ(res.GetDimension(0).Extent(), 2);
  EXPECT_EQ(*res.ZeroBasedIndexedElement<std::int32_t>(0), 1);
  EXPECT_EQ(*res.ZeroBasedIndexedElement<std::int32_t>(1), 1);
  res.Destroy();
  RTNAME(ParityDim)(res, *array, /*DIM=*/1, __FILE__, __LINE__);
  EXPECT_EQ(res.rank(), 1);
  EXPECT_EQ(res.type().raw(), (TypeCode{TypeCategory::Logical, 4}.raw()));
  EXPECT_EQ(res.GetDimension(0).LowerBound(), 1);
  EXPECT_EQ(res.GetDimension(0).Extent(), 2);
  EXPECT_EQ(*res.ZeroBasedIndexedElement<std::int32_t>(0), 0);
  EXPECT_EQ(*res.ZeroBasedIndexedElement<std::int32_t>(1), 0);
  res.Destroy();
  RTNAME(ParityDim)(res, *array, /*DIM=*/2, __FILE__, __LINE__);
  EXPECT_EQ(res.rank(), 1);
  EXPECT_EQ(res.type().raw(), (TypeCode{TypeCategory::Logical, 4}.raw()));
  EXPECT_EQ(res.GetDimension(0).LowerBound(), 1);
  EXPECT_EQ(res.GetDimension(0).Extent(), 2);
  EXPECT_EQ(*res.ZeroBasedIndexedElement<std::int32_t>(0), 1);
  EXPECT_EQ(*res.ZeroBasedIndexedElement<std::int32_t>(1), 1);
  res.Destroy();
  RTNAME(CountDim)(res, *array, /*DIM=*/1, /*KIND=*/4, __FILE__, __LINE__);
  EXPECT_EQ(res.rank(), 1);
  EXPECT_EQ(res.type().raw(), (TypeCode{TypeCategory::Integer, 4}.raw()));
  EXPECT_EQ(res.GetDimension(0).LowerBound(), 1);
  EXPECT_EQ(res.GetDimension(0).Extent(), 2);
  EXPECT_EQ(*res.ZeroBasedIndexedElement<std::int32_t>(0), 0);
  EXPECT_EQ(*res.ZeroBasedIndexedElement<std::int32_t>(1), 2);
  res.Destroy();
  RTNAME(CountDim)(res, *array, /*DIM=*/2, /*KIND=*/8, __FILE__, __LINE__);
  EXPECT_EQ(res.rank(), 1);
  EXPECT_EQ(res.type().raw(), (TypeCode{TypeCategory::Integer, 8}.raw()));
  EXPECT_EQ(res.GetDimension(0).LowerBound(), 1);
  EXPECT_EQ(res.GetDimension(0).Extent(), 2);
  EXPECT_EQ(*res.ZeroBasedIndexedElement<std::int64_t>(0), 1);
  EXPECT_EQ(*res.ZeroBasedIndexedElement<std::int64_t>(1), 1);
  res.Destroy();
  bool boolValue{false};
  Descriptor &target{statDesc[1].descriptor()};
  target.Establish(TypeCategory::Logical, 1, static_cast<void *>(&boolValue), 0,
      nullptr, CFI_attribute_pointer);
  RTNAME(Findloc)
  (res, *array, target, /*KIND=*/4, __FILE__, __LINE__, nullptr,
      /*BACK=*/false);
  EXPECT_EQ(res.rank(), 1);
  EXPECT_EQ(res.type().raw(), (TypeCode{TypeCategory::Integer, 4}.raw()));
  EXPECT_EQ(res.GetDimension(0).LowerBound(), 1);
  EXPECT_EQ(res.GetDimension(0).Extent(), 2);
  EXPECT_EQ(*res.ZeroBasedIndexedElement<std::int32_t>(0), 1);
  EXPECT_EQ(*res.ZeroBasedIndexedElement<std::int32_t>(1), 1);
  res.Destroy();
  boolValue = true;
  RTNAME(Findloc)
  (res, *array, target, /*KIND=*/4, __FILE__, __LINE__, nullptr, /*BACK=*/true);
  EXPECT_EQ(res.rank(), 1);
  EXPECT_EQ(res.type().raw(), (TypeCode{TypeCategory::Integer, 4}.raw()));
  EXPECT_EQ(res.GetDimension(0).LowerBound(), 1);
  EXPECT_EQ(res.GetDimension(0).Extent(), 2);
  EXPECT_EQ(*res.ZeroBasedIndexedElement<std::int32_t>(0), 2);
  EXPECT_EQ(*res.ZeroBasedIndexedElement<std::int32_t>(1), 2);
  res.Destroy();
}

TEST(Reductions, FindlocNumeric) {
  std::vector<int> shape{2, 3};
  auto realArray{MakeArray<TypeCategory::Real, 8>(shape,
      std::vector<double>{0.0, -0.0, 1.0, 3.14,
          std::numeric_limits<double>::quiet_NaN(),
          std::numeric_limits<double>::infinity()})};
  ASSERT_EQ(realArray->ElementBytes(), sizeof(double));
  StaticDescriptor<2> statDesc[2];
  Descriptor &res{statDesc[0].descriptor()};
  // Find the first zero
  Descriptor &target{statDesc[1].descriptor()};
  double value{0.0};
  target.Establish(TypeCategory::Real, 8, static_cast<void *>(&value), 0,
      nullptr, CFI_attribute_pointer);
  RTNAME(Findloc)
  (res, *realArray, target, 8, __FILE__, __LINE__, nullptr, /*BACK=*/false);
  EXPECT_EQ(res.rank(), 1);
  EXPECT_EQ(res.type().raw(), (TypeCode{TypeCategory::Integer, 8}.raw()));
  EXPECT_EQ(res.GetDimension(0).LowerBound(), 1);
  EXPECT_EQ(res.GetDimension(0).UpperBound(), 2);
  EXPECT_EQ(*res.ZeroBasedIndexedElement<SubscriptValue>(0), 1);
  EXPECT_EQ(*res.ZeroBasedIndexedElement<SubscriptValue>(1), 1);
  res.Destroy();
  // Find last zero (even though it's negative)
  RTNAME(Findloc)
  (res, *realArray, target, 8, __FILE__, __LINE__, nullptr, /*BACK=*/true);
  EXPECT_EQ(res.rank(), 1);
  EXPECT_EQ(res.type().raw(), (TypeCode{TypeCategory::Integer, 8}.raw()));
  EXPECT_EQ(res.GetDimension(0).LowerBound(), 1);
  EXPECT_EQ(res.GetDimension(0).UpperBound(), 2);
  EXPECT_EQ(*res.ZeroBasedIndexedElement<SubscriptValue>(0), 2);
  EXPECT_EQ(*res.ZeroBasedIndexedElement<SubscriptValue>(1), 1);
  res.Destroy();
  // Find the +Inf
  value = std::numeric_limits<double>::infinity();
  RTNAME(Findloc)
  (res, *realArray, target, 8, __FILE__, __LINE__, nullptr, /*BACK=*/false);
  EXPECT_EQ(res.rank(), 1);
  EXPECT_EQ(res.type().raw(), (TypeCode{TypeCategory::Integer, 8}.raw()));
  EXPECT_EQ(res.GetDimension(0).LowerBound(), 1);
  EXPECT_EQ(res.GetDimension(0).UpperBound(), 2);
  EXPECT_EQ(*res.ZeroBasedIndexedElement<SubscriptValue>(0), 2);
  EXPECT_EQ(*res.ZeroBasedIndexedElement<SubscriptValue>(1), 3);
  res.Destroy();
  // Ensure that we can't find a NaN
  value = std::numeric_limits<double>::quiet_NaN();
  RTNAME(Findloc)
  (res, *realArray, target, 8, __FILE__, __LINE__, nullptr, /*BACK=*/false);
  EXPECT_EQ(res.rank(), 1);
  EXPECT_EQ(res.type().raw(), (TypeCode{TypeCategory::Integer, 8}.raw()));
  EXPECT_EQ(res.GetDimension(0).LowerBound(), 1);
  EXPECT_EQ(res.GetDimension(0).UpperBound(), 2);
  EXPECT_EQ(*res.ZeroBasedIndexedElement<SubscriptValue>(0), 0);
  EXPECT_EQ(*res.ZeroBasedIndexedElement<SubscriptValue>(1), 0);
  res.Destroy();
  // Find a value of a distinct type
  int intValue{1};
  target.Establish(TypeCategory::Integer, 4, static_cast<void *>(&intValue), 0,
      nullptr, CFI_attribute_pointer);
  RTNAME(Findloc)
  (res, *realArray, target, 8, __FILE__, __LINE__, nullptr, /*BACK=*/false);
  EXPECT_EQ(res.rank(), 1);
  EXPECT_EQ(res.type().raw(), (TypeCode{TypeCategory::Integer, 8}.raw()));
  EXPECT_EQ(res.GetDimension(0).LowerBound(), 1);
  EXPECT_EQ(res.GetDimension(0).UpperBound(), 2);
  EXPECT_EQ(*res.ZeroBasedIndexedElement<SubscriptValue>(0), 1);
  EXPECT_EQ(*res.ZeroBasedIndexedElement<SubscriptValue>(1), 2);
  res.Destroy();
  // Partial reductions
  value = 1.0;
  target.Establish(TypeCategory::Real, 8, static_cast<void *>(&value), 0,
      nullptr, CFI_attribute_pointer);
  RTNAME(FindlocDim)
  (res, *realArray, target, 8, /*DIM=*/1, __FILE__, __LINE__, nullptr,
      /*BACK=*/false);
  EXPECT_EQ(res.rank(), 1);
  EXPECT_EQ(res.type().raw(), (TypeCode{TypeCategory::Integer, 8}.raw()));
  EXPECT_EQ(res.GetDimension(0).LowerBound(), 1);
  EXPECT_EQ(res.GetDimension(0).UpperBound(), 3);
  EXPECT_EQ(*res.ZeroBasedIndexedElement<SubscriptValue>(0), 0);
  EXPECT_EQ(*res.ZeroBasedIndexedElement<SubscriptValue>(1), 1);
  EXPECT_EQ(*res.ZeroBasedIndexedElement<SubscriptValue>(2), 0);
  res.Destroy();
  RTNAME(FindlocDim)
  (res, *realArray, target, 8, /*DIM=*/2, __FILE__, __LINE__, nullptr,
      /*BACK=*/true);
  EXPECT_EQ(res.rank(), 1);
  EXPECT_EQ(res.type().raw(), (TypeCode{TypeCategory::Integer, 8}.raw()));
  EXPECT_EQ(res.GetDimension(0).LowerBound(), 1);
  EXPECT_EQ(res.GetDimension(0).UpperBound(), 2);
  EXPECT_EQ(*res.ZeroBasedIndexedElement<SubscriptValue>(0), 2);
  EXPECT_EQ(*res.ZeroBasedIndexedElement<SubscriptValue>(1), 0);
  res.Destroy();
}
