//===-- flang/unittests/Runtime/Reductions.cpp ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Runtime/reduction.h"
#include "gtest/gtest.h"
#include "tools.h"
#include "flang/Common/float128.h"
#include "flang/Runtime/allocatable.h"
#include "flang/Runtime/cpp-type.h"
#include "flang/Runtime/descriptor.h"
#include "flang/Runtime/reduce.h"
#include "flang/Runtime/type-code.h"
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

using namespace Fortran::runtime;
using Fortran::common::TypeCategory;

TEST(Reductions, Int4Ops) {
  auto array{MakeArray<TypeCategory::Integer, 4>(
      std::vector<int>{2, 3}, std::vector<std::int32_t>{1, 2, 3, 4, 5, 6})};
  std::int32_t sum{RTNAME(SumInteger4)(*array, __FILE__, __LINE__)};
  EXPECT_EQ(sum, 21) << sum;
  std::int32_t all{RTNAME(IAll4)(*array, __FILE__, __LINE__)};
  EXPECT_EQ(all, 0) << all;
  std::int32_t any{RTNAME(IAny4)(*array, __FILE__, __LINE__)};
  EXPECT_EQ(any, 7) << any;
  std::int32_t eor{RTNAME(IParity4)(*array, __FILE__, __LINE__)};
  EXPECT_EQ(eor, 7) << eor;
}

TEST(Reductions, DimMaskProductInt4) {
  std::vector<int> shape{2, 3};
  auto array{MakeArray<TypeCategory::Integer, 4>(
      shape, std::vector<std::int32_t>{1, 2, 3, 4, 5, 6})};
  auto mask{MakeArray<TypeCategory::Logical, 1>(
      shape, std::vector<bool>{true, false, false, true, true, true})};
  StaticDescriptor<1, true> statDesc;
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

TEST(Reductions, DoubleMaxMinNorm2) {
  std::vector<int> shape{3, 4, 2}; // rows, columns, planes
  //   0  -3   6  -9     12 -15  18 -21
  //  -1   4  -7  10    -13  16 -19  22
  //   2  -5   8 -11     14 -17  20  22   <- note last two are equal to test
  //   BACK=
  std::vector<double> rawData{0, -1, 2, -3, 4, -5, 6, -7, 8, -9, 10, -11, 12,
      -13, 14, -15, 16, -17, 18, -19, 20, -21, 22, 22};
  auto array{MakeArray<TypeCategory::Real, 8>(shape, rawData)};
  EXPECT_EQ(RTNAME(MaxvalReal8)(*array, __FILE__, __LINE__), 22.0);
  EXPECT_EQ(RTNAME(MinvalReal8)(*array, __FILE__, __LINE__), -21.0);
  double naiveNorm2{0};
  for (auto x : rawData) {
    naiveNorm2 += x * x;
  }
  naiveNorm2 = std::sqrt(naiveNorm2);
  double norm2Error{
      std::abs(naiveNorm2 - RTNAME(Norm2_8)(*array, __FILE__, __LINE__))};
  EXPECT_LE(norm2Error, 0.000001 * naiveNorm2);
  StaticDescriptor<2, true> statDesc;
  Descriptor &loc{statDesc.descriptor()};
  RTNAME(MaxlocReal8)
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
  RTNAME(MaxlocReal8)
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
  // Test scalar result for MaxlocDim, MinlocDim, MaxvalDim, MinvalDim.
  // A scalar result occurs when you have a rank 1 array and dim == 1.
  std::vector<int> shape1{24};
  auto array1{MakeArray<TypeCategory::Real, 8>(shape1, rawData)};
  StaticDescriptor<1, true> statDesc0[1];
  Descriptor &scalarResult{statDesc0[0].descriptor()};
  RTNAME(MaxlocDim)
  (scalarResult, *array1, /*KIND=*/2, /*DIM=*/1, __FILE__, __LINE__,
      /*MASK=*/nullptr, /*BACK=*/false);
  EXPECT_EQ(scalarResult.rank(), 0);
  EXPECT_EQ(*scalarResult.ZeroBasedIndexedElement<std::int16_t>(0), 23);
  scalarResult.Destroy();

  // Test .FALSE. scalar MASK argument
  auto falseMask{MakeArray<TypeCategory::Logical, 4>(
      std::vector<int>{}, std::vector<std::int32_t>{0})};
  RTNAME(MaxlocDim)
  (loc, *array, /*KIND=*/4, /*DIM=*/2, __FILE__, __LINE__,
      /*MASK=*/&*falseMask, /*BACK=*/false);
  EXPECT_EQ(loc.rank(), 2);
  EXPECT_EQ(loc.type().raw(), (TypeCode{TypeCategory::Integer, 4}.raw()));
  EXPECT_EQ(loc.GetDimension(0).LowerBound(), 1);
  EXPECT_EQ(loc.GetDimension(0).Extent(), 3);
  EXPECT_EQ(loc.GetDimension(1).LowerBound(), 1);
  EXPECT_EQ(loc.GetDimension(1).Extent(), 2);
  for (int i{0}; i < 6; ++i) {
    EXPECT_EQ(*loc.ZeroBasedIndexedElement<std::int32_t>(0), 0);
  }
  loc.Destroy();

  // Test .TRUE. scalar MASK argument
  auto trueMask{MakeArray<TypeCategory::Logical, 4>(
      std::vector<int>{}, std::vector<std::int32_t>{1})};
  RTNAME(MaxlocDim)
  (loc, *array, /*KIND=*/4, /*DIM=*/2, __FILE__, __LINE__,
      /*MASK=*/&*trueMask, /*BACK=*/false);
  EXPECT_EQ(loc.rank(), 2);
  EXPECT_EQ(loc.type().raw(), (TypeCode{TypeCategory::Integer, 4}.raw()));
  EXPECT_EQ(loc.GetDimension(0).LowerBound(), 1);
  EXPECT_EQ(loc.GetDimension(0).Extent(), 3);
  EXPECT_EQ(loc.GetDimension(1).LowerBound(), 1);
  EXPECT_EQ(loc.GetDimension(1).Extent(), 2);
  EXPECT_EQ(*loc.ZeroBasedIndexedElement<std::int32_t>(0), 3);
  EXPECT_EQ(*loc.ZeroBasedIndexedElement<std::int32_t>(1), 4);
  EXPECT_EQ(*loc.ZeroBasedIndexedElement<std::int32_t>(2), 3);
  EXPECT_EQ(*loc.ZeroBasedIndexedElement<std::int32_t>(3), 3);
  EXPECT_EQ(*loc.ZeroBasedIndexedElement<std::int32_t>(4), 4);
  EXPECT_EQ(*loc.ZeroBasedIndexedElement<std::int32_t>(5), 4);
  loc.Destroy();

  RTNAME(MinlocDim)
  (scalarResult, *array1, /*KIND=*/2, /*DIM=*/1, __FILE__, __LINE__,
      /*MASK=*/nullptr, /*BACK=*/true);
  EXPECT_EQ(scalarResult.rank(), 0);
  EXPECT_EQ(*scalarResult.ZeroBasedIndexedElement<std::int16_t>(0), 22);
  scalarResult.Destroy();

  // Test .FALSE. scalar MASK argument
  RTNAME(MinlocDim)
  (loc, *array, /*KIND=*/4, /*DIM=*/2, __FILE__, __LINE__,
      /*MASK=*/&*falseMask, /*BACK=*/false);
  EXPECT_EQ(loc.rank(), 2);
  EXPECT_EQ(loc.type().raw(), (TypeCode{TypeCategory::Integer, 4}.raw()));
  EXPECT_EQ(loc.GetDimension(0).LowerBound(), 1);
  EXPECT_EQ(loc.GetDimension(0).Extent(), 3);
  EXPECT_EQ(loc.GetDimension(1).LowerBound(), 1);
  EXPECT_EQ(loc.GetDimension(1).Extent(), 2);
  for (int i{0}; i < 6; ++i) {
    EXPECT_EQ(*loc.ZeroBasedIndexedElement<std::int32_t>(0), 0);
  }
  loc.Destroy();

  // Test .TRUE. scalar MASK argument
  RTNAME(MinlocDim)
  (loc, *array, /*KIND=*/4, /*DIM=*/2, __FILE__, __LINE__,
      /*MASK=*/&*trueMask, /*BACK=*/false);
  EXPECT_EQ(loc.rank(), 2);
  EXPECT_EQ(loc.type().raw(), (TypeCode{TypeCategory::Integer, 4}.raw()));
  EXPECT_EQ(loc.GetDimension(0).LowerBound(), 1);
  EXPECT_EQ(loc.GetDimension(0).Extent(), 3);
  EXPECT_EQ(loc.GetDimension(1).LowerBound(), 1);
  EXPECT_EQ(loc.GetDimension(1).Extent(), 2);
  EXPECT_EQ(*loc.ZeroBasedIndexedElement<std::int32_t>(0), 4);
  EXPECT_EQ(*loc.ZeroBasedIndexedElement<std::int32_t>(1), 3);
  EXPECT_EQ(*loc.ZeroBasedIndexedElement<std::int32_t>(2), 4);
  EXPECT_EQ(*loc.ZeroBasedIndexedElement<std::int32_t>(3), 4);
  EXPECT_EQ(*loc.ZeroBasedIndexedElement<std::int32_t>(4), 3);
  EXPECT_EQ(*loc.ZeroBasedIndexedElement<std::int32_t>(5), 2);
  loc.Destroy();

  RTNAME(MaxvalDim)
  (scalarResult, *array1, /*DIM=*/1, __FILE__, __LINE__, /*MASK=*/nullptr);
  EXPECT_EQ(scalarResult.rank(), 0);
  EXPECT_EQ(*scalarResult.ZeroBasedIndexedElement<double>(0), 22.0);
  scalarResult.Destroy();
  RTNAME(MinvalDim)
  (scalarResult, *array1, /*DIM=*/1, __FILE__, __LINE__, /*MASK=*/nullptr);
  EXPECT_EQ(scalarResult.rank(), 0);
  EXPECT_EQ(*scalarResult.ZeroBasedIndexedElement<double>(0), -21.0);
  scalarResult.Destroy();
}

TEST(Reductions, Character) {
  std::vector<int> shape{2, 3};
  auto array{MakeArray<TypeCategory::Character, 1>(shape,
      std::vector<std::string>{"abc", "def", "ghi", "jkl", "mno", "abc"}, 3)};
  StaticDescriptor<1, true> statDesc[2];
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
  RTNAME(MaxlocCharacter)
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
  RTNAME(MaxlocCharacter)
  (res, *array, /*KIND=*/4, __FILE__, __LINE__, /*MASK=*/&*mask,
      /*BACK=*/false);
  EXPECT_EQ(res.rank(), 1);
  EXPECT_EQ(res.type().raw(), (TypeCode{TypeCategory::Integer, 4}.raw()));
  EXPECT_EQ(res.GetDimension(0).LowerBound(), 1);
  EXPECT_EQ(res.GetDimension(0).Extent(), 2);
  EXPECT_EQ(*res.ZeroBasedIndexedElement<std::int32_t>(0), 2);
  EXPECT_EQ(*res.ZeroBasedIndexedElement<std::int32_t>(1), 2);
  res.Destroy();
  RTNAME(MinlocCharacter)
  (res, *array, /*KIND=*/4, __FILE__, __LINE__, /*MASK=*/nullptr,
      /*BACK=*/false);
  EXPECT_EQ(res.rank(), 1);
  EXPECT_EQ(res.type().raw(), (TypeCode{TypeCategory::Integer, 4}.raw()));
  EXPECT_EQ(res.GetDimension(0).LowerBound(), 1);
  EXPECT_EQ(res.GetDimension(0).Extent(), 2);
  EXPECT_EQ(*res.ZeroBasedIndexedElement<std::int32_t>(0), 1);
  EXPECT_EQ(*res.ZeroBasedIndexedElement<std::int32_t>(1), 1);
  res.Destroy();
  RTNAME(MinlocCharacter)
  (res, *array, /*KIND=*/4, __FILE__, __LINE__, /*MASK=*/nullptr,
      /*BACK=*/true);
  EXPECT_EQ(res.rank(), 1);
  EXPECT_EQ(res.type().raw(), (TypeCode{TypeCategory::Integer, 4}.raw()));
  EXPECT_EQ(res.GetDimension(0).LowerBound(), 1);
  EXPECT_EQ(res.GetDimension(0).Extent(), 2);
  EXPECT_EQ(*res.ZeroBasedIndexedElement<std::int32_t>(0), 2);
  EXPECT_EQ(*res.ZeroBasedIndexedElement<std::int32_t>(1), 3);
  res.Destroy();
  RTNAME(MinlocCharacter)
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
  StaticDescriptor<2, true> statDesc[2];
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
  // Test scalar result for AllDim.
  // A scalar result occurs when you have a rank 1 array.
  std::vector<int> shape1{4};
  auto array1{MakeArray<TypeCategory::Logical, 4>(
      shape1, std::vector<std::int32_t>{false, false, true, true})};
  StaticDescriptor<1, true> statDesc0[1];
  Descriptor &scalarResult{statDesc0[0].descriptor()};
  RTNAME(AllDim)(scalarResult, *array1, /*DIM=*/1, __FILE__, __LINE__);
  EXPECT_EQ(scalarResult.rank(), 0);
  EXPECT_EQ(*scalarResult.ZeroBasedIndexedElement<std::int32_t>(0), 0);
  scalarResult.Destroy();
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
  // Test scalar result for AnyDim.
  // A scalar result occurs when you have a rank 1 array.
  RTNAME(AnyDim)(scalarResult, *array1, /*DIM=*/1, __FILE__, __LINE__);
  EXPECT_EQ(scalarResult.rank(), 0);
  EXPECT_EQ(*scalarResult.ZeroBasedIndexedElement<std::int32_t>(0), 1);
  scalarResult.Destroy();
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
  // Test scalar result for ParityDim.
  // A scalar result occurs when you have a rank 1 array.
  RTNAME(ParityDim)(scalarResult, *array1, /*DIM=*/1, __FILE__, __LINE__);
  EXPECT_EQ(scalarResult.rank(), 0);
  EXPECT_EQ(*scalarResult.ZeroBasedIndexedElement<std::int32_t>(0), 0);
  scalarResult.Destroy();
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
  // Test scalar result for CountDim.
  // A scalar result occurs when you have a rank 1 array and dim == 1.
  RTNAME(CountDim)
  (scalarResult, *array1, /*DIM=*/1, /*KIND=*/8, __FILE__, __LINE__);
  EXPECT_EQ(scalarResult.rank(), 0);
  EXPECT_EQ(*scalarResult.ZeroBasedIndexedElement<std::int64_t>(0), 2);
  scalarResult.Destroy();
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
  StaticDescriptor<2, true> statDesc[2];
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
  // Test scalar result for FindlocDim.
  // A scalar result occurs when you have a rank 1 array, value, and dim == 1.
  std::vector<int> shape1{6};
  auto realArray1{MakeArray<TypeCategory::Real, 8>(shape1,
      std::vector<double>{0.0, -0.0, 1.0, 3.14,
          std::numeric_limits<double>::quiet_NaN(),
          std::numeric_limits<double>::infinity()})};
  StaticDescriptor<1, true> statDesc0[1];
  Descriptor &scalarResult{statDesc0[0].descriptor()};
  RTNAME(FindlocDim)
  (scalarResult, *realArray1, target, 8, /*DIM=*/1, __FILE__, __LINE__, nullptr,
      /*BACK=*/false);
  EXPECT_EQ(scalarResult.rank(), 0);
  EXPECT_EQ(*scalarResult.ZeroBasedIndexedElement<SubscriptValue>(0), 3);
  scalarResult.Destroy();
}

TEST(Reductions, DotProduct) {
  auto realVector{MakeArray<TypeCategory::Real, 8>(
      std::vector<int>{4}, std::vector<double>{0.0, -0.0, 1.0, -2.0})};
  EXPECT_EQ(
      RTNAME(DotProductReal8)(*realVector, *realVector, __FILE__, __LINE__),
      5.0);
  auto complexVector{MakeArray<TypeCategory::Complex, 4>(std::vector<int>{4},
      std::vector<std::complex<float>>{
          {0.0}, {-0.0, -0.0}, {1.0, -2.0}, {-2.0, 4.0}})};
  std::complex<double> result8;
  RTNAME(CppDotProductComplex8)
  (result8, *realVector, *complexVector, __FILE__, __LINE__);
  EXPECT_EQ(result8, (std::complex<double>{5.0, -10.0}));
  RTNAME(CppDotProductComplex8)
  (result8, *complexVector, *realVector, __FILE__, __LINE__);
  EXPECT_EQ(result8, (std::complex<double>{5.0, 10.0}));
  std::complex<float> result4;
  RTNAME(CppDotProductComplex4)
  (result4, *complexVector, *complexVector, __FILE__, __LINE__);
  EXPECT_EQ(result4, (std::complex<float>{25.0, 0.0}));
  auto logicalVector1{MakeArray<TypeCategory::Logical, 1>(
      std::vector<int>{4}, std::vector<bool>{false, false, true, true})};
  EXPECT_TRUE(RTNAME(DotProductLogical)(
      *logicalVector1, *logicalVector1, __FILE__, __LINE__));
  auto logicalVector2{MakeArray<TypeCategory::Logical, 1>(
      std::vector<int>{4}, std::vector<bool>{true, true, false, false})};
  EXPECT_TRUE(RTNAME(DotProductLogical)(
      *logicalVector2, *logicalVector2, __FILE__, __LINE__));
  EXPECT_FALSE(RTNAME(DotProductLogical)(
      *logicalVector1, *logicalVector2, __FILE__, __LINE__));
  EXPECT_FALSE(RTNAME(DotProductLogical)(
      *logicalVector2, *logicalVector1, __FILE__, __LINE__));
}

#if LDBL_MANT_DIG == 113 || HAS_FLOAT128
TEST(Reductions, ExtremaReal16) {
  // The identity value for Min/Maxval for REAL(16) was mistakenly
  // set to 0.0.
  using ElemType = CppTypeFor<TypeCategory::Real, 16>;
  std::vector<int> shape{3};
  //   1.0  2.0  3.0
  std::vector<ElemType> rawMinData{1.0, 2.0, 3.0};
  auto minArray{MakeArray<TypeCategory::Real, 16>(shape, rawMinData)};
  EXPECT_EQ(RTNAME(MinvalReal16)(*minArray, __FILE__, __LINE__), 1.0);
  //   -1.0  -2.0  -3.0
  std::vector<ElemType> rawMaxData{-1.0, -2.0, -3.0};
  auto maxArray{MakeArray<TypeCategory::Real, 16>(shape, rawMaxData)};
  EXPECT_EQ(RTNAME(MaxvalReal16)(*maxArray, __FILE__, __LINE__), -1.0);
}
#endif // LDBL_MANT_DIG == 113 || HAS_FLOAT128

static std::int32_t IAdd(const std::int32_t *x, const std::int32_t *y) {
  return *x + *y;
}

static std::int32_t IMultiply(const std::int32_t *x, const std::int32_t *y) {
  return *x * *y;
}

TEST(Reductions, ReduceInt4) {
  auto intVector{MakeArray<TypeCategory::Integer, 4>(
      std::vector<int>{4}, std::vector<std::int32_t>{1, 2, 3, 4})};
  EXPECT_EQ(RTNAME(ReduceInteger4)(*intVector, IAdd, __FILE__, __LINE__), 10);
  EXPECT_EQ(
      RTNAME(ReduceInteger4)(*intVector, IMultiply, __FILE__, __LINE__), 24);
}
TEST(Reductions, ReduceInt4Dim) {
  auto intMatrix{MakeArray<TypeCategory::Integer, 4>(
      std::vector<int>{2, 2}, std::vector<std::int32_t>{1, 2, 3, 4})};
  StaticDescriptor<1, true> statDesc;
  Descriptor &sums{statDesc.descriptor()};
  RTNAME(ReduceInteger4Dim)(sums, *intMatrix, IAdd, __FILE__, __LINE__, 1);
  EXPECT_EQ(sums.rank(), 1);
  EXPECT_EQ(sums.GetDimension(0).LowerBound(), 1);
  EXPECT_EQ(sums.GetDimension(0).Extent(), 2);
  EXPECT_EQ(*sums.ZeroBasedIndexedElement<std::int32_t>(0), 3);
  EXPECT_EQ(*sums.ZeroBasedIndexedElement<std::int32_t>(1), 7);
  sums.Destroy();
  RTNAME(ReduceInteger4Dim)(sums, *intMatrix, IAdd, __FILE__, __LINE__, 2);
  EXPECT_EQ(sums.rank(), 1);
  EXPECT_EQ(sums.GetDimension(0).LowerBound(), 1);
  EXPECT_EQ(sums.GetDimension(0).Extent(), 2);
  EXPECT_EQ(*sums.ZeroBasedIndexedElement<std::int32_t>(0), 4);
  EXPECT_EQ(*sums.ZeroBasedIndexedElement<std::int32_t>(1), 6);
  sums.Destroy();
}
