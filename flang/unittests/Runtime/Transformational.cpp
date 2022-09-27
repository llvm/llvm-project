//===-- flang/unittests/Runtime/Transformational.cpp ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Runtime/transformational.h"
#include "gtest/gtest.h"
#include "tools.h"
#include "flang/Runtime/type-code.h"

using namespace Fortran::runtime;
using Fortran::common::TypeCategory;

TEST(Transformational, Shifts) {
  // ARRAY  1 3 5
  //        2 4 6
  auto array{MakeArray<TypeCategory::Integer, 4>(
      std::vector<int>{2, 3}, std::vector<std::int32_t>{1, 2, 3, 4, 5, 6})};
  array->GetDimension(0).SetLowerBound(0); // shouldn't matter
  array->GetDimension(1).SetLowerBound(-1);
  StaticDescriptor<2, true> statDesc;
  Descriptor &result{statDesc.descriptor()};

  auto shift3{MakeArray<TypeCategory::Integer, 8>(
      std::vector<int>{3}, std::vector<std::int64_t>{1, -1, 2})};
  RTNAME(Cshift)(result, *array, *shift3, 1, __FILE__, __LINE__);
  EXPECT_EQ(result.type(), array->type());
  EXPECT_EQ(result.rank(), 2);
  EXPECT_EQ(result.GetDimension(0).LowerBound(), 1);
  EXPECT_EQ(result.GetDimension(0).Extent(), 2);
  EXPECT_EQ(result.GetDimension(1).LowerBound(), 1);
  EXPECT_EQ(result.GetDimension(1).Extent(), 3);
  EXPECT_EQ(result.type(), (TypeCode{TypeCategory::Integer, 4}));
  static std::int32_t cshiftExpect1[6]{2, 1, 4, 3, 5, 6};
  for (int j{0}; j < 6; ++j) {
    EXPECT_EQ(
        *result.ZeroBasedIndexedElement<std::int32_t>(j), cshiftExpect1[j]);
  }
  result.Destroy();

  auto shift2{MakeArray<TypeCategory::Integer, 1>(
      std::vector<int>{2}, std::vector<std::int8_t>{1, -1})};
  shift2->GetDimension(0).SetLowerBound(-1); // shouldn't matter
  shift2->GetDimension(1).SetLowerBound(2);
  RTNAME(Cshift)(result, *array, *shift2, 2, __FILE__, __LINE__);
  EXPECT_EQ(result.type(), array->type());
  EXPECT_EQ(result.rank(), 2);
  EXPECT_EQ(result.GetDimension(0).LowerBound(), 1);
  EXPECT_EQ(result.GetDimension(0).Extent(), 2);
  EXPECT_EQ(result.GetDimension(1).LowerBound(), 1);
  EXPECT_EQ(result.GetDimension(1).Extent(), 3);
  EXPECT_EQ(result.type(), (TypeCode{TypeCategory::Integer, 4}));
  static std::int32_t cshiftExpect2[6]{3, 6, 5, 2, 1, 4};
  for (int j{0}; j < 6; ++j) {
    EXPECT_EQ(
        *result.ZeroBasedIndexedElement<std::int32_t>(j), cshiftExpect2[j]);
  }
  result.Destroy();

  // VECTOR  1 3 5 2 4 6
  auto vector{MakeArray<TypeCategory::Integer, 4>(
      std::vector<int>{6}, std::vector<std::int32_t>{1, 2, 3, 4, 5, 6})};
  vector->GetDimension(0).SetLowerBound(0);
  StaticDescriptor<1, true> vectorDesc;
  Descriptor &vectorResult{vectorDesc.descriptor()};

  RTNAME(CshiftVector)(vectorResult, *vector, 2, __FILE__, __LINE__);
  EXPECT_EQ(vectorResult.type(), array->type());
  EXPECT_EQ(vectorResult.rank(), 1);
  EXPECT_EQ(vectorResult.GetDimension(0).LowerBound(), 1);
  EXPECT_EQ(vectorResult.GetDimension(0).Extent(), 6);
  EXPECT_EQ(vectorResult.type(), (TypeCode{TypeCategory::Integer, 4}));
  static std::int32_t cshiftExpect3[6]{3, 4, 5, 6, 1, 2};
  for (int j{0}; j < 6; ++j) {
    EXPECT_EQ(*vectorResult.ZeroBasedIndexedElement<std::int32_t>(j),
        cshiftExpect3[j]);
  }
  vectorResult.Destroy();

  RTNAME(CshiftVector)(vectorResult, *vector, -2, __FILE__, __LINE__);
  EXPECT_EQ(vectorResult.type(), array->type());
  EXPECT_EQ(vectorResult.rank(), 1);
  EXPECT_EQ(vectorResult.GetDimension(0).LowerBound(), 1);
  EXPECT_EQ(vectorResult.GetDimension(0).Extent(), 6);
  EXPECT_EQ(vectorResult.type(), (TypeCode{TypeCategory::Integer, 4}));
  static std::int32_t cshiftExpect4[6]{5, 6, 1, 2, 3, 4};
  for (int j{0}; j < 6; ++j) {
    EXPECT_EQ(*vectorResult.ZeroBasedIndexedElement<std::int32_t>(j),
        cshiftExpect4[j]);
  }
  vectorResult.Destroy();

  // VECTOR  1 3 5 2 4 6 WITH non zero lower bound in a negative cshift.
  auto vectorWithLowerBounds{MakeArray<TypeCategory::Integer, 4>(
      std::vector<int>{6}, std::vector<std::int32_t>{1, 2, 3, 4, 5, 6})};
  vectorWithLowerBounds->GetDimension(0).SetLowerBound(2);

  RTNAME(CshiftVector)
  (vectorResult, *vectorWithLowerBounds, -2, __FILE__, __LINE__);
  EXPECT_EQ(vectorResult.type(), array->type());
  EXPECT_EQ(vectorResult.rank(), 1);
  EXPECT_EQ(vectorResult.GetDimension(0).LowerBound(), 1);
  EXPECT_EQ(vectorResult.GetDimension(0).Extent(), 6);
  EXPECT_EQ(vectorResult.type(), (TypeCode{TypeCategory::Integer, 4}));
  static std::int32_t cshiftExpect5[6]{5, 6, 1, 2, 3, 4};
  for (int j{0}; j < 6; ++j) {
    EXPECT_EQ(*vectorResult.ZeroBasedIndexedElement<std::int32_t>(j),
        cshiftExpect5[j]);
  }
  vectorResult.Destroy();

  auto boundary{MakeArray<TypeCategory::Integer, 4>(
      std::vector<int>{3}, std::vector<std::int32_t>{-1, -2, -3})};
  boundary->GetDimension(0).SetLowerBound(9); // shouldn't matter
  RTNAME(Eoshift)(result, *array, *shift3, &*boundary, 1, __FILE__, __LINE__);
  EXPECT_EQ(result.type(), array->type());
  EXPECT_EQ(result.rank(), 2);
  EXPECT_EQ(result.GetDimension(0).LowerBound(), 1);
  EXPECT_EQ(result.GetDimension(0).Extent(), 2);
  EXPECT_EQ(result.GetDimension(1).LowerBound(), 1);
  EXPECT_EQ(result.GetDimension(1).Extent(), 3);
  EXPECT_EQ(result.type(), (TypeCode{TypeCategory::Integer, 4}));
  static std::int32_t eoshiftExpect1[6]{2, -1, -2, 3, -3, -3};
  for (int j{0}; j < 6; ++j) {
    EXPECT_EQ(
        *result.ZeroBasedIndexedElement<std::int32_t>(j), eoshiftExpect1[j]);
  }
  result.Destroy();

  // VECTOR EOSHIFT
  StaticDescriptor<0> boundaryDescriptor;
  Descriptor vectorBoundary{boundaryDescriptor.descriptor()};
  std::int32_t boundaryValue{343};
  vectorBoundary.Establish(TypeCategory::Integer, 4,
      const_cast<void *>(reinterpret_cast<const void *>(&boundaryValue)), 0);
  RTNAME(EoshiftVector)
  (vectorResult, *vector, 2, &vectorBoundary, __FILE__, __LINE__);
  EXPECT_EQ(vectorResult.type(), array->type());
  EXPECT_EQ(vectorResult.rank(), 1);
  EXPECT_EQ(vectorResult.GetDimension(0).LowerBound(), 1);
  EXPECT_EQ(vectorResult.GetDimension(0).Extent(), 6);
  EXPECT_EQ(vectorResult.type(), (TypeCode{TypeCategory::Integer, 4}));
  static std::int32_t eoshiftVectorExpect[6]{3, 4, 5, 6, 343, 343};
  for (int j{0}; j < 6; ++j) {
    EXPECT_EQ(*vectorResult.ZeroBasedIndexedElement<std::int32_t>(j),
        eoshiftVectorExpect[j]);
  }
  vectorResult.Destroy();

  // VECTOR EOSHIFT on input with non zero lower bounds
  RTNAME(EoshiftVector)
  (vectorResult, *vectorWithLowerBounds, -2, &vectorBoundary, __FILE__,
      __LINE__);
  EXPECT_EQ(vectorResult.type(), array->type());
  EXPECT_EQ(vectorResult.rank(), 1);
  EXPECT_EQ(vectorResult.GetDimension(0).LowerBound(), 1);
  EXPECT_EQ(vectorResult.GetDimension(0).Extent(), 6);
  EXPECT_EQ(vectorResult.type(), (TypeCode{TypeCategory::Integer, 4}));
  static std::int32_t eoshiftVectorExpect2[6]{343, 343, 1, 2, 3, 4};
  for (int j{0}; j < 6; ++j) {
    EXPECT_EQ(*vectorResult.ZeroBasedIndexedElement<std::int32_t>(j),
        eoshiftVectorExpect2[j]);
  }
  vectorResult.Destroy();
}

TEST(Transformational, Pack) {
  // ARRAY  1 3 5
  //        2 4 6
  auto array{MakeArray<TypeCategory::Integer, 4>(
      std::vector<int>{2, 3}, std::vector<std::int32_t>{1, 2, 3, 4, 5, 6})};
  array->GetDimension(0).SetLowerBound(2); // shouldn't matter
  array->GetDimension(1).SetLowerBound(-1);
  auto mask{MakeArray<TypeCategory::Logical, 1>(std::vector<int>{2, 3},
      std::vector<std::uint8_t>{false, true, true, false, false, true})};
  mask->GetDimension(0).SetLowerBound(0); // shouldn't matter
  mask->GetDimension(1).SetLowerBound(2);
  StaticDescriptor<1, true> statDesc;
  Descriptor &result{statDesc.descriptor()};

  RTNAME(Pack)(result, *array, *mask, nullptr, __FILE__, __LINE__);
  EXPECT_EQ(result.type(), array->type());
  EXPECT_EQ(result.rank(), 1);
  EXPECT_EQ(result.GetDimension(0).LowerBound(), 1);
  EXPECT_EQ(result.GetDimension(0).Extent(), 3);
  static std::int32_t packExpect1[3]{2, 3, 6};
  for (int j{0}; j < 3; ++j) {
    EXPECT_EQ(*result.ZeroBasedIndexedElement<std::int32_t>(j), packExpect1[j])
        << " at " << j;
  }
  result.Destroy();

  auto vector{MakeArray<TypeCategory::Integer, 4>(
      std::vector<int>{5}, std::vector<std::int32_t>{-1, -2, -3, -4, -5})};
  RTNAME(Pack)(result, *array, *mask, &*vector, __FILE__, __LINE__);
  EXPECT_EQ(result.type(), array->type());
  EXPECT_EQ(result.rank(), 1);
  EXPECT_EQ(result.GetDimension(0).LowerBound(), 1);
  EXPECT_EQ(result.GetDimension(0).Extent(), 5);
  static std::int32_t packExpect2[5]{2, 3, 6, -4, -5};
  for (int j{0}; j < 5; ++j) {
    EXPECT_EQ(*result.ZeroBasedIndexedElement<std::int32_t>(j), packExpect2[j])
        << " at " << j;
  }
  result.Destroy();
}

TEST(Transformational, Spread) {
  auto array{MakeArray<TypeCategory::Integer, 4>(
      std::vector<int>{3}, std::vector<std::int32_t>{1, 2, 3})};
  array->GetDimension(0).SetLowerBound(2); // shouldn't matter
  StaticDescriptor<2, true> statDesc;
  Descriptor &result{statDesc.descriptor()};

  RTNAME(Spread)(result, *array, 1, 2, __FILE__, __LINE__);
  EXPECT_EQ(result.type(), array->type());
  EXPECT_EQ(result.rank(), 2);
  EXPECT_EQ(result.GetDimension(0).LowerBound(), 1);
  EXPECT_EQ(result.GetDimension(0).Extent(), 2);
  EXPECT_EQ(result.GetDimension(1).LowerBound(), 1);
  EXPECT_EQ(result.GetDimension(1).Extent(), 3);
  for (int j{0}; j < 6; ++j) {
    EXPECT_EQ(*result.ZeroBasedIndexedElement<std::int32_t>(j), 1 + j / 2);
  }
  result.Destroy();

  RTNAME(Spread)(result, *array, 2, 2, __FILE__, __LINE__);
  EXPECT_EQ(result.type(), array->type());
  EXPECT_EQ(result.rank(), 2);
  EXPECT_EQ(result.GetDimension(0).LowerBound(), 1);
  EXPECT_EQ(result.GetDimension(0).Extent(), 3);
  EXPECT_EQ(result.GetDimension(1).LowerBound(), 1);
  EXPECT_EQ(result.GetDimension(1).Extent(), 2);
  for (int j{0}; j < 6; ++j) {
    EXPECT_EQ(*result.ZeroBasedIndexedElement<std::int32_t>(j), 1 + j % 3);
  }
  result.Destroy();

  auto scalar{MakeArray<TypeCategory::Integer, 4>(
      std::vector<int>{}, std::vector<std::int32_t>{1})};
  RTNAME(Spread)(result, *scalar, 1, 2, __FILE__, __LINE__);
  EXPECT_EQ(result.type(), array->type());
  EXPECT_EQ(result.rank(), 1);
  EXPECT_EQ(result.GetDimension(0).LowerBound(), 1);
  EXPECT_EQ(result.GetDimension(0).Extent(), 2);
  for (int j{0}; j < 2; ++j) {
    EXPECT_EQ(*result.ZeroBasedIndexedElement<std::int32_t>(j), 1);
  }
  result.Destroy();
}

TEST(Transformational, Transpose) {
  // ARRAY  1 3 5
  //        2 4 6
  auto array{MakeArray<TypeCategory::Integer, 4>(
      std::vector<int>{2, 3}, std::vector<std::int32_t>{1, 2, 3, 4, 5, 6})};
  array->GetDimension(0).SetLowerBound(2); // shouldn't matter
  array->GetDimension(1).SetLowerBound(-6);
  StaticDescriptor<2, true> statDesc;
  Descriptor &result{statDesc.descriptor()};
  RTNAME(Transpose)(result, *array, __FILE__, __LINE__);
  EXPECT_EQ(result.type(), array->type());
  EXPECT_EQ(result.rank(), 2);
  EXPECT_EQ(result.GetDimension(0).LowerBound(), 1);
  EXPECT_EQ(result.GetDimension(0).Extent(), 3);
  EXPECT_EQ(result.GetDimension(1).LowerBound(), 1);
  EXPECT_EQ(result.GetDimension(1).Extent(), 2);
  static std::int32_t expect[6]{1, 3, 5, 2, 4, 6};
  for (int j{0}; j < 6; ++j) {
    EXPECT_EQ(*result.ZeroBasedIndexedElement<std::int32_t>(j), expect[j]);
  }
  result.Destroy();
}

TEST(Transformational, Unpack) {
  auto vector{MakeArray<TypeCategory::Integer, 4>(
      std::vector<int>{4}, std::vector<std::int32_t>{1, 2, 3, 4})};
  vector->GetDimension(0).SetLowerBound(2); // shouldn't matter
  auto mask{MakeArray<TypeCategory::Logical, 1>(std::vector<int>{2, 3},
      std::vector<std::uint8_t>{false, true, true, false, false, true})};
  mask->GetDimension(0).SetLowerBound(0); // shouldn't matter
  mask->GetDimension(1).SetLowerBound(2);
  auto field{MakeArray<TypeCategory::Integer, 4>(std::vector<int>{2, 3},
      std::vector<std::int32_t>{-1, -2, -3, -4, -5, -6})};
  field->GetDimension(0).SetLowerBound(-1); // shouldn't matter
  StaticDescriptor<2, true> statDesc;
  Descriptor &result{statDesc.descriptor()};
  RTNAME(Unpack)(result, *vector, *mask, *field, __FILE__, __LINE__);
  EXPECT_EQ(result.type(), vector->type());
  EXPECT_EQ(result.rank(), 2);
  EXPECT_EQ(result.GetDimension(0).LowerBound(), 1);
  EXPECT_EQ(result.GetDimension(0).Extent(), 2);
  EXPECT_EQ(result.GetDimension(1).LowerBound(), 1);
  EXPECT_EQ(result.GetDimension(1).Extent(), 3);
  static std::int32_t expect[6]{-1, 1, 2, -4, -5, 3};
  for (int j{0}; j < 6; ++j) {
    EXPECT_EQ(*result.ZeroBasedIndexedElement<std::int32_t>(j), expect[j]);
  }
  result.Destroy();

  // Test for scalar value of the "field" argument
  auto scalarField{MakeArray<TypeCategory::Integer, 4>(
      std::vector<int>{}, std::vector<std::int32_t>{343})};
  RTNAME(Unpack)(result, *vector, *mask, *scalarField, __FILE__, __LINE__);
  EXPECT_EQ(result.rank(), 2);
  EXPECT_EQ(result.GetDimension(0).LowerBound(), 1);
  EXPECT_EQ(result.GetDimension(0).Extent(), 2);
  EXPECT_EQ(result.GetDimension(1).LowerBound(), 1);
  EXPECT_EQ(result.GetDimension(1).Extent(), 3);
  static std::int32_t scalarExpect[6]{343, 1, 2, 343, 343, 3};
  for (int j{0}; j < 6; ++j) {
    EXPECT_EQ(
        *result.ZeroBasedIndexedElement<std::int32_t>(j), scalarExpect[j]);
  }
  result.Destroy();
}

#if LDBL_MANT_DIG == 64
// Make sure the destination descriptor is created by the runtime
// with proper element size, when REAL*10 maps to 'long double'.
#define Real10CppType long double
TEST(Transformational, TransposeReal10) {
  // ARRAY  1 3 5
  //        2 4 6
  auto array{MakeArray<TypeCategory::Real, 10>(std::vector<int>{2, 3},
      std::vector<Real10CppType>{1.0, 2.0, 3.0, 4.0, 5.0, 6.0},
      sizeof(Real10CppType))};
  StaticDescriptor<2, true> statDesc;
  Descriptor &result{statDesc.descriptor()};
  RTNAME(Transpose)(result, *array, __FILE__, __LINE__);
  EXPECT_EQ(result.ElementBytes(), sizeof(Real10CppType));
  EXPECT_EQ(result.type(), array->type());
  EXPECT_EQ(result.rank(), 2);
  EXPECT_EQ(result.GetDimension(0).LowerBound(), 1);
  EXPECT_EQ(result.GetDimension(0).Extent(), 3);
  EXPECT_EQ(result.GetDimension(1).LowerBound(), 1);
  EXPECT_EQ(result.GetDimension(1).Extent(), 2);
  static Real10CppType expect[6]{1.0, 3.0, 5.0, 2.0, 4.0, 6.0};
  for (int j{0}; j < 6; ++j) {
    EXPECT_EQ(*result.ZeroBasedIndexedElement<Real10CppType>(j), expect[j]);
  }
  result.Destroy();
}
#endif
