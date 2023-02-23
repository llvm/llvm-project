//===-- flang/unittests/Runtime/ArrayConstructor.cpp-------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "gtest/gtest.h"
#include "tools.h"
#include "flang/Runtime/allocatable.h"
#include "flang/Runtime/array-constructor.h"
#include "flang/Runtime/cpp-type.h"
#include "flang/Runtime/descriptor.h"
#include "flang/Runtime/type-code.h"

#include <memory>

using namespace Fortran::runtime;
using Fortran::common::TypeCategory;

TEST(ArrayConstructor, Basic) {
  // X(4) = [1,2,3,4]
  // Y(2:3,4:6) = RESHAPE([5,6,7,8,9,10], shape=[2,3])
  //
  // Test creation of: [(i, X, Y, i=0,99,1)]
  auto x{MakeArray<TypeCategory::Integer, 4>(
      std::vector<int>{4}, std::vector<std::int32_t>{1, 2, 3, 4})};
  auto y{MakeArray<TypeCategory::Integer, 4>(
      std::vector<int>{2, 3}, std::vector<std::int32_t>{5, 6, 7, 8, 9, 10})};
  y->GetDimension(0).SetBounds(2, 3);
  y->GetDimension(1).SetBounds(4, 6);

  StaticDescriptor<1, false> statDesc;
  Descriptor &result{statDesc.descriptor()};
  result.Establish(TypeCode{CFI_type_int32_t}, 4, /*p=*/nullptr,
      /*rank=*/1, /*extents=*/nullptr, CFI_attribute_allocatable);
  std::allocator<ArrayConstructorVector> cookieAllocator;
  ArrayConstructorVector *acVector{cookieAllocator.allocate(1)};
  ASSERT_TRUE(acVector);

  // Case 1: result is a temp and extent is unknown before first call.
  result.GetDimension(0).SetBounds(1, 0);

  RTNAME(InitArrayConstructorVector)
  (*acVector, result, /*useValueLengthParameters=*/false,
      /*vectorClassSize=*/sizeof(ArrayConstructorVector));
  for (std::int32_t i{0}; i <= 99; ++i) {
    RTNAME(PushArrayConstructorSimpleScalar)(*acVector, &i);
    RTNAME(PushArrayConstructorValue)(*acVector, *x);
    RTNAME(PushArrayConstructorValue)(*acVector, *y);
  }

  ASSERT_TRUE(result.IsAllocated());
  ASSERT_EQ(result.Elements(), static_cast<std::size_t>(100 * (1 + 4 + 2 * 3)));
  SubscriptValue subscript[1]{1};
  for (std::int32_t i{0}; i <= 99; ++i) {
    ASSERT_EQ(*result.Element<std::int32_t>(subscript), i);
    ++subscript[0];
    for (std::int32_t j{1}; j <= 10; ++j) {
      EXPECT_EQ(*result.Element<std::int32_t>(subscript), j);
      ++subscript[0];
    }
  }
  EXPECT_LE(result.Elements(),
      static_cast<std::size_t>(acVector->actualAllocationSize));
  result.Deallocate();
  ASSERT_TRUE(!result.IsAllocated());

  // Case 2: result is an unallocated temp and extent is know before first call.
  // and is allocated when the first value is pushed.
  result.GetDimension(0).SetBounds(1, 1234);
  RTNAME(InitArrayConstructorVector)
  (*acVector, result, /*useValueLengthParameters=*/false,
      /*vectorClassSize=*/sizeof(ArrayConstructorVector));
  EXPECT_EQ(0, acVector->actualAllocationSize);
  std::int32_t i{42};
  RTNAME(PushArrayConstructorSimpleScalar)(*acVector, &i);
  ASSERT_TRUE(result.IsAllocated());
  EXPECT_EQ(1234, acVector->actualAllocationSize);
  result.Deallocate();

  cookieAllocator.deallocate(acVector, 1);
}

TEST(ArrayConstructor, Character) {
  // CHARACTER(2) :: C = "12"
  // X(4) = ["ab", "cd", "ef", "gh"]
  // Y(2:3,4:6) = RESHAPE(["ij", "jl", "mn", "op", "qr","st"], shape=[2,3])
  auto x{MakeArray<TypeCategory::Character, 1>(std::vector<int>{4},
      std::vector<std::string>{"ab", "cd", "ef", "gh"}, 2)};
  auto y{MakeArray<TypeCategory::Character, 1>(std::vector<int>{2, 3},
      std::vector<std::string>{"ij", "kl", "mn", "op", "qr", "st"}, 2)};
  y->GetDimension(0).SetBounds(2, 3);
  y->GetDimension(1).SetBounds(4, 6);
  auto c{MakeArray<TypeCategory::Character, 1>(
      std::vector<int>{}, std::vector<std::string>{"12"}, 2)};

  StaticDescriptor<1, false> statDesc;
  Descriptor &result{statDesc.descriptor()};
  result.Establish(TypeCode{CFI_type_char}, 0, /*p=*/nullptr,
      /*rank=*/1, /*extents=*/nullptr, CFI_attribute_allocatable);
  std::allocator<ArrayConstructorVector> cookieAllocator;
  ArrayConstructorVector *acVector{cookieAllocator.allocate(1)};
  ASSERT_TRUE(acVector);

  // Case 1: result is a temp and extent and length are unknown before the first
  // call. Test creation of: [(C, X, Y, i=1,10,1)]
  static constexpr std::size_t expectedElements{10 * (1 + 4 + 2 * 3)};
  result.GetDimension(0).SetBounds(1, 0);
  RTNAME(InitArrayConstructorVector)
  (*acVector, result, /*useValueLengthParameters=*/true,
      /*vectorClassSize=*/sizeof(ArrayConstructorVector));
  for (std::int32_t i{1}; i <= 10; ++i) {
    RTNAME(PushArrayConstructorValue)(*acVector, *c);
    RTNAME(PushArrayConstructorValue)(*acVector, *x);
    RTNAME(PushArrayConstructorValue)(*acVector, *y);
  }
  ASSERT_TRUE(result.IsAllocated());
  ASSERT_EQ(result.Elements(), expectedElements);
  ASSERT_EQ(result.ElementBytes(), 2u);
  EXPECT_LE(result.Elements(),
      static_cast<std::size_t>(acVector->actualAllocationSize));
  std::string CXY{"12abcdefghijklmnopqrst"};
  std::string expect;
  for (int i{0}; i < 10; ++i)
    expect.append(CXY);
  EXPECT_EQ(std::memcmp(
                result.OffsetElement<char>(0), expect.data(), expect.length()),
      0);
  result.Deallocate();
  cookieAllocator.deallocate(acVector, 1);
}

TEST(ArrayConstructor, CharacterRuntimeCheck) {
  // CHARACTER(2) :: C2
  // CHARACTER(3) :: C3
  // Test the runtime catch bad [C2, C3] array constructors (Fortran 2018 7.8
  // point 2.)
  auto c2{MakeArray<TypeCategory::Character, 1>(
      std::vector<int>{}, std::vector<std::string>{"ab"}, 2)};
  auto c3{MakeArray<TypeCategory::Character, 1>(
      std::vector<int>{}, std::vector<std::string>{"abc"}, 3)};
  StaticDescriptor<1, false> statDesc;
  Descriptor &result{statDesc.descriptor()};
  result.Establish(TypeCode{CFI_type_char}, 0, /*p=*/nullptr,
      /*rank=*/1, /*extents=*/nullptr, CFI_attribute_allocatable);
  std::allocator<ArrayConstructorVector> cookieAllocator;
  ArrayConstructorVector *acVector{cookieAllocator.allocate(1)};
  ASSERT_TRUE(acVector);

  result.GetDimension(0).SetBounds(1, 0);
  RTNAME(InitArrayConstructorVector)
  (*acVector, result, /*useValueLengthParameters=*/true,
      /*vectorClassSize=*/sizeof(ArrayConstructorVector));
  RTNAME(PushArrayConstructorValue)(*acVector, *c2);
  ASSERT_DEATH(RTNAME(PushArrayConstructorValue)(*acVector, *c3),
      "Array constructor: mismatched character lengths");
}
