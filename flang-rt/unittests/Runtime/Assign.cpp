//===-- unittests/Runtime/Assign.cpp ------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Runtime/assign.h"
#include "tools.h"
#include "gtest/gtest.h"
#include <vector>

using namespace Fortran::runtime;
using Fortran::common::TypeCategory;

TEST(Assign, RTNAME(CopyInAssign)) {
  // contiguous -> contiguous copy in
  auto intArray{MakeArray<TypeCategory::Integer, 1>(
      std::vector<int>{2, 3}, std::vector<int>{1, 2, 3, 4, 5, 6}, sizeof(int))};
  StaticDescriptor<2> staticIntResult;
  Descriptor &intResult{staticIntResult.descriptor()};

  RTNAME(CopyInAssign(intResult, *intArray));
  ASSERT_TRUE(intResult.IsAllocated());
  ASSERT_TRUE(intResult.IsContiguous());
  ASSERT_EQ(intResult.type(), intArray->type());
  ASSERT_EQ(intResult.ElementBytes(), sizeof(int));
  EXPECT_EQ(intResult.GetDimension(0).LowerBound(), 1);
  EXPECT_EQ(intResult.GetDimension(0).Extent(), 2);
  EXPECT_EQ(intResult.GetDimension(1).LowerBound(), 1);
  EXPECT_EQ(intResult.GetDimension(1).Extent(), 3);
  int expected[6] = {1, 2, 3, 4, 5, 6};
  EXPECT_EQ(
      std::memcmp(intResult.OffsetElement<int>(0), expected, 6 * sizeof(int)),
      0);
  intResult.Destroy();

  // discontiguous -> contiguous rank-1 copy in
  intArray = MakeArray<TypeCategory::Integer, 1>(std::vector<int>{8},
      std::vector<int>{1, 2, 3, 4, 5, 6, 7, 8}, sizeof(int));
  StaticDescriptor<1> staticIntResultStrided;
  Descriptor &intResultStrided{staticIntResultStrided.descriptor()};
  // Treat the descriptor as a strided array of 4
  intArray->GetDimension(0).SetByteStride(sizeof(int) * 2);
  intArray->GetDimension(0).SetExtent(4);
  RTNAME(CopyInAssign(intResultStrided, *intArray));

  int expectedStrided[4] = {1, 3, 5, 7};
  EXPECT_EQ(std::memcmp(intResultStrided.OffsetElement<int>(0), expectedStrided,
                4 * sizeof(int)),
      0);

  intResultStrided.Destroy();
}
