//===-- unittests/Runtime/Pointer.cpp ---------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang-rt/runtime/descriptor.h"
#include "tools.h"
#include "gtest/gtest.h"

using namespace Fortran::runtime;

TEST(Descriptor, FixedStride) {
  StaticDescriptor<4> staticDesc[2];
  Descriptor &descriptor{staticDesc[0].descriptor()};
  using Type = std::int32_t;
  Type data[8][8][8];
  constexpr int four{static_cast<int>(sizeof data[0][0][0])};
  TypeCode integer{TypeCategory::Integer, four};
  // Scalar
  descriptor.Establish(integer, four, data, 0);
  EXPECT_TRUE(descriptor.IsContiguous());
  EXPECT_EQ(descriptor.FixedStride().value_or(-666), four);
  // Empty vector
  SubscriptValue extent[3]{0, 0, 0};
  descriptor.Establish(integer, four, data, 1, extent);
  EXPECT_TRUE(descriptor.IsContiguous());
  EXPECT_EQ(descriptor.FixedStride().value_or(-666), 0);
  // Contiguous vector (0:7:1)
  extent[0] = 8;
  descriptor.Establish(integer, four, data, 1, extent);
  ASSERT_EQ(descriptor.rank(), 1);
  ASSERT_EQ(descriptor.Elements(), 8u);
  ASSERT_EQ(descriptor.ElementBytes(), static_cast<unsigned>(four));
  ASSERT_EQ(descriptor.GetDimension(0).LowerBound(), 0);
  ASSERT_EQ(descriptor.GetDimension(0).ByteStride(), four);
  ASSERT_EQ(descriptor.GetDimension(0).Extent(), 8);
  EXPECT_TRUE(descriptor.IsContiguous());
  EXPECT_EQ(descriptor.FixedStride().value_or(-666), four);
  // Contiguous reverse vector (7:0:-1)
  descriptor.GetDimension(0).SetByteStride(-four);
  EXPECT_FALSE(descriptor.IsContiguous());
  EXPECT_EQ(descriptor.FixedStride().value_or(-666), -four);
  // Discontiguous vector (0:6:2)
  descriptor.GetDimension(0).SetExtent(4);
  descriptor.GetDimension(0).SetByteStride(2 * four);
  EXPECT_FALSE(descriptor.IsContiguous());
  EXPECT_EQ(descriptor.FixedStride().value_or(-666), 2 * four);
  // Empty matrix
  extent[0] = 0;
  descriptor.Establish(integer, four, data, 2, extent);
  EXPECT_TRUE(descriptor.IsContiguous());
  EXPECT_EQ(descriptor.FixedStride().value_or(-666), 0);
  // Contiguous matrix (0:7, 0:7)
  extent[0] = extent[1] = 8;
  descriptor.Establish(integer, four, data, 2, extent);
  EXPECT_TRUE(descriptor.IsContiguous());
  EXPECT_EQ(descriptor.FixedStride().value_or(-666), four);
  // Contiguous row (0:7, 0)
  descriptor.GetDimension(1).SetExtent(1);
  EXPECT_TRUE(descriptor.IsContiguous());
  EXPECT_EQ(descriptor.FixedStride().value_or(-666), four);
  // Contiguous column (0, 0:7)
  descriptor.GetDimension(0).SetExtent(1);
  descriptor.GetDimension(1).SetExtent(7);
  descriptor.GetDimension(1).SetByteStride(8 * four);
  EXPECT_FALSE(descriptor.IsContiguous());
  EXPECT_EQ(descriptor.FixedStride().value_or(-666), 8 * four);
  // Contiguous reverse row (7:0:-1, 0)
  descriptor.GetDimension(0).SetExtent(8);
  descriptor.GetDimension(0).SetByteStride(-four);
  descriptor.GetDimension(1).SetExtent(1);
  EXPECT_FALSE(descriptor.IsContiguous());
  EXPECT_EQ(descriptor.FixedStride().value_or(-666), -four);
  // Contiguous reverse column (0, 7:0:-1)
  descriptor.GetDimension(0).SetExtent(1);
  descriptor.GetDimension(0).SetByteStride(four);
  descriptor.GetDimension(1).SetExtent(7);
  descriptor.GetDimension(1).SetByteStride(8 * -four);
  EXPECT_FALSE(descriptor.IsContiguous());
  EXPECT_EQ(descriptor.FixedStride().value_or(-666), 8 * -four);
  // Discontiguous row (0:6:2, 0)
  descriptor.GetDimension(0).SetExtent(4);
  descriptor.GetDimension(0).SetByteStride(2 * four);
  descriptor.GetDimension(1).SetExtent(1);
  descriptor.GetDimension(1).SetByteStride(four);
  EXPECT_FALSE(descriptor.IsContiguous());
  EXPECT_EQ(descriptor.FixedStride().value_or(-666), 2 * four);
  // Discontiguous column (0, 0:6:2)
  descriptor.GetDimension(0).SetExtent(1);
  descriptor.GetDimension(0).SetByteStride(four);
  descriptor.GetDimension(1).SetExtent(4);
  descriptor.GetDimension(1).SetByteStride(8 * 2 * four);
  EXPECT_FALSE(descriptor.IsContiguous());
  EXPECT_EQ(descriptor.FixedStride().value_or(-666), 8 * 2 * four);
  // Discontiguous reverse row (7:1:-2, 0)
  descriptor.GetDimension(0).SetExtent(4);
  descriptor.GetDimension(0).SetByteStride(-2 * four);
  descriptor.GetDimension(1).SetExtent(1);
  descriptor.GetDimension(1).SetByteStride(four);
  EXPECT_FALSE(descriptor.IsContiguous());
  EXPECT_EQ(descriptor.FixedStride().value_or(-666), -2 * four);
  // Discontiguous reverse column (0, 7:1:-2)
  descriptor.GetDimension(0).SetExtent(1);
  descriptor.GetDimension(0).SetByteStride(four);
  descriptor.GetDimension(1).SetExtent(4);
  descriptor.GetDimension(1).SetByteStride(8 * -2 * four);
  EXPECT_FALSE(descriptor.IsContiguous());
  EXPECT_EQ(descriptor.FixedStride().value_or(-666), 8 * -2 * four);
  // Discontiguous rows (0:6:2, 0:1)
  descriptor.GetDimension(0).SetExtent(4);
  descriptor.GetDimension(0).SetByteStride(2 * four);
  descriptor.GetDimension(1).SetExtent(2);
  descriptor.GetDimension(1).SetByteStride(8 * four);
  EXPECT_FALSE(descriptor.IsContiguous());
  EXPECT_FALSE(descriptor.FixedStride().has_value());
  // Discontiguous columns (0:1, 0:6:2)
  descriptor.GetDimension(0).SetExtent(2);
  descriptor.GetDimension(0).SetByteStride(four);
  descriptor.GetDimension(1).SetExtent(4);
  descriptor.GetDimension(1).SetByteStride(8 * four);
  EXPECT_FALSE(descriptor.IsContiguous());
  EXPECT_FALSE(descriptor.FixedStride().has_value());
  // Empty 3-D array
  extent[0] = extent[1] = extent[2] = 0;
  ;
  descriptor.Establish(integer, four, data, 3, extent);
  EXPECT_TRUE(descriptor.IsContiguous());
  EXPECT_EQ(descriptor.FixedStride().value_or(-666), 0);
  // Contiguous 3-D array (0:7, 0:7, 0:7)
  extent[0] = extent[1] = extent[2] = 8;
  descriptor.Establish(integer, four, data, 3, extent);
  EXPECT_TRUE(descriptor.IsContiguous());
  EXPECT_EQ(descriptor.FixedStride().value_or(-666), four);
  // Discontiguous 3-D array (0:7, 0:6:2, 0:6:2)
  descriptor.GetDimension(1).SetExtent(4);
  descriptor.GetDimension(1).SetByteStride(8 * 2 * four);
  descriptor.GetDimension(2).SetExtent(4);
  descriptor.GetDimension(2).SetByteStride(8 * 8 * 2 * four);
  EXPECT_FALSE(descriptor.IsContiguous());
  EXPECT_FALSE(descriptor.FixedStride().has_value());
  // Discontiguous-looking empty 3-D array (0:-1, 0:6:2, 0:6:2)
  descriptor.GetDimension(0).SetExtent(0);
  EXPECT_TRUE(descriptor.IsContiguous());
  EXPECT_EQ(descriptor.FixedStride().value_or(-666), 0);
  // Discontiguous-looking empty 3-D array (0:6:2, 0:-1, 0:6:2)
  descriptor.GetDimension(0).SetExtent(4);
  descriptor.GetDimension(0).SetByteStride(2 * four);
  descriptor.GetDimension(1).SetExtent(0);
  EXPECT_TRUE(descriptor.IsContiguous());
  EXPECT_EQ(descriptor.FixedStride().value_or(-666), 0);
  // Discontiguous-looking empty 3-D array (0:6:2, 0:6:2, 0:-1)
  descriptor.GetDimension(1).SetExtent(4);
  descriptor.GetDimension(1).SetExtent(8 * 2 * four);
  descriptor.GetDimension(2).SetExtent(0);
  EXPECT_TRUE(descriptor.IsContiguous());
  EXPECT_EQ(descriptor.FixedStride().value_or(-666), 0);
}
