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
#include <regex>

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

// The test below uses file operations that have nuances across multiple
// platforms. Hence limit coverage by linux only unless wider coverage
// should be required.
#if defined(__linux__) && !defined(__ANDROID__)
TEST(Descriptor, Dump) {
  StaticDescriptor<4> staticDesc[2];
  Descriptor &descriptor{staticDesc[0].descriptor()};
  using Type = std::int32_t;
  Type data[8][8][8];
  constexpr int four{static_cast<int>(sizeof data[0][0][0])};
  TypeCode integer{TypeCategory::Integer, four};
  // Scalar
  descriptor.Establish(integer, four, data, 0);
  FILE *tmpf{tmpfile()};
  ASSERT_TRUE(tmpf) << "tmpfile returned NULL";
  auto resetTmpFile = [tmpf]() {
    fflush(tmpf);
    rewind(tmpf);
    ftruncate(fileno(tmpf), 0);
  };

  auto getAddrFilteredContent = [tmpf]() -> std::string {
    rewind(tmpf);
    std::ostringstream content;
    char buffer[1024];
    size_t bytes_read;
    while ((bytes_read = fread(buffer, 1, sizeof(buffer), tmpf)) > 0) {
      content.write(buffer, bytes_read);
    }

    return std::regex_replace(
        std::regex_replace(content.str(), std::regex("Descriptor @.*:"),
            "Descriptor @ [addr]:"),
        std::regex("base_addr .*"), "base_addr [addr]");
  };

  descriptor.Dump(tmpf, /*dumpRawType=*/false);
  // also dump as CFI type
  descriptor.Dump(tmpf, /*dumpRawType=*/true);
  std::string output{getAddrFilteredContent()};
  ASSERT_STREQ(output.c_str(),
      "Descriptor @ [addr]:\n"
      "  base_addr [addr]\n"
      "  elem_len  4\n"
      "  version   20240719\n"
      "  rank      0 (scalar)\n"
      "  type      9 \"INTEGER(kind=4)\"\n"
      "  attribute 0\n"
      "  extra     0\n"
      "    addendum  0\n"
      "    alloc_idx 0\n"
      "Descriptor @ [addr]:\n"
      "  base_addr [addr]\n"
      "  elem_len  4\n"
      "  version   20240719\n"
      "  rank      0 (scalar)\n"
      "  type      9 \"CFI_type_int32_t\"\n"
      "  attribute 0\n"
      "  extra     0\n"
      "    addendum  0\n"
      "    alloc_idx 0\n");

  // Contiguous matrix (0:7, 0:7)
  SubscriptValue extent[3]{8, 8, 8};
  descriptor.Establish(integer, four, data, 2, extent);
  resetTmpFile();
  descriptor.Dump(tmpf, /*dumpRawType=*/false);
  output = getAddrFilteredContent();
  ASSERT_STREQ(output.c_str(),
      "Descriptor @ [addr]:\n"
      "  base_addr [addr]\n"
      "  elem_len  4\n"
      "  version   20240719\n"
      "  rank      2\n"
      "  type      9 \"INTEGER(kind=4)\"\n"
      "  attribute 0\n"
      "  extra     0\n"
      "    addendum  0\n"
      "    alloc_idx 0\n"
      "  dim[0] lower_bound 0\n"
      "         extent      8\n"
      "         sm          4\n"
      "  dim[1] lower_bound 0\n"
      "         extent      8\n"
      "         sm          32\n");

  TypeCode real{TypeCategory::Real, four};
  // Discontiguous real 3-D array (0:7, 0:6:2, 0:6:2)
  descriptor.Establish(real, four, data, 3, extent);
  descriptor.GetDimension(1).SetExtent(4);
  descriptor.GetDimension(1).SetByteStride(8 * 2 * four);
  descriptor.GetDimension(2).SetExtent(4);
  descriptor.GetDimension(2).SetByteStride(8 * 8 * 2 * four);

  resetTmpFile();
  descriptor.Dump(tmpf, /*dumpRawType=*/false);
  output = getAddrFilteredContent();
  ASSERT_STREQ(output.c_str(),
      "Descriptor @ [addr]:\n"
      "  base_addr [addr]\n"
      "  elem_len  4\n"
      "  version   20240719\n"
      "  rank      3\n"
      "  type      27 \"REAL(kind=4)\"\n"
      "  attribute 0\n"
      "  extra     0\n"
      "    addendum  0\n"
      "    alloc_idx 0\n"
      "  dim[0] lower_bound 0\n"
      "         extent      8\n"
      "         sm          4\n"
      "  dim[1] lower_bound 0\n"
      "         extent      4\n"
      "         sm          64\n"
      "  dim[2] lower_bound 0\n"
      "         extent      4\n"
      "         sm          512\n");
  fclose(tmpf);
}
#endif // defined(__linux__) && !defined(__ANDROID__)
