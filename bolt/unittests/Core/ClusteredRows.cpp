//===- bolt/unittest/Core/ClusteredRows.cpp ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bolt/Core/DebugData.h"
#include "llvm/Support/SMLoc.h"
#include "gtest/gtest.h"
#include <vector>

using namespace llvm;
using namespace llvm::bolt;

namespace {

class ClusteredRowsTest : public ::testing::Test {
protected:
  void SetUp() override {
    Container = std::make_unique<ClusteredRowsContainer>();
  }

  std::unique_ptr<ClusteredRowsContainer> Container;
};

TEST_F(ClusteredRowsTest, CreateSingleElement) {
  ClusteredRows *CR = Container->createClusteredRows(1);
  ASSERT_NE(CR, nullptr);
  EXPECT_EQ(CR->size(), 1u);

  // Test population with single element
  std::vector<DebugLineTableRowRef> TestRefs = {{42, 100}};
  CR->populate(TestRefs);

  ArrayRef<DebugLineTableRowRef> Rows = CR->getRows();
  EXPECT_EQ(Rows.size(), 1u);
  EXPECT_EQ(Rows[0].DwCompileUnitIndex, 42u);
  EXPECT_EQ(Rows[0].RowIndex, 100u);
}

TEST_F(ClusteredRowsTest, CreateMultipleElements) {
  ClusteredRows *CR = Container->createClusteredRows(3);
  ASSERT_NE(CR, nullptr);
  EXPECT_EQ(CR->size(), 3u);

  // Test population with multiple elements
  std::vector<DebugLineTableRowRef> TestRefs = {{10, 20}, {30, 40}, {50, 60}};
  CR->populate(TestRefs);

  ArrayRef<DebugLineTableRowRef> Rows = CR->getRows();
  EXPECT_EQ(Rows.size(), 3u);

  EXPECT_EQ(Rows[0].DwCompileUnitIndex, 10u);
  EXPECT_EQ(Rows[0].RowIndex, 20u);

  EXPECT_EQ(Rows[1].DwCompileUnitIndex, 30u);
  EXPECT_EQ(Rows[1].RowIndex, 40u);

  EXPECT_EQ(Rows[2].DwCompileUnitIndex, 50u);
  EXPECT_EQ(Rows[2].RowIndex, 60u);
}

TEST_F(ClusteredRowsTest, SMLoc_Conversion) {
  ClusteredRows *CR = Container->createClusteredRows(2);
  ASSERT_NE(CR, nullptr);

  // Test SMLoc conversion
  SMLoc Loc = CR->toSMLoc();
  EXPECT_TRUE(Loc.isValid());

  // Test round-trip conversion
  const ClusteredRows *CR2 = ClusteredRows::fromSMLoc(Loc);
  EXPECT_EQ(CR, CR2);
  EXPECT_EQ(CR2->size(), 2u);
}

TEST_F(ClusteredRowsTest, PopulateWithArrayRef) {
  ClusteredRows *CR = Container->createClusteredRows(4);
  ASSERT_NE(CR, nullptr);

  // Test population with ArrayRef
  DebugLineTableRowRef TestArray[] = {{1, 2}, {3, 4}, {5, 6}, {7, 8}};
  ArrayRef<DebugLineTableRowRef> TestRefs(TestArray, 4);
  CR->populate(TestRefs);

  ArrayRef<DebugLineTableRowRef> Rows = CR->getRows();
  EXPECT_EQ(Rows.size(), 4u);

  for (size_t i = 0; i < 4; ++i) {
    EXPECT_EQ(Rows[i].DwCompileUnitIndex, TestArray[i].DwCompileUnitIndex);
    EXPECT_EQ(Rows[i].RowIndex, TestArray[i].RowIndex);
  }
}

TEST_F(ClusteredRowsTest, MultipleClusteredRows) {
  // Test creating multiple ClusteredRows objects
  ClusteredRows *CR1 = Container->createClusteredRows(2);
  ClusteredRows *CR2 = Container->createClusteredRows(3);
  ClusteredRows *CR3 = Container->createClusteredRows(1);

  ASSERT_NE(CR1, nullptr);
  ASSERT_NE(CR2, nullptr);
  ASSERT_NE(CR3, nullptr);

  // Ensure they are different objects
  EXPECT_NE(CR1, CR2);
  EXPECT_NE(CR2, CR3);
  EXPECT_NE(CR1, CR3);

  // Verify sizes
  EXPECT_EQ(CR1->size(), 2u);
  EXPECT_EQ(CR2->size(), 3u);
  EXPECT_EQ(CR3->size(), 1u);

  // Populate each with different data
  std::vector<DebugLineTableRowRef> TestRefs1 = {{100, 200}, {300, 400}};
  std::vector<DebugLineTableRowRef> TestRefs2 = {{10, 20}, {30, 40}, {50, 60}};
  std::vector<DebugLineTableRowRef> TestRefs3 = {{999, 888}};

  CR1->populate(TestRefs1);
  CR2->populate(TestRefs2);
  CR3->populate(TestRefs3);

  // Verify data integrity
  ArrayRef<DebugLineTableRowRef> Rows1 = CR1->getRows();
  ArrayRef<DebugLineTableRowRef> Rows2 = CR2->getRows();
  ArrayRef<DebugLineTableRowRef> Rows3 = CR3->getRows();

  EXPECT_EQ(Rows1[0].DwCompileUnitIndex, 100u);
  EXPECT_EQ(Rows1[1].RowIndex, 400u);

  EXPECT_EQ(Rows2[1].DwCompileUnitIndex, 30u);
  EXPECT_EQ(Rows2[2].RowIndex, 60u);

  EXPECT_EQ(Rows3[0].DwCompileUnitIndex, 999u);
  EXPECT_EQ(Rows3[0].RowIndex, 888u);
}

} // namespace
