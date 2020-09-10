//===- IntervalSet.cpp -- interval set unit tests -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../../lib/Lower/IntervalSet.h"
#include "gtest/gtest.h"

struct IntervalSetTest : public testing::Test {
  void SetUp() { iset = new Fortran::lower::IntervalSet; }
  void TearDown() { delete iset; }

  Fortran::lower::IntervalSet *iset;
};

// Test for creating an interval set
TEST_F(IntervalSetTest, TrivialCreation) {
  iset->merge(0, 9);
  iset->merge(10, 13);
  iset->merge(400, 449);

  // expect 3 non-overlapping members in the set
  EXPECT_NE(iset->empty(), true);
  EXPECT_EQ(iset->size(), 3u);
}

TEST_F(IntervalSetTest, TrivialMerge) {
  iset->merge(4, 9);
  iset->merge(8, 11);
  iset->merge(0, 12);

  // expect 1 member in the set as all 3 intervals overlap
  EXPECT_NE(iset->empty(), true);
  EXPECT_EQ(iset->size(), 1u);
}

TEST_F(IntervalSetTest, TrivialProbe) {
  iset->merge(0, 9);
  iset->merge(8, 11);
  iset->merge(20, 23);
  iset->merge(21, 21);

  // expect 2 members in the set as there are 2 pairs of overlapping intervals
  EXPECT_EQ(iset->size(), 2u);

  // test that find correctly determines if a point is a member of the set
  // `== end()` means not a member here
  EXPECT_NE(iset->find(0), iset->end());
  EXPECT_NE(iset->find(5), iset->end());
  EXPECT_NE(iset->find(11), iset->end());
  EXPECT_EQ(iset->find(12), iset->end());
  EXPECT_EQ(iset->find(19), iset->end());
  EXPECT_NE(iset->find(20), iset->end());
  EXPECT_NE(iset->find(23), iset->end());
  EXPECT_EQ(iset->find(24), iset->end());

  // test that the two interval bounds are correct
  auto iter1 = iset->find(6);
  EXPECT_EQ(iter1->first, 0u);
  EXPECT_EQ(iter1->second, 11u);
  auto iter2 = iset->find(21);
  EXPECT_EQ(iter2->first, 20u);
  EXPECT_EQ(iter2->second, 23u);  
}
