//===- unittests/Analysis/Scalable/UnsafeBufferUsageExtractorTest.cpp ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Analysis/Scalable/Analyses/UnsafeBufferUsage.h"
#include "clang/Analysis/Scalable/Model/EntityId.h"
#include "clang/Analysis/Scalable/Model/EntityIdTable.h"
#include "clang/Analysis/Scalable/Model/EntityName.h"
#include "gtest/gtest.h"
#include <memory>

using namespace clang::ssaf;
using namespace clang;

namespace {

class UnsafeBufferUsageTest : public testing::Test {
protected:
  EntityIdTable Table;
  UnsafeBufferUsageTUSummaryBuilder Builder;
};

//////////////////////////////////////////////////////////////
//                   Data Structure Tests                   //
//////////////////////////////////////////////////////////////

TEST_F(UnsafeBufferUsageTest, PointerKindVariableComparison) {
  EntityId E1 = Table.getId({"c:@F@foo", "", {}});
  EntityId E2 = Table.getId({"c:@F@bar", "", {}});

  auto P1 = Builder.buildPointerKindVariable(E1, 2);
  auto P2 = Builder.buildPointerKindVariable(E1, 2);
  auto P3 = Builder.buildPointerKindVariable(E1, 1);
  auto P4 = Builder.buildPointerKindVariable(E2, 2);

  EXPECT_EQ(P1, P2);
  EXPECT_NE(P1, P3);
  EXPECT_NE(P1, P4);
  EXPECT_NE(P3, P4);
  EXPECT_TRUE(P3 < P2);
  EXPECT_TRUE(P3 < P4);
  EXPECT_FALSE(P1 < P2);
  EXPECT_FALSE(P2 < P1);
}

TEST_F(UnsafeBufferUsageTest, UnsafeBufferUsageEntitySummaryTest) {
  EntityId E1 = Table.getId({"c:@F@foo", "", {}});
  EntityId E2 = Table.getId({"c:@F@bar", "", {}});
  EntityId E3 = Table.getId({"c:@F@baz", "", {}});

  auto P1 = Builder.buildPointerKindVariable(E1, 1);
  auto P2 = Builder.buildPointerKindVariable(E1, 2);
  auto P3 = Builder.buildPointerKindVariable(E2, 1);
  auto P4 = Builder.buildPointerKindVariable(E2, 2);
  auto P5 = Builder.buildPointerKindVariable(E3, 1);
  auto P6 = Builder.buildPointerKindVariable(E3, 2);

  PointerKindVariableSet Set{P1, P2, P3, P4, P5};
  auto ES = Builder.buildUnsafeBufferUsageEntitySummary(std::move(Set));

  EXPECT_NE(ES->find(P1), ES->end());
  EXPECT_NE(ES->find(P2), ES->end());
  EXPECT_NE(ES->find(P3), ES->end());
  EXPECT_NE(ES->find(P4), ES->end());
  EXPECT_NE(ES->find(P5), ES->end());
  EXPECT_EQ(ES->find(P6), ES->end());

  PointerKindVariableSet Subset1{ES->getSubsetOf(E1).begin(),
                                 ES->getSubsetOf(E1).end()};

  EXPECT_NE(Subset1.find(P1), Subset1.end());
  EXPECT_NE(Subset1.find(P2), Subset1.end());
  EXPECT_EQ(Subset1.size(), static_cast<size_t>(2));

  PointerKindVariableSet Subset2{ES->getSubsetOf(E2).begin(),
                                 ES->getSubsetOf(E2).end()};

  EXPECT_NE(Subset2.find(P3), Subset2.end());
  EXPECT_NE(Subset2.find(P4), Subset2.end());
  EXPECT_EQ(Subset2.size(), static_cast<size_t>(2));

  PointerKindVariableSet Subset3{ES->getSubsetOf(E3).begin(),
                                 ES->getSubsetOf(E3).end()};

  EXPECT_NE(Subset3.find(P5), Subset3.end());
  EXPECT_EQ(Subset3.find(P6), Subset3.end());
  EXPECT_EQ(Subset3.size(), static_cast<size_t>(1));
}
} // namespace
