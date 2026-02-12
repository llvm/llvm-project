//===------- unittests/Analysis/Scalable/UnsafeBufferUsageTest.cpp --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Analysis/Scalable/Analyses/UnsafeBufferUsage.h"
#include "clang/Analysis/Scalable/Model/EntityId.h"
#include "clang/Analysis/Scalable/Model/EntityIdTable.h"
#include "gtest/gtest.h"
#include <memory>

using namespace clang;
using namespace ssaf;

namespace {

class UnsafeBufferUsageTest : public testing::Test {
protected:
  EntityIdTable Table;
  UnsafeBufferUsageTUSummaryBuilder Builder;
};

//////////////////////////////////////////////////////////////
//                   Data Structure Tests                   //
//////////////////////////////////////////////////////////////

#define EXPECT_CONTAINS(Set, Elt) EXPECT_NE((Set)->find(Elt), (Set)->end());
#define EXPECT_EXCLUDES(Set, Elt) EXPECT_EQ((Set)->find(Elt), (Set)->end());

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

  EXPECT_CONTAINS(ES, P1);
  EXPECT_CONTAINS(ES, P2);
  EXPECT_CONTAINS(ES, P3);
  EXPECT_CONTAINS(ES, P4);
  EXPECT_CONTAINS(ES, P5);
  EXPECT_EXCLUDES(ES, P6);

  PointerKindVariableSet Subset1{ES->getSubsetOf(E1).begin(),
                                 ES->getSubsetOf(E1).end()};

  EXPECT_CONTAINS(&Subset1, P1);
  EXPECT_CONTAINS(&Subset1, P2);
  EXPECT_EQ(Subset1.size(), 2U);

  PointerKindVariableSet Subset2{ES->getSubsetOf(E2).begin(),
                                 ES->getSubsetOf(E2).end()};

  EXPECT_CONTAINS(&Subset2, P3);
  EXPECT_CONTAINS(&Subset2, P4);
  EXPECT_EQ(Subset2.size(), 2U);

  PointerKindVariableSet Subset3{ES->getSubsetOf(E3).begin(),
                                 ES->getSubsetOf(E3).end()};

  EXPECT_CONTAINS(&Subset3, P5);
  EXPECT_EXCLUDES(&Subset3, P6);
  EXPECT_EQ(Subset3.size(), 1U);
}
} // namespace
