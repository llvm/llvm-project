//===- unittests/Analysis/FlowSensitive/DataflowAnalysisContextTest.cpp ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Analysis/FlowSensitive/DataflowAnalysisContext.h"
#include "clang/Analysis/FlowSensitive/WatchedLiteralsSolver.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <memory>

namespace {

using namespace clang;
using namespace dataflow;

class DataflowAnalysisContextTest : public ::testing::Test {
protected:
  DataflowAnalysisContextTest()
      : Context(std::make_unique<WatchedLiteralsSolver>()), A(Context.arena()) {
  }

  DataflowAnalysisContext Context;
  Arena &A;
};

TEST_F(DataflowAnalysisContextTest, DistinctTopsNotEquivalent) {
  auto &X = A.create<TopBoolValue>();
  auto &Y = A.create<TopBoolValue>();
  EXPECT_FALSE(Context.equivalentBoolValues(X, Y));
}

TEST_F(DataflowAnalysisContextTest, EmptyFlowCondition) {
  auto &FC = A.makeFlowConditionToken();
  auto &C = A.create<AtomicBoolValue>();
  EXPECT_FALSE(Context.flowConditionImplies(FC, C));
}

TEST_F(DataflowAnalysisContextTest, AddFlowConditionConstraint) {
  auto &FC = A.makeFlowConditionToken();
  auto &C = A.create<AtomicBoolValue>();
  Context.addFlowConditionConstraint(FC, C);
  EXPECT_TRUE(Context.flowConditionImplies(FC, C));
}

TEST_F(DataflowAnalysisContextTest, ForkFlowCondition) {
  auto &FC1 = A.makeFlowConditionToken();
  auto &C1 = A.create<AtomicBoolValue>();
  Context.addFlowConditionConstraint(FC1, C1);

  // Forked flow condition inherits the constraints of its parent flow
  // condition.
  auto &FC2 = Context.forkFlowCondition(FC1);
  EXPECT_TRUE(Context.flowConditionImplies(FC2, C1));

  // Adding a new constraint to the forked flow condition does not affect its
  // parent flow condition.
  auto &C2 = A.create<AtomicBoolValue>();
  Context.addFlowConditionConstraint(FC2, C2);
  EXPECT_TRUE(Context.flowConditionImplies(FC2, C2));
  EXPECT_FALSE(Context.flowConditionImplies(FC1, C2));
}

TEST_F(DataflowAnalysisContextTest, JoinFlowConditions) {
  auto &C1 = A.create<AtomicBoolValue>();
  auto &C2 = A.create<AtomicBoolValue>();
  auto &C3 = A.create<AtomicBoolValue>();

  auto &FC1 = A.makeFlowConditionToken();
  Context.addFlowConditionConstraint(FC1, C1);
  Context.addFlowConditionConstraint(FC1, C3);

  auto &FC2 = A.makeFlowConditionToken();
  Context.addFlowConditionConstraint(FC2, C2);
  Context.addFlowConditionConstraint(FC2, C3);

  auto &FC3 = Context.joinFlowConditions(FC1, FC2);
  EXPECT_FALSE(Context.flowConditionImplies(FC3, C1));
  EXPECT_FALSE(Context.flowConditionImplies(FC3, C2));
  EXPECT_TRUE(Context.flowConditionImplies(FC3, C3));
}

TEST_F(DataflowAnalysisContextTest, FlowConditionTautologies) {
  // Fresh flow condition with empty/no constraints is always true.
  auto &FC1 = A.makeFlowConditionToken();
  EXPECT_TRUE(Context.flowConditionIsTautology(FC1));

  // Literal `true` is always true.
  auto &FC2 = A.makeFlowConditionToken();
  Context.addFlowConditionConstraint(FC2, A.makeLiteral(true));
  EXPECT_TRUE(Context.flowConditionIsTautology(FC2));

  // Literal `false` is never true.
  auto &FC3 = A.makeFlowConditionToken();
  Context.addFlowConditionConstraint(FC3, A.makeLiteral(false));
  EXPECT_FALSE(Context.flowConditionIsTautology(FC3));

  // We can't prove that an arbitrary bool A is always true...
  auto &C1 = A.create<AtomicBoolValue>();
  auto &FC4 = A.makeFlowConditionToken();
  Context.addFlowConditionConstraint(FC4, C1);
  EXPECT_FALSE(Context.flowConditionIsTautology(FC4));

  // ... but we can prove A || !A is true.
  auto &FC5 = A.makeFlowConditionToken();
  Context.addFlowConditionConstraint(FC5, A.makeOr(C1, A.makeNot(C1)));
  EXPECT_TRUE(Context.flowConditionIsTautology(FC5));
}

TEST_F(DataflowAnalysisContextTest, EquivBoolVals) {
  auto &X = A.create<AtomicBoolValue>();
  auto &Y = A.create<AtomicBoolValue>();
  auto &Z = A.create<AtomicBoolValue>();
  auto &True = A.makeLiteral(true);
  auto &False = A.makeLiteral(false);

  // X == X
  EXPECT_TRUE(Context.equivalentBoolValues(X, X));
  // X != Y
  EXPECT_FALSE(Context.equivalentBoolValues(X, Y));

  // !X != X
  EXPECT_FALSE(Context.equivalentBoolValues(A.makeNot(X), X));
  // !(!X) = X
  EXPECT_TRUE(Context.equivalentBoolValues(A.makeNot(A.makeNot(X)), X));

  // (X || X) == X
  EXPECT_TRUE(Context.equivalentBoolValues(A.makeOr(X, X), X));
  // (X || Y) != X
  EXPECT_FALSE(Context.equivalentBoolValues(A.makeOr(X, Y), X));
  // (X || True) == True
  EXPECT_TRUE(Context.equivalentBoolValues(A.makeOr(X, True), True));
  // (X || False) == X
  EXPECT_TRUE(Context.equivalentBoolValues(A.makeOr(X, False), X));

  // (X && X) == X
  EXPECT_TRUE(Context.equivalentBoolValues(A.makeAnd(X, X), X));
  // (X && Y) != X
  EXPECT_FALSE(Context.equivalentBoolValues(A.makeAnd(X, Y), X));
  // (X && True) == X
  EXPECT_TRUE(Context.equivalentBoolValues(A.makeAnd(X, True), X));
  // (X && False) == False
  EXPECT_TRUE(Context.equivalentBoolValues(A.makeAnd(X, False), False));

  // (X || Y) == (Y || X)
  EXPECT_TRUE(Context.equivalentBoolValues(A.makeOr(X, Y), A.makeOr(Y, X)));
  // (X && Y) == (Y && X)
  EXPECT_TRUE(Context.equivalentBoolValues(A.makeAnd(X, Y), A.makeAnd(Y, X)));

  // ((X || Y) || Z) == (X || (Y || Z))
  EXPECT_TRUE(Context.equivalentBoolValues(A.makeOr(A.makeOr(X, Y), Z),
                                           A.makeOr(X, A.makeOr(Y, Z))));
  // ((X && Y) && Z) == (X && (Y && Z))
  EXPECT_TRUE(Context.equivalentBoolValues(A.makeAnd(A.makeAnd(X, Y), Z),
                                           A.makeAnd(X, A.makeAnd(Y, Z))));
}

} // namespace
