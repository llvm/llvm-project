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
      : Context(std::make_unique<WatchedLiteralsSolver>()) {}

  DataflowAnalysisContext Context;
};

TEST_F(DataflowAnalysisContextTest,
       CreateAtomicBoolValueReturnsDistinctValues) {
  auto &X = Context.createAtomicBoolValue();
  auto &Y = Context.createAtomicBoolValue();
  EXPECT_NE(&X, &Y);
}

TEST_F(DataflowAnalysisContextTest,
       GetOrCreateConjunctionValueReturnsSameExprGivenSameArgs) {
  auto &X = Context.createAtomicBoolValue();
  auto &XAndX = Context.getOrCreateConjunctionValue(X, X);
  EXPECT_EQ(&XAndX, &X);
}

TEST_F(DataflowAnalysisContextTest,
       GetOrCreateConjunctionValueReturnsSameExprOnSubsequentCalls) {
  auto &X = Context.createAtomicBoolValue();
  auto &Y = Context.createAtomicBoolValue();
  auto &XAndY1 = Context.getOrCreateConjunctionValue(X, Y);
  auto &XAndY2 = Context.getOrCreateConjunctionValue(X, Y);
  EXPECT_EQ(&XAndY1, &XAndY2);

  auto &YAndX = Context.getOrCreateConjunctionValue(Y, X);
  EXPECT_EQ(&XAndY1, &YAndX);

  auto &Z = Context.createAtomicBoolValue();
  auto &XAndZ = Context.getOrCreateConjunctionValue(X, Z);
  EXPECT_NE(&XAndY1, &XAndZ);
}

TEST_F(DataflowAnalysisContextTest,
       GetOrCreateDisjunctionValueReturnsSameExprGivenSameArgs) {
  auto &X = Context.createAtomicBoolValue();
  auto &XOrX = Context.getOrCreateDisjunctionValue(X, X);
  EXPECT_EQ(&XOrX, &X);
}

TEST_F(DataflowAnalysisContextTest,
       GetOrCreateDisjunctionValueReturnsSameExprOnSubsequentCalls) {
  auto &X = Context.createAtomicBoolValue();
  auto &Y = Context.createAtomicBoolValue();
  auto &XOrY1 = Context.getOrCreateDisjunctionValue(X, Y);
  auto &XOrY2 = Context.getOrCreateDisjunctionValue(X, Y);
  EXPECT_EQ(&XOrY1, &XOrY2);

  auto &YOrX = Context.getOrCreateDisjunctionValue(Y, X);
  EXPECT_EQ(&XOrY1, &YOrX);

  auto &Z = Context.createAtomicBoolValue();
  auto &XOrZ = Context.getOrCreateDisjunctionValue(X, Z);
  EXPECT_NE(&XOrY1, &XOrZ);
}

TEST_F(DataflowAnalysisContextTest,
       GetOrCreateNegationValueReturnsSameExprOnSubsequentCalls) {
  auto &X = Context.createAtomicBoolValue();
  auto &NotX1 = Context.getOrCreateNegationValue(X);
  auto &NotX2 = Context.getOrCreateNegationValue(X);
  EXPECT_EQ(&NotX1, &NotX2);

  auto &Y = Context.createAtomicBoolValue();
  auto &NotY = Context.getOrCreateNegationValue(Y);
  EXPECT_NE(&NotX1, &NotY);
}

TEST_F(DataflowAnalysisContextTest, EmptyFlowCondition) {
  auto &FC = Context.makeFlowConditionToken();
  auto &C = Context.createAtomicBoolValue();
  EXPECT_FALSE(Context.flowConditionImplies(FC, C));
}

TEST_F(DataflowAnalysisContextTest, AddFlowConditionConstraint) {
  auto &FC = Context.makeFlowConditionToken();
  auto &C = Context.createAtomicBoolValue();
  Context.addFlowConditionConstraint(FC, C);
  EXPECT_TRUE(Context.flowConditionImplies(FC, C));
}

TEST_F(DataflowAnalysisContextTest, ForkFlowCondition) {
  auto &FC1 = Context.makeFlowConditionToken();
  auto &C1 = Context.createAtomicBoolValue();
  Context.addFlowConditionConstraint(FC1, C1);

  // Forked flow condition inherits the constraints of its parent flow
  // condition.
  auto &FC2 = Context.forkFlowCondition(FC1);
  EXPECT_TRUE(Context.flowConditionImplies(FC2, C1));

  // Adding a new constraint to the forked flow condition does not affect its
  // parent flow condition.
  auto &C2 = Context.createAtomicBoolValue();
  Context.addFlowConditionConstraint(FC2, C2);
  EXPECT_TRUE(Context.flowConditionImplies(FC2, C2));
  EXPECT_FALSE(Context.flowConditionImplies(FC1, C2));
}

TEST_F(DataflowAnalysisContextTest, JoinFlowConditions) {
  auto &C1 = Context.createAtomicBoolValue();
  auto &C2 = Context.createAtomicBoolValue();
  auto &C3 = Context.createAtomicBoolValue();

  auto &FC1 = Context.makeFlowConditionToken();
  Context.addFlowConditionConstraint(FC1, C1);
  Context.addFlowConditionConstraint(FC1, C3);

  auto &FC2 = Context.makeFlowConditionToken();
  Context.addFlowConditionConstraint(FC2, C2);
  Context.addFlowConditionConstraint(FC2, C3);

  auto &FC3 = Context.joinFlowConditions(FC1, FC2);
  EXPECT_FALSE(Context.flowConditionImplies(FC3, C1));
  EXPECT_FALSE(Context.flowConditionImplies(FC3, C2));
  EXPECT_TRUE(Context.flowConditionImplies(FC3, C3));
}

TEST_F(DataflowAnalysisContextTest, FlowConditionTautologies) {
  // Fresh flow condition with empty/no constraints is always true.
  auto &FC1 = Context.makeFlowConditionToken();
  EXPECT_TRUE(Context.flowConditionIsTautology(FC1));

  // Literal `true` is always true.
  auto &FC2 = Context.makeFlowConditionToken();
  Context.addFlowConditionConstraint(FC2, Context.getBoolLiteralValue(true));
  EXPECT_TRUE(Context.flowConditionIsTautology(FC2));

  // Literal `false` is never true.
  auto &FC3 = Context.makeFlowConditionToken();
  Context.addFlowConditionConstraint(FC3, Context.getBoolLiteralValue(false));
  EXPECT_FALSE(Context.flowConditionIsTautology(FC3));

  // We can't prove that an arbitrary bool A is always true...
  auto &C1 = Context.createAtomicBoolValue();
  auto &FC4 = Context.makeFlowConditionToken();
  Context.addFlowConditionConstraint(FC4, C1);
  EXPECT_FALSE(Context.flowConditionIsTautology(FC4));

  // ... but we can prove A || !A is true.
  auto &FC5 = Context.makeFlowConditionToken();
  Context.addFlowConditionConstraint(
      FC5, Context.getOrCreateDisjunctionValue(
               C1, Context.getOrCreateNegationValue(C1)));
  EXPECT_TRUE(Context.flowConditionIsTautology(FC5));
}

} // namespace
