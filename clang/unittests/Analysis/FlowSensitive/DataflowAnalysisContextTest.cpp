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
       GetOrCreateConjunctionReturnsSameExprGivenSameArgs) {
  auto &X = Context.createAtomicBoolValue();
  auto &XAndX = Context.getOrCreateConjunction(X, X);
  EXPECT_EQ(&XAndX, &X);
}

TEST_F(DataflowAnalysisContextTest,
       GetOrCreateConjunctionReturnsSameExprOnSubsequentCalls) {
  auto &X = Context.createAtomicBoolValue();
  auto &Y = Context.createAtomicBoolValue();
  auto &XAndY1 = Context.getOrCreateConjunction(X, Y);
  auto &XAndY2 = Context.getOrCreateConjunction(X, Y);
  EXPECT_EQ(&XAndY1, &XAndY2);

  auto &YAndX = Context.getOrCreateConjunction(Y, X);
  EXPECT_EQ(&XAndY1, &YAndX);

  auto &Z = Context.createAtomicBoolValue();
  auto &XAndZ = Context.getOrCreateConjunction(X, Z);
  EXPECT_NE(&XAndY1, &XAndZ);
}

TEST_F(DataflowAnalysisContextTest,
       GetOrCreateDisjunctionReturnsSameExprGivenSameArgs) {
  auto &X = Context.createAtomicBoolValue();
  auto &XOrX = Context.getOrCreateDisjunction(X, X);
  EXPECT_EQ(&XOrX, &X);
}

TEST_F(DataflowAnalysisContextTest,
       GetOrCreateDisjunctionReturnsSameExprOnSubsequentCalls) {
  auto &X = Context.createAtomicBoolValue();
  auto &Y = Context.createAtomicBoolValue();
  auto &XOrY1 = Context.getOrCreateDisjunction(X, Y);
  auto &XOrY2 = Context.getOrCreateDisjunction(X, Y);
  EXPECT_EQ(&XOrY1, &XOrY2);

  auto &YOrX = Context.getOrCreateDisjunction(Y, X);
  EXPECT_EQ(&XOrY1, &YOrX);

  auto &Z = Context.createAtomicBoolValue();
  auto &XOrZ = Context.getOrCreateDisjunction(X, Z);
  EXPECT_NE(&XOrY1, &XOrZ);
}

TEST_F(DataflowAnalysisContextTest,
       GetOrCreateNegationReturnsSameExprOnSubsequentCalls) {
  auto &X = Context.createAtomicBoolValue();
  auto &NotX1 = Context.getOrCreateNegation(X);
  auto &NotX2 = Context.getOrCreateNegation(X);
  EXPECT_EQ(&NotX1, &NotX2);

  auto &Y = Context.createAtomicBoolValue();
  auto &NotY = Context.getOrCreateNegation(Y);
  EXPECT_NE(&NotX1, &NotY);
}

TEST_F(DataflowAnalysisContextTest,
       GetOrCreateImplicationReturnsTrueGivenSameArgs) {
  auto &X = Context.createAtomicBoolValue();
  auto &XImpliesX = Context.getOrCreateImplication(X, X);
  EXPECT_EQ(&XImpliesX, &Context.getBoolLiteralValue(true));
}

TEST_F(DataflowAnalysisContextTest,
       GetOrCreateImplicationReturnsSameExprOnSubsequentCalls) {
  auto &X = Context.createAtomicBoolValue();
  auto &Y = Context.createAtomicBoolValue();
  auto &XImpliesY1 = Context.getOrCreateImplication(X, Y);
  auto &XImpliesY2 = Context.getOrCreateImplication(X, Y);
  EXPECT_EQ(&XImpliesY1, &XImpliesY2);

  auto &YImpliesX = Context.getOrCreateImplication(Y, X);
  EXPECT_NE(&XImpliesY1, &YImpliesX);

  auto &Z = Context.createAtomicBoolValue();
  auto &XImpliesZ = Context.getOrCreateImplication(X, Z);
  EXPECT_NE(&XImpliesY1, &XImpliesZ);
}

TEST_F(DataflowAnalysisContextTest, GetOrCreateIffReturnsTrueGivenSameArgs) {
  auto &X = Context.createAtomicBoolValue();
  auto &XIffX = Context.getOrCreateIff(X, X);
  EXPECT_EQ(&XIffX, &Context.getBoolLiteralValue(true));
}

TEST_F(DataflowAnalysisContextTest,
       GetOrCreateIffReturnsSameExprOnSubsequentCalls) {
  auto &X = Context.createAtomicBoolValue();
  auto &Y = Context.createAtomicBoolValue();
  auto &XIffY1 = Context.getOrCreateIff(X, Y);
  auto &XIffY2 = Context.getOrCreateIff(X, Y);
  EXPECT_EQ(&XIffY1, &XIffY2);

  auto &YIffX = Context.getOrCreateIff(Y, X);
  EXPECT_EQ(&XIffY1, &YIffX);

  auto &Z = Context.createAtomicBoolValue();
  auto &XIffZ = Context.getOrCreateIff(X, Z);
  EXPECT_NE(&XIffY1, &XIffZ);
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
      FC5, Context.getOrCreateDisjunction(C1, Context.getOrCreateNegation(C1)));
  EXPECT_TRUE(Context.flowConditionIsTautology(FC5));
}

TEST_F(DataflowAnalysisContextTest, EquivBoolVals) {
  auto &X = Context.createAtomicBoolValue();
  auto &Y = Context.createAtomicBoolValue();
  auto &Z = Context.createAtomicBoolValue();
  auto &True = Context.getBoolLiteralValue(true);
  auto &False = Context.getBoolLiteralValue(false);

  // X == X
  EXPECT_TRUE(Context.equivalentBoolValues(X, X));
  // X != Y
  EXPECT_FALSE(Context.equivalentBoolValues(X, Y));

  // !X != X
  EXPECT_FALSE(Context.equivalentBoolValues(Context.getOrCreateNegation(X), X));
  // !(!X) = X
  EXPECT_TRUE(Context.equivalentBoolValues(
      Context.getOrCreateNegation(Context.getOrCreateNegation(X)), X));

  // (X || X) == X
  EXPECT_TRUE(
      Context.equivalentBoolValues(Context.getOrCreateDisjunction(X, X), X));
  // (X || Y) != X
  EXPECT_FALSE(
      Context.equivalentBoolValues(Context.getOrCreateDisjunction(X, Y), X));
  // (X || True) == True
  EXPECT_TRUE(Context.equivalentBoolValues(
      Context.getOrCreateDisjunction(X, True), True));
  // (X || False) == X
  EXPECT_TRUE(Context.equivalentBoolValues(
      Context.getOrCreateDisjunction(X, False), X));

  // (X && X) == X
  EXPECT_TRUE(
      Context.equivalentBoolValues(Context.getOrCreateConjunction(X, X), X));
  // (X && Y) != X
  EXPECT_FALSE(
      Context.equivalentBoolValues(Context.getOrCreateConjunction(X, Y), X));
  // (X && True) == X
  EXPECT_TRUE(
      Context.equivalentBoolValues(Context.getOrCreateConjunction(X, True), X));
  // (X && False) == False
  EXPECT_TRUE(Context.equivalentBoolValues(
      Context.getOrCreateConjunction(X, False), False));

  // (X || Y) == (Y || X)
  EXPECT_TRUE(
      Context.equivalentBoolValues(Context.getOrCreateDisjunction(X, Y),
                                   Context.getOrCreateDisjunction(Y, X)));
  // (X && Y) == (Y && X)
  EXPECT_TRUE(
      Context.equivalentBoolValues(Context.getOrCreateConjunction(X, Y),
                                   Context.getOrCreateConjunction(Y, X)));

  // ((X || Y) || Z) == (X || (Y || Z))
  EXPECT_TRUE(Context.equivalentBoolValues(
      Context.getOrCreateDisjunction(Context.getOrCreateDisjunction(X, Y), Z),
      Context.getOrCreateDisjunction(X, Context.getOrCreateDisjunction(Y, Z))));
  // ((X && Y) && Z) == (X && (Y && Z))
  EXPECT_TRUE(Context.equivalentBoolValues(
      Context.getOrCreateConjunction(Context.getOrCreateConjunction(X, Y), Z),
      Context.getOrCreateConjunction(X, Context.getOrCreateConjunction(Y, Z))));
}

TEST_F(DataflowAnalysisContextTest, SubstituteFlowConditionsAtomicFC) {
  auto &X = Context.createAtomicBoolValue();
  auto &True = Context.getBoolLiteralValue(true);
  auto &False = Context.getBoolLiteralValue(false);

  // FC = X
  auto &FC = Context.makeFlowConditionToken();
  Context.addFlowConditionConstraint(FC, X);

  // If X is true in FC, FC = X must be true
  auto &FCWithXTrue =
      Context.buildAndSubstituteFlowCondition(FC, {{&X, &True}});
  EXPECT_TRUE(Context.equivalentBoolValues(FCWithXTrue, True));

  // If X is false in FC, FC = X must be false
  auto &FC1WithXFalse =
      Context.buildAndSubstituteFlowCondition(FC, {{&X, &False}});
  EXPECT_TRUE(Context.equivalentBoolValues(FC1WithXFalse, False));
}

TEST_F(DataflowAnalysisContextTest, SubstituteFlowConditionsNegatedFC) {
  auto &X = Context.createAtomicBoolValue();
  auto &True = Context.getBoolLiteralValue(true);
  auto &False = Context.getBoolLiteralValue(false);

  // FC = !X
  auto &FC = Context.makeFlowConditionToken();
  Context.addFlowConditionConstraint(FC, Context.getOrCreateNegation(X));

  // If X is true in FC, FC = !X must be false
  auto &FCWithXTrue =
      Context.buildAndSubstituteFlowCondition(FC, {{&X, &True}});
  EXPECT_TRUE(Context.equivalentBoolValues(FCWithXTrue, False));

  // If X is false in FC, FC = !X must be true
  auto &FC1WithXFalse =
      Context.buildAndSubstituteFlowCondition(FC, {{&X, &False}});
  EXPECT_TRUE(Context.equivalentBoolValues(FC1WithXFalse, True));
}

TEST_F(DataflowAnalysisContextTest, SubstituteFlowConditionsDisjunctiveFC) {
  auto &X = Context.createAtomicBoolValue();
  auto &Y = Context.createAtomicBoolValue();
  auto &True = Context.getBoolLiteralValue(true);
  auto &False = Context.getBoolLiteralValue(false);

  // FC = X || Y
  auto &FC = Context.makeFlowConditionToken();
  Context.addFlowConditionConstraint(FC, Context.getOrCreateDisjunction(X, Y));

  // If X is true in FC, FC = X || Y must be true
  auto &FCWithXTrue =
      Context.buildAndSubstituteFlowCondition(FC, {{&X, &True}});
  EXPECT_TRUE(Context.equivalentBoolValues(FCWithXTrue, True));

  // If X is false in FC, FC = X || Y is equivalent to evaluating Y
  auto &FC1WithXFalse =
      Context.buildAndSubstituteFlowCondition(FC, {{&X, &False}});
  EXPECT_TRUE(Context.equivalentBoolValues(FC1WithXFalse, Y));
}

TEST_F(DataflowAnalysisContextTest, SubstituteFlowConditionsConjunctiveFC) {
  auto &X = Context.createAtomicBoolValue();
  auto &Y = Context.createAtomicBoolValue();
  auto &True = Context.getBoolLiteralValue(true);
  auto &False = Context.getBoolLiteralValue(false);

  // FC = X && Y
  auto &FC = Context.makeFlowConditionToken();
  Context.addFlowConditionConstraint(FC, Context.getOrCreateConjunction(X, Y));

  // If X is true in FC, FC = X && Y is equivalent to evaluating Y
  auto &FCWithXTrue =
      Context.buildAndSubstituteFlowCondition(FC, {{&X, &True}});
  EXPECT_TRUE(Context.equivalentBoolValues(FCWithXTrue, Y));

  // If X is false in FC, FC = X && Y must be false
  auto &FCWithXFalse =
      Context.buildAndSubstituteFlowCondition(FC, {{&X, &False}});
  EXPECT_TRUE(Context.equivalentBoolValues(FCWithXFalse, False));
}

TEST_F(DataflowAnalysisContextTest, SubstituteFlowConditionsForkedFC) {
  auto &X = Context.createAtomicBoolValue();
  auto &Y = Context.createAtomicBoolValue();
  auto &Z = Context.createAtomicBoolValue();
  auto &True = Context.getBoolLiteralValue(true);
  auto &False = Context.getBoolLiteralValue(false);

  // FC = X && Y
  auto &FC = Context.makeFlowConditionToken();
  Context.addFlowConditionConstraint(FC, Context.getOrCreateConjunction(X, Y));
  // ForkedFC = FC && Z = X && Y && Z
  auto &ForkedFC = Context.forkFlowCondition(FC);
  Context.addFlowConditionConstraint(ForkedFC, Z);

  // If any of X,Y,Z is true in ForkedFC, ForkedFC = X && Y && Z is equivalent
  // to evaluating the conjunction of the remaining values
  auto &ForkedFCWithZTrue =
      Context.buildAndSubstituteFlowCondition(ForkedFC, {{&Z, &True}});
  EXPECT_TRUE(Context.equivalentBoolValues(
      ForkedFCWithZTrue, Context.getOrCreateConjunction(X, Y)));
  auto &ForkedFCWithYAndZTrue = Context.buildAndSubstituteFlowCondition(
      ForkedFC, {{&Y, &True}, {&Z, &True}});
  EXPECT_TRUE(Context.equivalentBoolValues(ForkedFCWithYAndZTrue, X));

  // If any of X,Y,Z is false in ForkedFC, ForkedFC = X && Y && Z must be false
  auto &ForkedFCWithXFalse =
      Context.buildAndSubstituteFlowCondition(ForkedFC, {{&X, &False}});
  auto &ForkedFCWithYFalse =
      Context.buildAndSubstituteFlowCondition(ForkedFC, {{&Y, &False}});
  auto &ForkedFCWithZFalse =
      Context.buildAndSubstituteFlowCondition(ForkedFC, {{&Z, &False}});
  EXPECT_TRUE(Context.equivalentBoolValues(ForkedFCWithXFalse, False));
  EXPECT_TRUE(Context.equivalentBoolValues(ForkedFCWithYFalse, False));
  EXPECT_TRUE(Context.equivalentBoolValues(ForkedFCWithZFalse, False));
}

TEST_F(DataflowAnalysisContextTest, SubstituteFlowConditionsJoinedFC) {
  auto &X = Context.createAtomicBoolValue();
  auto &Y = Context.createAtomicBoolValue();
  auto &Z = Context.createAtomicBoolValue();
  auto &True = Context.getBoolLiteralValue(true);
  auto &False = Context.getBoolLiteralValue(false);

  // FC1 = X
  auto &FC1 = Context.makeFlowConditionToken();
  Context.addFlowConditionConstraint(FC1, X);
  // FC2 = Y
  auto &FC2 = Context.makeFlowConditionToken();
  Context.addFlowConditionConstraint(FC2, Y);
  // JoinedFC = (FC1 || FC2) && Z = (X || Y) && Z
  auto &JoinedFC = Context.joinFlowConditions(FC1, FC2);
  Context.addFlowConditionConstraint(JoinedFC, Z);

  // If any of X, Y is true in JoinedFC, JoinedFC = (X || Y) && Z is equivalent
  // to evaluating the remaining Z
  auto &JoinedFCWithXTrue =
      Context.buildAndSubstituteFlowCondition(JoinedFC, {{&X, &True}});
  auto &JoinedFCWithYTrue =
      Context.buildAndSubstituteFlowCondition(JoinedFC, {{&Y, &True}});
  EXPECT_TRUE(Context.equivalentBoolValues(JoinedFCWithXTrue, Z));
  EXPECT_TRUE(Context.equivalentBoolValues(JoinedFCWithYTrue, Z));

  // If Z is true in JoinedFC, JoinedFC = (X || Y) && Z is equivalent to
  // evaluating the remaining disjunction (X || Y)
  auto &JoinedFCWithZTrue =
      Context.buildAndSubstituteFlowCondition(JoinedFC, {{&Z, &True}});
  EXPECT_TRUE(Context.equivalentBoolValues(
      JoinedFCWithZTrue, Context.getOrCreateDisjunction(X, Y)));

  // If any of X, Y is false in JoinedFC, JoinedFC = (X || Y) && Z is equivalent
  // to evaluating the conjunction of the other value and Z
  auto &JoinedFCWithXFalse =
      Context.buildAndSubstituteFlowCondition(JoinedFC, {{&X, &False}});
  auto &JoinedFCWithYFalse =
      Context.buildAndSubstituteFlowCondition(JoinedFC, {{&Y, &False}});
  EXPECT_TRUE(Context.equivalentBoolValues(
      JoinedFCWithXFalse, Context.getOrCreateConjunction(Y, Z)));
  EXPECT_TRUE(Context.equivalentBoolValues(
      JoinedFCWithYFalse, Context.getOrCreateConjunction(X, Z)));

  // If Z is false in JoinedFC, JoinedFC = (X || Y) && Z must be false
  auto &JoinedFCWithZFalse =
      Context.buildAndSubstituteFlowCondition(JoinedFC, {{&Z, &False}});
  EXPECT_TRUE(Context.equivalentBoolValues(JoinedFCWithZFalse, False));
}

} // namespace
