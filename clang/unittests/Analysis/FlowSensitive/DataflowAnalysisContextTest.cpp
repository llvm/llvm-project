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

using ::testing::IsEmpty;
using ::testing::UnorderedElementsAre;

class DataflowAnalysisContextTest : public ::testing::Test {
protected:
  DataflowAnalysisContextTest()
      : Context(std::make_unique<WatchedLiteralsSolver>()), A(Context.arena()) {
  }

  DataflowAnalysisContext Context;
  Arena &A;
};

TEST_F(DataflowAnalysisContextTest, DistinctTopsNotEquivalent) {
  auto &X = A.makeTopValue();
  auto &Y = A.makeTopValue();
  EXPECT_FALSE(Context.equivalentFormulas(X.formula(), Y.formula()));
}

TEST_F(DataflowAnalysisContextTest, TautologicalFlowConditionImplies) {
  Atom FC = A.makeFlowConditionToken();
  EXPECT_TRUE(Context.flowConditionImplies(FC, A.makeLiteral(true)));
  EXPECT_FALSE(Context.flowConditionImplies(FC, A.makeLiteral(false)));
  EXPECT_FALSE(Context.flowConditionImplies(FC, A.makeAtomRef(A.makeAtom())));
}

TEST_F(DataflowAnalysisContextTest, TautologicalFlowConditionAllows) {
  Atom FC = A.makeFlowConditionToken();
  EXPECT_TRUE(Context.flowConditionAllows(FC, A.makeLiteral(true)));
  EXPECT_FALSE(Context.flowConditionAllows(FC, A.makeLiteral(false)));
  EXPECT_TRUE(Context.flowConditionAllows(FC, A.makeAtomRef(A.makeAtom())));
}

TEST_F(DataflowAnalysisContextTest, ContradictoryFlowConditionImpliesAnything) {
  Atom FC = A.makeFlowConditionToken();
  Context.addFlowConditionConstraint(FC, A.makeLiteral(false));
  EXPECT_TRUE(Context.flowConditionImplies(FC, A.makeLiteral(true)));
  EXPECT_TRUE(Context.flowConditionImplies(FC, A.makeLiteral(false)));
  EXPECT_TRUE(Context.flowConditionImplies(FC, A.makeAtomRef(A.makeAtom())));
}

TEST_F(DataflowAnalysisContextTest, ContradictoryFlowConditionAllowsNothing) {
  Atom FC = A.makeFlowConditionToken();
  Context.addFlowConditionConstraint(FC, A.makeLiteral(false));
  EXPECT_FALSE(Context.flowConditionAllows(FC, A.makeLiteral(true)));
  EXPECT_FALSE(Context.flowConditionAllows(FC, A.makeLiteral(false)));
  EXPECT_FALSE(Context.flowConditionAllows(FC, A.makeAtomRef(A.makeAtom())));
}

TEST_F(DataflowAnalysisContextTest, AddFlowConditionConstraint) {
  Atom FC = A.makeFlowConditionToken();
  auto &C = A.makeAtomRef(A.makeAtom());
  Context.addFlowConditionConstraint(FC, C);
  EXPECT_TRUE(Context.flowConditionImplies(FC, C));
}

TEST_F(DataflowAnalysisContextTest, AddInvariant) {
  Atom FC = A.makeFlowConditionToken();
  auto &C = A.makeAtomRef(A.makeAtom());
  Context.addInvariant(C);
  EXPECT_TRUE(Context.flowConditionImplies(FC, C));
}

TEST_F(DataflowAnalysisContextTest, InvariantAndFCConstraintInteract) {
  Atom FC = A.makeFlowConditionToken();
  auto &C = A.makeAtomRef(A.makeAtom());
  auto &D = A.makeAtomRef(A.makeAtom());
  Context.addInvariant(A.makeImplies(C, D));
  Context.addFlowConditionConstraint(FC, C);
  EXPECT_TRUE(Context.flowConditionImplies(FC, D));
}

TEST_F(DataflowAnalysisContextTest, ForkFlowCondition) {
  Atom FC1 = A.makeFlowConditionToken();
  auto &C1 = A.makeAtomRef(A.makeAtom());
  Context.addFlowConditionConstraint(FC1, C1);

  // Forked flow condition inherits the constraints of its parent flow
  // condition.
  Atom FC2 = Context.forkFlowCondition(FC1);
  EXPECT_TRUE(Context.flowConditionImplies(FC2, C1));

  // Adding a new constraint to the forked flow condition does not affect its
  // parent flow condition.
  auto &C2 = A.makeAtomRef(A.makeAtom());
  Context.addFlowConditionConstraint(FC2, C2);
  EXPECT_TRUE(Context.flowConditionImplies(FC2, C2));
  EXPECT_FALSE(Context.flowConditionImplies(FC1, C2));
}

TEST_F(DataflowAnalysisContextTest, JoinFlowConditions) {
  auto &C1 = A.makeAtomRef(A.makeAtom());
  auto &C2 = A.makeAtomRef(A.makeAtom());
  auto &C3 = A.makeAtomRef(A.makeAtom());

  Atom FC1 = A.makeFlowConditionToken();
  Context.addFlowConditionConstraint(FC1, C1);
  Context.addFlowConditionConstraint(FC1, C3);

  Atom FC2 = A.makeFlowConditionToken();
  Context.addFlowConditionConstraint(FC2, C2);
  Context.addFlowConditionConstraint(FC2, C3);

  Atom FC3 = Context.joinFlowConditions(FC1, FC2);
  EXPECT_FALSE(Context.flowConditionImplies(FC3, C1));
  EXPECT_FALSE(Context.flowConditionImplies(FC3, C2));
  EXPECT_TRUE(Context.flowConditionImplies(FC3, C3));
}

TEST_F(DataflowAnalysisContextTest, EquivBoolVals) {
  auto &X = A.makeAtomRef(A.makeAtom());
  auto &Y = A.makeAtomRef(A.makeAtom());
  auto &Z = A.makeAtomRef(A.makeAtom());
  auto &True = A.makeLiteral(true);
  auto &False = A.makeLiteral(false);

  // X == X
  EXPECT_TRUE(Context.equivalentFormulas(X, X));
  // X != Y
  EXPECT_FALSE(Context.equivalentFormulas(X, Y));

  // !X != X
  EXPECT_FALSE(Context.equivalentFormulas(A.makeNot(X), X));
  // !(!X) = X
  EXPECT_TRUE(Context.equivalentFormulas(A.makeNot(A.makeNot(X)), X));

  // (X || X) == X
  EXPECT_TRUE(Context.equivalentFormulas(A.makeOr(X, X), X));
  // (X || Y) != X
  EXPECT_FALSE(Context.equivalentFormulas(A.makeOr(X, Y), X));
  // (X || True) == True
  EXPECT_TRUE(Context.equivalentFormulas(A.makeOr(X, True), True));
  // (X || False) == X
  EXPECT_TRUE(Context.equivalentFormulas(A.makeOr(X, False), X));

  // (X && X) == X
  EXPECT_TRUE(Context.equivalentFormulas(A.makeAnd(X, X), X));
  // (X && Y) != X
  EXPECT_FALSE(Context.equivalentFormulas(A.makeAnd(X, Y), X));
  // (X && True) == X
  EXPECT_TRUE(Context.equivalentFormulas(A.makeAnd(X, True), X));
  // (X && False) == False
  EXPECT_TRUE(Context.equivalentFormulas(A.makeAnd(X, False), False));

  // (X || Y) == (Y || X)
  EXPECT_TRUE(Context.equivalentFormulas(A.makeOr(X, Y), A.makeOr(Y, X)));
  // (X && Y) == (Y && X)
  EXPECT_TRUE(Context.equivalentFormulas(A.makeAnd(X, Y), A.makeAnd(Y, X)));

  // ((X || Y) || Z) == (X || (Y || Z))
  EXPECT_TRUE(Context.equivalentFormulas(A.makeOr(A.makeOr(X, Y), Z),
                                         A.makeOr(X, A.makeOr(Y, Z))));
  // ((X && Y) && Z) == (X && (Y && Z))
  EXPECT_TRUE(Context.equivalentFormulas(A.makeAnd(A.makeAnd(X, Y), Z),
                                         A.makeAnd(X, A.makeAnd(Y, Z))));
}

using ExportLogicalContextTest = DataflowAnalysisContextTest;

TEST_F(ExportLogicalContextTest, EmptySet) {
  EXPECT_THAT(Context.exportLogicalContext({}).TokenDefs, IsEmpty());
}

// Only constrainted tokens are included in the output.
TEST_F(ExportLogicalContextTest, UnconstrainedIgnored) {
  Atom FC1 = A.makeFlowConditionToken();
  EXPECT_THAT(Context.exportLogicalContext({FC1}).TokenDefs, IsEmpty());
}

TEST_F(ExportLogicalContextTest, SingletonSet) {
  Atom FC1 = A.makeFlowConditionToken();
  auto &C1 = A.makeAtomRef(A.makeAtom());
  Context.addFlowConditionConstraint(FC1, C1);
  EXPECT_THAT(Context.exportLogicalContext({FC1}).TokenDefs.keys(),
              UnorderedElementsAre(FC1));
}

TEST_F(ExportLogicalContextTest, NoDependency) {
  Atom FC1 = A.makeFlowConditionToken();
  Atom FC2 = A.makeFlowConditionToken();
  Atom FC3 = A.makeFlowConditionToken();
  auto &C1 = A.makeAtomRef(A.makeAtom());
  auto &C2 = A.makeAtomRef(A.makeAtom());
  auto &C3 = A.makeAtomRef(A.makeAtom());

  Context.addFlowConditionConstraint(FC1, C1);
  Context.addFlowConditionConstraint(FC2, C2);
  Context.addFlowConditionConstraint(FC3, C3);

  // FCs are independent.
  EXPECT_THAT(Context.exportLogicalContext({FC1}).TokenDefs.keys(),
              UnorderedElementsAre(FC1));
  EXPECT_THAT(Context.exportLogicalContext({FC2}).TokenDefs.keys(),
              UnorderedElementsAre(FC2));
  EXPECT_THAT(Context.exportLogicalContext({FC3}).TokenDefs.keys(),
              UnorderedElementsAre(FC3));
}

TEST_F(ExportLogicalContextTest, SimpleDependencyChain) {
  Atom FC1 = A.makeFlowConditionToken();
  const Formula &C = A.makeAtomRef(A.makeAtom());
  Context.addFlowConditionConstraint(FC1, C);
  Atom FC2 = Context.forkFlowCondition(FC1);
  Atom FC3 = Context.forkFlowCondition(FC2);

  EXPECT_THAT(Context.exportLogicalContext({FC3}).TokenDefs.keys(),
              UnorderedElementsAre(FC1, FC2, FC3));
}

TEST_F(ExportLogicalContextTest, DependencyTree) {
  Atom FC1 = A.makeFlowConditionToken();
  const Formula &C = A.makeAtomRef(A.makeAtom());
  Context.addFlowConditionConstraint(FC1, C);
  Atom FC2 = Context.forkFlowCondition(FC1);
  Atom FC3 = A.makeFlowConditionToken();
  Context.addFlowConditionConstraint(FC3, C);
  Atom FC4 = Context.joinFlowConditions(FC2, FC3);

  EXPECT_THAT(Context.exportLogicalContext({FC4}).TokenDefs.keys(),
              UnorderedElementsAre(FC1, FC2, FC3, FC4));
}

TEST_F(ExportLogicalContextTest, DependencyDAG) {
  Atom FC1 = A.makeFlowConditionToken();
  const Formula &C = A.makeAtomRef(A.makeAtom());
  Context.addFlowConditionConstraint(FC1, C);

  Atom FC2 = Context.forkFlowCondition(FC1);
  Atom FC3 = Context.forkFlowCondition(FC1);
  Atom FC4 = Context.joinFlowConditions(FC2, FC3);

  EXPECT_THAT(Context.exportLogicalContext({FC4}).TokenDefs.keys(),
              UnorderedElementsAre(FC1, FC2, FC3, FC4));
}

TEST_F(ExportLogicalContextTest, MixedDependencies) {
  Atom FC1 = A.makeFlowConditionToken();
  const Formula &C = A.makeAtomRef(A.makeAtom());
  Context.addFlowConditionConstraint(FC1, C);

  Atom FC2 = Context.forkFlowCondition(FC1);
  Atom FC3 = Context.forkFlowCondition(FC2);
  (void)FC3; // unused, and we test below that it is not included.

  Atom FC4 = A.makeFlowConditionToken();
  Context.addFlowConditionConstraint(FC4, C);

  EXPECT_THAT(Context.exportLogicalContext({FC2, FC4}).TokenDefs.keys(),
              UnorderedElementsAre(FC1, FC2, FC4));
}
} // namespace
