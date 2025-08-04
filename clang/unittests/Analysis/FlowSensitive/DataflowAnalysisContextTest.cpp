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

TEST_F(DataflowAnalysisContextTest, GetInvariant) {
  auto &C = A.makeAtomRef(A.makeAtom());
  Context.addInvariant(C);
  const Formula *Inv = Context.getInvariant();
  ASSERT_NE(Inv, nullptr);
  EXPECT_TRUE(Context.equivalentFormulas(*Inv, C));
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

TEST_F(DataflowAnalysisContextTest, GetFlowConditionConstraints) {
  auto &C1 = A.makeAtomRef(A.makeAtom());
  auto &C2 = A.makeAtomRef(A.makeAtom());
  auto &C3 = A.makeAtomRef(A.makeAtom());

  Atom FC1 = A.makeFlowConditionToken();
  Context.addFlowConditionConstraint(FC1, C1);
  Context.addFlowConditionConstraint(FC1, C3);

  Atom FC2 = A.makeFlowConditionToken();
  Context.addFlowConditionConstraint(FC2, C2);
  Context.addFlowConditionConstraint(FC2, C3);

  const Formula *CS1 = Context.getFlowConditionConstraints(FC1);
  ASSERT_NE(CS1, nullptr);
  EXPECT_TRUE(Context.equivalentFormulas(*CS1, A.makeAnd(C1, C3)));

  const Formula *CS2 = Context.getFlowConditionConstraints(FC2);
  ASSERT_NE(CS2, nullptr);
  EXPECT_TRUE(Context.equivalentFormulas(*CS2, A.makeAnd(C2, C3)));
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

using FlowConditionDepsTest = DataflowAnalysisContextTest;

TEST_F(FlowConditionDepsTest, AddEmptyDeps) {
  Atom FC1 = A.makeFlowConditionToken();

  // Add empty dependencies to FC1.
  Context.addFlowConditionDeps(FC1, {});

  // Verify that FC1 has no dependencies.
  EXPECT_EQ(Context.getFlowConditionDeps(FC1), nullptr);
}

TEST_F(FlowConditionDepsTest, AddAndGetDeps) {
  Atom FC1 = A.makeFlowConditionToken();
  Atom FC2 = A.makeFlowConditionToken();
  Atom FC3 = A.makeFlowConditionToken();

  // Add dependencies: FC1 depends on FC2 and FC3.
  Context.addFlowConditionDeps(FC1, {FC2, FC3});

  // Verify that FC1 depends on FC2 and FC3.
  const llvm::DenseSet<Atom> *Deps = Context.getFlowConditionDeps(FC1);
  ASSERT_NE(Deps, nullptr);
  EXPECT_THAT(*Deps, UnorderedElementsAre(FC2, FC3));

  // Verify that FC2 and FC3 have no dependencies.
  EXPECT_EQ(Context.getFlowConditionDeps(FC2), nullptr);
  EXPECT_EQ(Context.getFlowConditionDeps(FC3), nullptr);
}

TEST_F(FlowConditionDepsTest, AddDepsToExisting) {
  Atom FC1 = A.makeFlowConditionToken();
  Atom FC2 = A.makeFlowConditionToken();
  Atom FC3 = A.makeFlowConditionToken();

  // Add initial dependency: FC1 depends on FC2.
  Context.addFlowConditionDeps(FC1, {FC2});

  // Add more dependencies: FC1 depends on FC2 and FC3.
  Context.addFlowConditionDeps(FC1, {FC3});

  // Verify that FC1 depends on FC2 and FC3.
  const llvm::DenseSet<Atom> *Deps = Context.getFlowConditionDeps(FC1);
  ASSERT_NE(Deps, nullptr);
  EXPECT_THAT(*Deps, UnorderedElementsAre(FC2, FC3));
}

using GetTransitiveClosureTest = DataflowAnalysisContextTest;

TEST_F(GetTransitiveClosureTest, EmptySet) {
  EXPECT_THAT(Context.getTransitiveClosure({}), IsEmpty());
}

TEST_F(GetTransitiveClosureTest, SingletonSet) {
  Atom FC1 = A.makeFlowConditionToken();
  EXPECT_THAT(Context.getTransitiveClosure({FC1}), UnorderedElementsAre(FC1));
}

TEST_F(GetTransitiveClosureTest, NoDependency) {
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
  EXPECT_THAT(Context.getTransitiveClosure({FC1}), UnorderedElementsAre(FC1));
  EXPECT_THAT(Context.getTransitiveClosure({FC2}), UnorderedElementsAre(FC2));
  EXPECT_THAT(Context.getTransitiveClosure({FC3}), UnorderedElementsAre(FC3));
}

TEST_F(GetTransitiveClosureTest, SimpleDependencyChain) {
  Atom FC1 = A.makeFlowConditionToken();
  Atom FC2 = A.makeFlowConditionToken();
  Atom FC3 = A.makeFlowConditionToken();

  Context.addFlowConditionConstraint(FC1, A.makeAtomRef(FC2));
  Context.addFlowConditionDeps(FC1, {FC2});

  Context.addFlowConditionConstraint(FC2, A.makeAtomRef(FC3));
  Context.addFlowConditionDeps(FC2, {FC3});

  EXPECT_THAT(Context.getTransitiveClosure({FC1}),
              UnorderedElementsAre(FC1, FC2, FC3));
}

TEST_F(GetTransitiveClosureTest, DependencyTree) {
  Atom FC1 = A.makeFlowConditionToken();
  Atom FC2 = A.makeFlowConditionToken();
  Atom FC3 = A.makeFlowConditionToken();
  Atom FC4 = A.makeFlowConditionToken();

  Context.addFlowConditionDeps(FC1, {FC2, FC3});
  Context.addFlowConditionConstraint(
      FC1, A.makeAnd(A.makeAtomRef(FC2), A.makeAtomRef(FC3)));

  Context.addFlowConditionDeps(FC2, {FC4});
  Context.addFlowConditionConstraint(FC2, A.makeAtomRef(FC4));

  EXPECT_THAT(Context.getTransitiveClosure({FC1}),
              UnorderedElementsAre(FC1, FC2, FC3, FC4));
}

TEST_F(GetTransitiveClosureTest, DependencyDAG) {
  Atom FC1 = A.makeFlowConditionToken();
  Atom FC2 = A.makeFlowConditionToken();
  Atom FC3 = A.makeFlowConditionToken();
  Atom FC4 = A.makeFlowConditionToken();

  Context.addFlowConditionDeps(FC1, {FC2, FC3});
  Context.addFlowConditionConstraint(
      FC1, A.makeAnd(A.makeAtomRef(FC2), A.makeAtomRef(FC3)));

  Context.addFlowConditionDeps(FC2, {FC4});
  Context.addFlowConditionConstraint(FC2, A.makeAtomRef(FC4));

  Context.addFlowConditionDeps(FC3, {FC4});
  Context.addFlowConditionConstraint(FC3, A.makeAtomRef(FC4));

  EXPECT_THAT(Context.getTransitiveClosure({FC1}),
              UnorderedElementsAre(FC1, FC2, FC3, FC4));
}

TEST_F(GetTransitiveClosureTest, DependencyCycle) {
  Atom FC1 = A.makeFlowConditionToken();
  Atom FC2 = A.makeFlowConditionToken();
  Atom FC3 = A.makeFlowConditionToken();

  Context.addFlowConditionDeps(FC1, {FC2});
  Context.addFlowConditionConstraint(FC1, A.makeAtomRef(FC2));
  Context.addFlowConditionDeps(FC2, {FC3});
  Context.addFlowConditionConstraint(FC2, A.makeAtomRef(FC3));
  Context.addFlowConditionDeps(FC3, {FC1});
  Context.addFlowConditionConstraint(FC3, A.makeAtomRef(FC1));

  EXPECT_THAT(Context.getTransitiveClosure({FC1}),
              UnorderedElementsAre(FC1, FC2, FC3));
}

TEST_F(GetTransitiveClosureTest, MixedDependencies) {
  Atom FC1 = A.makeFlowConditionToken();
  Atom FC2 = A.makeFlowConditionToken();
  Atom FC3 = A.makeFlowConditionToken();
  Atom FC4 = A.makeFlowConditionToken();

  Context.addFlowConditionDeps(FC1, {FC2});
  Context.addFlowConditionConstraint(FC1, A.makeAtomRef(FC2));

  EXPECT_THAT(Context.getTransitiveClosure({FC1, FC3, FC4}),
              UnorderedElementsAre(FC1, FC2, FC3, FC4));
}
} // namespace
