//===- unittests/Analysis/FlowSensitive/SolverTest.cpp --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <utility>

#include "TestingSupport.h"
#include "clang/Analysis/FlowSensitive/Solver.h"
#include "TestingSupport.h"
#include "clang/Analysis/FlowSensitive/Value.h"
#include "clang/Analysis/FlowSensitive/WatchedLiteralsSolver.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace {

using namespace clang;
using namespace dataflow;

using test::ConstraintContext;
using testing::_;
using testing::AnyOf;
using testing::Optional;
using testing::Pair;
using testing::UnorderedElementsAre;

// Checks if the conjunction of `Vals` is satisfiable and returns the
// corresponding result.
Solver::Result solve(llvm::DenseSet<BoolValue *> Vals) {
  return WatchedLiteralsSolver().solve(std::move(Vals));
}

void expectUnsatisfiable(Solver::Result Result) {
  EXPECT_EQ(Result.getStatus(), Solver::Result::Status::Unsatisfiable);
  EXPECT_FALSE(Result.getSolution().has_value());
}

template <typename Matcher>
void expectSatisfiable(Solver::Result Result, Matcher Solution) {
  EXPECT_EQ(Result.getStatus(), Solver::Result::Status::Satisfiable);
  EXPECT_THAT(Result.getSolution(), Optional(Solution));
}

TEST(SolverTest, Var) {
  ConstraintContext Ctx;
  auto X = Ctx.atom();

  // X
  expectSatisfiable(
      solve({X}),
      UnorderedElementsAre(Pair(X, Solver::Result::Assignment::AssignedTrue)));
}

TEST(SolverTest, NegatedVar) {
  ConstraintContext Ctx;
  auto X = Ctx.atom();
  auto NotX = Ctx.neg(X);

  // !X
  expectSatisfiable(
      solve({NotX}),
      UnorderedElementsAre(Pair(X, Solver::Result::Assignment::AssignedFalse)));
}

TEST(SolverTest, UnitConflict) {
  ConstraintContext Ctx;
  auto X = Ctx.atom();
  auto NotX = Ctx.neg(X);

  // X ^ !X
  expectUnsatisfiable(solve({X, NotX}));
}

TEST(SolverTest, DistinctVars) {
  ConstraintContext Ctx;
  auto X = Ctx.atom();
  auto Y = Ctx.atom();
  auto NotY = Ctx.neg(Y);

  // X ^ !Y
  expectSatisfiable(
      solve({X, NotY}),
      UnorderedElementsAre(Pair(X, Solver::Result::Assignment::AssignedTrue),
                           Pair(Y, Solver::Result::Assignment::AssignedFalse)));
}

TEST(SolverTest, DoubleNegation) {
  ConstraintContext Ctx;
  auto X = Ctx.atom();
  auto NotX = Ctx.neg(X);
  auto NotNotX = Ctx.neg(NotX);

  // !!X ^ !X
  expectUnsatisfiable(solve({NotNotX, NotX}));
}

TEST(SolverTest, NegatedDisjunction) {
  ConstraintContext Ctx;
  auto X = Ctx.atom();
  auto Y = Ctx.atom();
  auto XOrY = Ctx.disj(X, Y);
  auto NotXOrY = Ctx.neg(XOrY);

  // !(X v Y) ^ (X v Y)
  expectUnsatisfiable(solve({NotXOrY, XOrY}));
}

TEST(SolverTest, NegatedConjunction) {
  ConstraintContext Ctx;
  auto X = Ctx.atom();
  auto Y = Ctx.atom();
  auto XAndY = Ctx.conj(X, Y);
  auto NotXAndY = Ctx.neg(XAndY);

  // !(X ^ Y) ^ (X ^ Y)
  expectUnsatisfiable(solve({NotXAndY, XAndY}));
}

TEST(SolverTest, DisjunctionSameVars) {
  ConstraintContext Ctx;
  auto X = Ctx.atom();
  auto NotX = Ctx.neg(X);
  auto XOrNotX = Ctx.disj(X, NotX);

  // X v !X
  expectSatisfiable(solve({XOrNotX}), _);
}

TEST(SolverTest, ConjunctionSameVarsConflict) {
  ConstraintContext Ctx;
  auto X = Ctx.atom();
  auto NotX = Ctx.neg(X);
  auto XAndNotX = Ctx.conj(X, NotX);

  // X ^ !X
  expectUnsatisfiable(solve({XAndNotX}));
}

TEST(SolverTest, PureVar) {
  ConstraintContext Ctx;
  auto X = Ctx.atom();
  auto Y = Ctx.atom();
  auto NotX = Ctx.neg(X);
  auto NotXOrY = Ctx.disj(NotX, Y);
  auto NotY = Ctx.neg(Y);
  auto NotXOrNotY = Ctx.disj(NotX, NotY);

  // (!X v Y) ^ (!X v !Y)
  expectSatisfiable(
      solve({NotXOrY, NotXOrNotY}),
      UnorderedElementsAre(Pair(X, Solver::Result::Assignment::AssignedFalse),
                           Pair(Y, _)));
}

TEST(SolverTest, MustAssumeVarIsFalse) {
  ConstraintContext Ctx;
  auto X = Ctx.atom();
  auto Y = Ctx.atom();
  auto XOrY = Ctx.disj(X, Y);
  auto NotX = Ctx.neg(X);
  auto NotXOrY = Ctx.disj(NotX, Y);
  auto NotY = Ctx.neg(Y);
  auto NotXOrNotY = Ctx.disj(NotX, NotY);

  // (X v Y) ^ (!X v Y) ^ (!X v !Y)
  expectSatisfiable(
      solve({XOrY, NotXOrY, NotXOrNotY}),
      UnorderedElementsAre(Pair(X, Solver::Result::Assignment::AssignedFalse),
                           Pair(Y, Solver::Result::Assignment::AssignedTrue)));
}

TEST(SolverTest, DeepConflict) {
  ConstraintContext Ctx;
  auto X = Ctx.atom();
  auto Y = Ctx.atom();
  auto XOrY = Ctx.disj(X, Y);
  auto NotX = Ctx.neg(X);
  auto NotXOrY = Ctx.disj(NotX, Y);
  auto NotY = Ctx.neg(Y);
  auto NotXOrNotY = Ctx.disj(NotX, NotY);
  auto XOrNotY = Ctx.disj(X, NotY);

  // (X v Y) ^ (!X v Y) ^ (!X v !Y) ^ (X v !Y)
  expectUnsatisfiable(solve({XOrY, NotXOrY, NotXOrNotY, XOrNotY}));
}

TEST(SolverTest, IffSameVars) {
  ConstraintContext Ctx;
  auto X = Ctx.atom();
  auto XEqX = Ctx.iff(X, X);

  // X <=> X
  expectSatisfiable(solve({XEqX}), _);
}

TEST(SolverTest, IffDistinctVars) {
  ConstraintContext Ctx;
  auto X = Ctx.atom();
  auto Y = Ctx.atom();
  auto XEqY = Ctx.iff(X, Y);

  // X <=> Y
  expectSatisfiable(
      solve({XEqY}),
      AnyOf(UnorderedElementsAre(
                Pair(X, Solver::Result::Assignment::AssignedTrue),
                Pair(Y, Solver::Result::Assignment::AssignedTrue)),
            UnorderedElementsAre(
                Pair(X, Solver::Result::Assignment::AssignedFalse),
                Pair(Y, Solver::Result::Assignment::AssignedFalse))));
}

TEST(SolverTest, IffWithUnits) {
  ConstraintContext Ctx;
  auto X = Ctx.atom();
  auto Y = Ctx.atom();
  auto XEqY = Ctx.iff(X, Y);

  // (X <=> Y) ^ X ^ Y
  expectSatisfiable(
      solve({XEqY, X, Y}),
      UnorderedElementsAre(Pair(X, Solver::Result::Assignment::AssignedTrue),
                           Pair(Y, Solver::Result::Assignment::AssignedTrue)));
}

TEST(SolverTest, IffWithUnitsConflict) {
  ConstraintContext Ctx;
  auto X = Ctx.atom();
  auto Y = Ctx.atom();
  auto XEqY = Ctx.iff(X, Y);
  auto NotY = Ctx.neg(Y);

  // (X <=> Y) ^ X  !Y
  expectUnsatisfiable(solve({XEqY, X, NotY}));
}

TEST(SolverTest, IffTransitiveConflict) {
  ConstraintContext Ctx;
  auto X = Ctx.atom();
  auto Y = Ctx.atom();
  auto Z = Ctx.atom();
  auto XEqY = Ctx.iff(X, Y);
  auto YEqZ = Ctx.iff(Y, Z);
  auto NotX = Ctx.neg(X);

  // (X <=> Y) ^ (Y <=> Z) ^ Z ^ !X
  expectUnsatisfiable(solve({XEqY, YEqZ, Z, NotX}));
}

TEST(SolverTest, DeMorgan) {
  ConstraintContext Ctx;
  auto X = Ctx.atom();
  auto Y = Ctx.atom();
  auto Z = Ctx.atom();
  auto W = Ctx.atom();

  // !(X v Y) <=> !X ^ !Y
  auto A = Ctx.iff(Ctx.neg(Ctx.disj(X, Y)), Ctx.conj(Ctx.neg(X), Ctx.neg(Y)));

  // !(Z ^ W) <=> !Z v !W
  auto B = Ctx.iff(Ctx.neg(Ctx.conj(Z, W)), Ctx.disj(Ctx.neg(Z), Ctx.neg(W)));

  // A ^ B
  expectSatisfiable(solve({A, B}), _);
}

TEST(SolverTest, RespectsAdditionalConstraints) {
  ConstraintContext Ctx;
  auto X = Ctx.atom();
  auto Y = Ctx.atom();
  auto XEqY = Ctx.iff(X, Y);
  auto NotY = Ctx.neg(Y);

  // (X <=> Y) ^ X ^ !Y
  expectUnsatisfiable(solve({XEqY, X, NotY}));
}

TEST(SolverTest, ImplicationConflict) {
  ConstraintContext Ctx;
  auto X = Ctx.atom();
  auto Y = Ctx.atom();
  auto *XImplY = Ctx.impl(X, Y);
  auto *XAndNotY = Ctx.conj(X, Ctx.neg(Y));

  // X => Y ^ X ^ !Y
  expectUnsatisfiable(solve({XImplY, XAndNotY}));
}

} // namespace
