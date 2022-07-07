//===- unittests/Analysis/FlowSensitive/SolverTest.cpp --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Analysis/FlowSensitive/Solver.h"
#include "clang/Analysis/FlowSensitive/Value.h"
#include "clang/Analysis/FlowSensitive/WatchedLiteralsSolver.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <memory>
#include <utility>
#include <vector>

namespace {

using namespace clang;
using namespace dataflow;

using testing::_;
using testing::AnyOf;
using testing::Optional;
using testing::Pair;
using testing::UnorderedElementsAre;

class SolverTest : public ::testing::Test {
protected:
  // Checks if the conjunction of `Vals` is satisfiable and returns the
  // corresponding result.
  Solver::Result solve(llvm::DenseSet<BoolValue *> Vals) {
    return WatchedLiteralsSolver().solve(std::move(Vals));
  }

  // Creates an atomic boolean value.
  BoolValue *atom() {
    Vals.push_back(std::make_unique<AtomicBoolValue>());
    return Vals.back().get();
  }

  // Creates a boolean conjunction value.
  BoolValue *conj(BoolValue *LeftSubVal, BoolValue *RightSubVal) {
    Vals.push_back(
        std::make_unique<ConjunctionValue>(*LeftSubVal, *RightSubVal));
    return Vals.back().get();
  }

  // Creates a boolean disjunction value.
  BoolValue *disj(BoolValue *LeftSubVal, BoolValue *RightSubVal) {
    Vals.push_back(
        std::make_unique<DisjunctionValue>(*LeftSubVal, *RightSubVal));
    return Vals.back().get();
  }

  // Creates a boolean negation value.
  BoolValue *neg(BoolValue *SubVal) {
    Vals.push_back(std::make_unique<NegationValue>(*SubVal));
    return Vals.back().get();
  }

  // Creates a boolean implication value.
  BoolValue *impl(BoolValue *LeftSubVal, BoolValue *RightSubVal) {
    return disj(neg(LeftSubVal), RightSubVal);
  }

  // Creates a boolean biconditional value.
  BoolValue *iff(BoolValue *LeftSubVal, BoolValue *RightSubVal) {
    return conj(impl(LeftSubVal, RightSubVal), impl(RightSubVal, LeftSubVal));
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

private:
  std::vector<std::unique_ptr<BoolValue>> Vals;
};

TEST_F(SolverTest, Var) {
  auto X = atom();

  // X
  expectSatisfiable(
      solve({X}),
      UnorderedElementsAre(Pair(X, Solver::Result::Assignment::AssignedTrue)));
}

TEST_F(SolverTest, NegatedVar) {
  auto X = atom();
  auto NotX = neg(X);

  // !X
  expectSatisfiable(
      solve({NotX}),
      UnorderedElementsAre(Pair(X, Solver::Result::Assignment::AssignedFalse)));
}

TEST_F(SolverTest, UnitConflict) {
  auto X = atom();
  auto NotX = neg(X);

  // X ^ !X
  expectUnsatisfiable(solve({X, NotX}));
}

TEST_F(SolverTest, DistinctVars) {
  auto X = atom();
  auto Y = atom();
  auto NotY = neg(Y);

  // X ^ !Y
  expectSatisfiable(
      solve({X, NotY}),
      UnorderedElementsAre(Pair(X, Solver::Result::Assignment::AssignedTrue),
                           Pair(Y, Solver::Result::Assignment::AssignedFalse)));
}

TEST_F(SolverTest, DoubleNegation) {
  auto X = atom();
  auto NotX = neg(X);
  auto NotNotX = neg(NotX);

  // !!X ^ !X
  expectUnsatisfiable(solve({NotNotX, NotX}));
}

TEST_F(SolverTest, NegatedDisjunction) {
  auto X = atom();
  auto Y = atom();
  auto XOrY = disj(X, Y);
  auto NotXOrY = neg(XOrY);

  // !(X v Y) ^ (X v Y)
  expectUnsatisfiable(solve({NotXOrY, XOrY}));
}

TEST_F(SolverTest, NegatedConjunction) {
  auto X = atom();
  auto Y = atom();
  auto XAndY = conj(X, Y);
  auto NotXAndY = neg(XAndY);

  // !(X ^ Y) ^ (X ^ Y)
  expectUnsatisfiable(solve({NotXAndY, XAndY}));
}

TEST_F(SolverTest, DisjunctionSameVars) {
  auto X = atom();
  auto NotX = neg(X);
  auto XOrNotX = disj(X, NotX);

  // X v !X
  expectSatisfiable(solve({XOrNotX}), _);
}

TEST_F(SolverTest, ConjunctionSameVarsConflict) {
  auto X = atom();
  auto NotX = neg(X);
  auto XAndNotX = conj(X, NotX);

  // X ^ !X
  expectUnsatisfiable(solve({XAndNotX}));
}

TEST_F(SolverTest, PureVar) {
  auto X = atom();
  auto Y = atom();
  auto NotX = neg(X);
  auto NotXOrY = disj(NotX, Y);
  auto NotY = neg(Y);
  auto NotXOrNotY = disj(NotX, NotY);

  // (!X v Y) ^ (!X v !Y)
  expectSatisfiable(
      solve({NotXOrY, NotXOrNotY}),
      UnorderedElementsAre(Pair(X, Solver::Result::Assignment::AssignedFalse),
                           Pair(Y, _)));
}

TEST_F(SolverTest, MustAssumeVarIsFalse) {
  auto X = atom();
  auto Y = atom();
  auto XOrY = disj(X, Y);
  auto NotX = neg(X);
  auto NotXOrY = disj(NotX, Y);
  auto NotY = neg(Y);
  auto NotXOrNotY = disj(NotX, NotY);

  // (X v Y) ^ (!X v Y) ^ (!X v !Y)
  expectSatisfiable(
      solve({XOrY, NotXOrY, NotXOrNotY}),
      UnorderedElementsAre(Pair(X, Solver::Result::Assignment::AssignedFalse),
                           Pair(Y, Solver::Result::Assignment::AssignedTrue)));
}

TEST_F(SolverTest, DeepConflict) {
  auto X = atom();
  auto Y = atom();
  auto XOrY = disj(X, Y);
  auto NotX = neg(X);
  auto NotXOrY = disj(NotX, Y);
  auto NotY = neg(Y);
  auto NotXOrNotY = disj(NotX, NotY);
  auto XOrNotY = disj(X, NotY);

  // (X v Y) ^ (!X v Y) ^ (!X v !Y) ^ (X v !Y)
  expectUnsatisfiable(solve({XOrY, NotXOrY, NotXOrNotY, XOrNotY}));
}

TEST_F(SolverTest, IffSameVars) {
  auto X = atom();
  auto XEqX = iff(X, X);

  // X <=> X
  expectSatisfiable(solve({XEqX}), _);
}

TEST_F(SolverTest, IffDistinctVars) {
  auto X = atom();
  auto Y = atom();
  auto XEqY = iff(X, Y);

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

TEST_F(SolverTest, IffWithUnits) {
  auto X = atom();
  auto Y = atom();
  auto XEqY = iff(X, Y);

  // (X <=> Y) ^ X ^ Y
  expectSatisfiable(
      solve({XEqY, X, Y}),
      UnorderedElementsAre(Pair(X, Solver::Result::Assignment::AssignedTrue),
                           Pair(Y, Solver::Result::Assignment::AssignedTrue)));
}

TEST_F(SolverTest, IffWithUnitsConflict) {
  auto X = atom();
  auto Y = atom();
  auto XEqY = iff(X, Y);
  auto NotY = neg(Y);

  // (X <=> Y) ^ X  !Y
  expectUnsatisfiable(solve({XEqY, X, NotY}));
}

TEST_F(SolverTest, IffTransitiveConflict) {
  auto X = atom();
  auto Y = atom();
  auto Z = atom();
  auto XEqY = iff(X, Y);
  auto YEqZ = iff(Y, Z);
  auto NotX = neg(X);

  // (X <=> Y) ^ (Y <=> Z) ^ Z ^ !X
  expectUnsatisfiable(solve({XEqY, YEqZ, Z, NotX}));
}

TEST_F(SolverTest, DeMorgan) {
  auto X = atom();
  auto Y = atom();
  auto Z = atom();
  auto W = atom();

  // !(X v Y) <=> !X ^ !Y
  auto A = iff(neg(disj(X, Y)), conj(neg(X), neg(Y)));

  // !(Z ^ W) <=> !Z v !W
  auto B = iff(neg(conj(Z, W)), disj(neg(Z), neg(W)));

  // A ^ B
  expectSatisfiable(solve({A, B}), _);
}

TEST_F(SolverTest, RespectsAdditionalConstraints) {
  auto X = atom();
  auto Y = atom();
  auto XEqY = iff(X, Y);
  auto NotY = neg(Y);

  // (X <=> Y) ^ X ^ !Y
  expectUnsatisfiable(solve({XEqY, X, NotY}));
}

TEST_F(SolverTest, ImplicationConflict) {
  auto X = atom();
  auto Y = atom();
  auto *XImplY = impl(X, Y);
  auto *XAndNotY = conj(X, neg(Y));

  // X => Y ^ X ^ !Y
  expectUnsatisfiable(solve({XImplY, XAndNotY}));
}

} // namespace
