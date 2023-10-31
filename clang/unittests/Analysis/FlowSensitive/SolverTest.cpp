//===- unittests/Analysis/FlowSensitive/SolverTest.cpp --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <utility>

#include "TestingSupport.h"
#include "clang/Analysis/FlowSensitive/Arena.h"
#include "clang/Analysis/FlowSensitive/Formula.h"
#include "clang/Analysis/FlowSensitive/Solver.h"
#include "clang/Analysis/FlowSensitive/WatchedLiteralsSolver.h"
#include "clang/Basic/LLVM.h"
#include "llvm/ADT/ArrayRef.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <vector>

namespace {

using namespace clang;
using namespace dataflow;

using test::ConstraintContext;
using test::parseFormulas;
using testing::_;
using testing::AnyOf;
using testing::Pair;
using testing::UnorderedElementsAre;

constexpr auto AssignedTrue = Solver::Result::Assignment::AssignedTrue;
constexpr auto AssignedFalse = Solver::Result::Assignment::AssignedFalse;

// Checks if the conjunction of `Vals` is satisfiable and returns the
// corresponding result.
Solver::Result solve(llvm::ArrayRef<const Formula *> Vals) {
  return WatchedLiteralsSolver().solve(Vals);
}

MATCHER(unsat, "") {
  return arg.getStatus() == Solver::Result::Status::Unsatisfiable;
}
MATCHER_P(sat, SolutionMatcher,
          "is satisfiable, where solution " +
              (testing::DescribeMatcher<
                  llvm::DenseMap<Atom, Solver::Result::Assignment>>(
                  SolutionMatcher))) {
  if (arg.getStatus() != Solver::Result::Status::Satisfiable)
    return false;
  auto Solution = *arg.getSolution();
  return testing::ExplainMatchResult(SolutionMatcher, Solution,
                                     result_listener);
}

TEST(SolverTest, Var) {
  ConstraintContext Ctx;
  auto X = Ctx.atom();

  // X
  EXPECT_THAT(solve({X}),
              sat(UnorderedElementsAre(Pair(X->getAtom(), AssignedTrue))));
}

TEST(SolverTest, NegatedVar) {
  ConstraintContext Ctx;
  auto X = Ctx.atom();
  auto NotX = Ctx.neg(X);

  // !X
  EXPECT_THAT(solve({NotX}),
              sat(UnorderedElementsAre(Pair(X->getAtom(), AssignedFalse))));
}

TEST(SolverTest, UnitConflict) {
  ConstraintContext Ctx;
  auto X = Ctx.atom();
  auto NotX = Ctx.neg(X);

  // X ^ !X
  EXPECT_THAT(solve({X, NotX}), unsat());
}

TEST(SolverTest, DistinctVars) {
  ConstraintContext Ctx;
  auto X = Ctx.atom();
  auto Y = Ctx.atom();
  auto NotY = Ctx.neg(Y);

  // X ^ !Y
  EXPECT_THAT(solve({X, NotY}),
              sat(UnorderedElementsAre(Pair(X->getAtom(), AssignedTrue),
                                       Pair(Y->getAtom(), AssignedFalse))));
}

TEST(SolverTest, DoubleNegation) {
  ConstraintContext Ctx;
  auto X = Ctx.atom();
  auto NotX = Ctx.neg(X);
  auto NotNotX = Ctx.neg(NotX);

  // !!X ^ !X
  EXPECT_THAT(solve({NotNotX, NotX}), unsat());
}

TEST(SolverTest, NegatedDisjunction) {
  ConstraintContext Ctx;
  auto X = Ctx.atom();
  auto Y = Ctx.atom();
  auto XOrY = Ctx.disj(X, Y);
  auto NotXOrY = Ctx.neg(XOrY);

  // !(X v Y) ^ (X v Y)
  EXPECT_THAT(solve({NotXOrY, XOrY}), unsat());
}

TEST(SolverTest, NegatedConjunction) {
  ConstraintContext Ctx;
  auto X = Ctx.atom();
  auto Y = Ctx.atom();
  auto XAndY = Ctx.conj(X, Y);
  auto NotXAndY = Ctx.neg(XAndY);

  // !(X ^ Y) ^ (X ^ Y)
  EXPECT_THAT(solve({NotXAndY, XAndY}), unsat());
}

TEST(SolverTest, DisjunctionSameVarWithNegation) {
  ConstraintContext Ctx;
  auto X = Ctx.atom();
  auto NotX = Ctx.neg(X);
  auto XOrNotX = Ctx.disj(X, NotX);

  // X v !X
  EXPECT_THAT(solve({XOrNotX}), sat(_));
}

TEST(SolverTest, DisjunctionSameVar) {
  ConstraintContext Ctx;
  auto X = Ctx.atom();
  auto XOrX = Ctx.disj(X, X);

  // X v X
  EXPECT_THAT(solve({XOrX}), sat(_));
}

TEST(SolverTest, ConjunctionSameVarsConflict) {
  ConstraintContext Ctx;
  auto X = Ctx.atom();
  auto NotX = Ctx.neg(X);
  auto XAndNotX = Ctx.conj(X, NotX);

  // X ^ !X
  EXPECT_THAT(solve({XAndNotX}), unsat());
}

TEST(SolverTest, ConjunctionSameVar) {
  ConstraintContext Ctx;
  auto X = Ctx.atom();
  auto XAndX = Ctx.conj(X, X);

  // X ^ X
  EXPECT_THAT(solve({XAndX}), sat(_));
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
  EXPECT_THAT(solve({NotXOrY, NotXOrNotY}),
              sat(UnorderedElementsAre(Pair(X->getAtom(), AssignedFalse),
                                       Pair(Y->getAtom(), _))));
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
  EXPECT_THAT(solve({XOrY, NotXOrY, NotXOrNotY}),
              sat(UnorderedElementsAre(Pair(X->getAtom(), AssignedFalse),
                                       Pair(Y->getAtom(), AssignedTrue))));
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
  EXPECT_THAT(solve({XOrY, NotXOrY, NotXOrNotY, XOrNotY}), unsat());
}

TEST(SolverTest, IffIsEquivalentToDNF) {
  ConstraintContext Ctx;
  auto X = Ctx.atom();
  auto Y = Ctx.atom();
  auto NotX = Ctx.neg(X);
  auto NotY = Ctx.neg(Y);
  auto XIffY = Ctx.iff(X, Y);
  auto XIffYDNF = Ctx.disj(Ctx.conj(X, Y), Ctx.conj(NotX, NotY));
  auto NotEquivalent = Ctx.neg(Ctx.iff(XIffY, XIffYDNF));

  // !((X <=> Y) <=> ((X ^ Y) v (!X ^ !Y)))
  EXPECT_THAT(solve({NotEquivalent}), unsat());
}

TEST(SolverTest, IffSameVars) {
  ConstraintContext Ctx;
  auto X = Ctx.atom();
  auto XEqX = Ctx.iff(X, X);

  // X <=> X
  EXPECT_THAT(solve({XEqX}), sat(_));
}

TEST(SolverTest, IffDistinctVars) {
  ConstraintContext Ctx;
  auto X = Ctx.atom();
  auto Y = Ctx.atom();
  auto XEqY = Ctx.iff(X, Y);

  // X <=> Y
  EXPECT_THAT(
      solve({XEqY}),
      sat(AnyOf(UnorderedElementsAre(Pair(X->getAtom(), AssignedTrue),
                                     Pair(Y->getAtom(), AssignedTrue)),
                UnorderedElementsAre(Pair(X->getAtom(), AssignedFalse),
                                     Pair(Y->getAtom(), AssignedFalse)))));
}

TEST(SolverTest, IffWithUnits) {
  ConstraintContext Ctx;
  auto X = Ctx.atom();
  auto Y = Ctx.atom();
  auto XEqY = Ctx.iff(X, Y);

  // (X <=> Y) ^ X ^ Y
  EXPECT_THAT(solve({XEqY, X, Y}),
              sat(UnorderedElementsAre(Pair(X->getAtom(), AssignedTrue),
                                       Pair(Y->getAtom(), AssignedTrue))));
}

TEST(SolverTest, IffWithUnitsConflict) {
  Arena A;
  auto Constraints = parseFormulas(A, R"(
     (V0 = V1)
     V0
     !V1
  )");
  EXPECT_THAT(solve(Constraints), unsat());
}

TEST(SolverTest, IffTransitiveConflict) {
  Arena A;
  auto Constraints = parseFormulas(A, R"(
     (V0 = V1)
     (V1 = V2)
     V2
     !V0
  )");
  EXPECT_THAT(solve(Constraints), unsat());
}

TEST(SolverTest, DeMorgan) {
  Arena A;
  auto Constraints = parseFormulas(A, R"(
     (!(V0 | V1) = (!V0 & !V1))
     (!(V2 & V3) = (!V2 | !V3))
  )");
  EXPECT_THAT(solve(Constraints), sat(_));
}

TEST(SolverTest, RespectsAdditionalConstraints) {
  Arena A;
  auto Constraints = parseFormulas(A, R"(
     (V0 = V1)
     V0
     !V1
  )");
  EXPECT_THAT(solve(Constraints), unsat());
}

TEST(SolverTest, ImplicationIsEquivalentToDNF) {
  Arena A;
  auto Constraints = parseFormulas(A, R"(
     !((V0 => V1) = (!V0 | V1))
  )");
  EXPECT_THAT(solve(Constraints), unsat());
}

TEST(SolverTest, ImplicationConflict) {
  Arena A;
  auto Constraints = parseFormulas(A, R"(
     (V0 => V1)
     (V0 & !V1)
  )");
  EXPECT_THAT(solve(Constraints), unsat());
}

TEST(SolverTest, ReachedLimitsReflectsTimeouts) {
  Arena A;
  auto Constraints = parseFormulas(A, R"(
     (!(V0 | V1) = (!V0 & !V1))
     (!(V2 & V3) = (!V2 & !V3))
  )");
  WatchedLiteralsSolver solver(10);
  ASSERT_EQ(solver.solve(Constraints).getStatus(),
            Solver::Result::Status::TimedOut);
  EXPECT_TRUE(solver.reachedLimit());
}

TEST(SolverTest, SimpleButLargeContradiction) {
  // This test ensures that the solver takes a short-cut on known
  // contradictory inputs, without using max_iterations. At the time
  // this test is added, formulas that are easily recognized to be
  // contradictory at CNF construction time would lead to timeout.
  WatchedLiteralsSolver solver(10);
  ConstraintContext Ctx;
  auto first = Ctx.atom();
  auto last = first;
  for (int i = 1; i < 10000; ++i) {
    last = Ctx.conj(last, Ctx.atom());
  }
  last = Ctx.conj(Ctx.neg(first), last);
  ASSERT_EQ(solver.solve({last}).getStatus(),
            Solver::Result::Status::Unsatisfiable);
  EXPECT_FALSE(solver.reachedLimit());

  first = Ctx.atom();
  last = Ctx.neg(first);
  for (int i = 1; i < 10000; ++i) {
    last = Ctx.conj(last, Ctx.neg(Ctx.atom()));
  }
  last = Ctx.conj(first, last);
  ASSERT_EQ(solver.solve({last}).getStatus(),
            Solver::Result::Status::Unsatisfiable);
  EXPECT_FALSE(solver.reachedLimit());
}

} // namespace
