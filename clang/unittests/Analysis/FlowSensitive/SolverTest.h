//===--- SolverTest.h - Type-parameterized test for solvers ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_ANALYSIS_FLOW_SENSITIVE_SOLVER_TEST_H_
#define LLVM_CLANG_ANALYSIS_FLOW_SENSITIVE_SOLVER_TEST_H_

#include "TestingSupport.h"
#include "clang/Analysis/FlowSensitive/Solver.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace clang::dataflow::test {

namespace {

constexpr auto AssignedTrue = Solver::Result::Assignment::AssignedTrue;
constexpr auto AssignedFalse = Solver::Result::Assignment::AssignedFalse;

using testing::_;
using testing::AnyOf;
using testing::Pair;
using testing::UnorderedElementsAre;

} // namespace

/// Type-parameterized test for implementations of the `Solver` interface.
/// To use:
/// 1.  Implement a specialization of `createSolverWithLowTimeout()` for the
///     solver you want to test.
/// 2.  Instantiate the test suite for the solver you want to test using
///     `INSTANTIATE_TYPED_TEST_SUITE_P()`.
/// See WatchedLiteralsSolverTest.cpp for an example.
template <typename SolverT> class SolverTest : public ::testing::Test {
protected:
  // Checks if the conjunction of `Vals` is satisfiable and returns the
  // corresponding result.
  Solver::Result solve(llvm::ArrayRef<const Formula *> Vals) {
    return SolverT().solve(Vals);
  }

  // Create a specialization for the solver type to test.
  SolverT createSolverWithLowTimeout();
};

TYPED_TEST_SUITE_P(SolverTest);

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

TYPED_TEST_P(SolverTest, Var) {
  ConstraintContext Ctx;
  auto X = Ctx.atom();

  // X
  EXPECT_THAT(this->solve({X}),
              sat(UnorderedElementsAre(Pair(X->getAtom(), AssignedTrue))));
}

TYPED_TEST_P(SolverTest, NegatedVar) {
  ConstraintContext Ctx;
  auto X = Ctx.atom();
  auto NotX = Ctx.neg(X);

  // !X
  EXPECT_THAT(this->solve({NotX}),
              sat(UnorderedElementsAre(Pair(X->getAtom(), AssignedFalse))));
}

TYPED_TEST_P(SolverTest, UnitConflict) {
  ConstraintContext Ctx;
  auto X = Ctx.atom();
  auto NotX = Ctx.neg(X);

  // X ^ !X
  EXPECT_THAT(this->solve({X, NotX}), unsat());
}

TYPED_TEST_P(SolverTest, DistinctVars) {
  ConstraintContext Ctx;
  auto X = Ctx.atom();
  auto Y = Ctx.atom();
  auto NotY = Ctx.neg(Y);

  // X ^ !Y
  EXPECT_THAT(this->solve({X, NotY}),
              sat(UnorderedElementsAre(Pair(X->getAtom(), AssignedTrue),
                                       Pair(Y->getAtom(), AssignedFalse))));
}

TYPED_TEST_P(SolverTest, DoubleNegation) {
  ConstraintContext Ctx;
  auto X = Ctx.atom();
  auto NotX = Ctx.neg(X);
  auto NotNotX = Ctx.neg(NotX);

  // !!X ^ !X
  EXPECT_THAT(this->solve({NotNotX, NotX}), unsat());
}

TYPED_TEST_P(SolverTest, NegatedDisjunction) {
  ConstraintContext Ctx;
  auto X = Ctx.atom();
  auto Y = Ctx.atom();
  auto XOrY = Ctx.disj(X, Y);
  auto NotXOrY = Ctx.neg(XOrY);

  // !(X v Y) ^ (X v Y)
  EXPECT_THAT(this->solve({NotXOrY, XOrY}), unsat());
}

TYPED_TEST_P(SolverTest, NegatedConjunction) {
  ConstraintContext Ctx;
  auto X = Ctx.atom();
  auto Y = Ctx.atom();
  auto XAndY = Ctx.conj(X, Y);
  auto NotXAndY = Ctx.neg(XAndY);

  // !(X ^ Y) ^ (X ^ Y)
  EXPECT_THAT(this->solve({NotXAndY, XAndY}), unsat());
}

TYPED_TEST_P(SolverTest, DisjunctionSameVarWithNegation) {
  ConstraintContext Ctx;
  auto X = Ctx.atom();
  auto NotX = Ctx.neg(X);
  auto XOrNotX = Ctx.disj(X, NotX);

  // X v !X
  EXPECT_THAT(this->solve({XOrNotX}), sat(_));
}

TYPED_TEST_P(SolverTest, DisjunctionSameVar) {
  ConstraintContext Ctx;
  auto X = Ctx.atom();
  auto XOrX = Ctx.disj(X, X);

  // X v X
  EXPECT_THAT(this->solve({XOrX}), sat(_));
}

TYPED_TEST_P(SolverTest, ConjunctionSameVarsConflict) {
  ConstraintContext Ctx;
  auto X = Ctx.atom();
  auto NotX = Ctx.neg(X);
  auto XAndNotX = Ctx.conj(X, NotX);

  // X ^ !X
  EXPECT_THAT(this->solve({XAndNotX}), unsat());
}

TYPED_TEST_P(SolverTest, ConjunctionSameVar) {
  ConstraintContext Ctx;
  auto X = Ctx.atom();
  auto XAndX = Ctx.conj(X, X);

  // X ^ X
  EXPECT_THAT(this->solve({XAndX}), sat(_));
}

TYPED_TEST_P(SolverTest, PureVar) {
  ConstraintContext Ctx;
  auto X = Ctx.atom();
  auto Y = Ctx.atom();
  auto NotX = Ctx.neg(X);
  auto NotXOrY = Ctx.disj(NotX, Y);
  auto NotY = Ctx.neg(Y);
  auto NotXOrNotY = Ctx.disj(NotX, NotY);

  // (!X v Y) ^ (!X v !Y)
  EXPECT_THAT(this->solve({NotXOrY, NotXOrNotY}),
              sat(UnorderedElementsAre(Pair(X->getAtom(), AssignedFalse),
                                       Pair(Y->getAtom(), _))));
}

TYPED_TEST_P(SolverTest, MustAssumeVarIsFalse) {
  ConstraintContext Ctx;
  auto X = Ctx.atom();
  auto Y = Ctx.atom();
  auto XOrY = Ctx.disj(X, Y);
  auto NotX = Ctx.neg(X);
  auto NotXOrY = Ctx.disj(NotX, Y);
  auto NotY = Ctx.neg(Y);
  auto NotXOrNotY = Ctx.disj(NotX, NotY);

  // (X v Y) ^ (!X v Y) ^ (!X v !Y)
  EXPECT_THAT(this->solve({XOrY, NotXOrY, NotXOrNotY}),
              sat(UnorderedElementsAre(Pair(X->getAtom(), AssignedFalse),
                                       Pair(Y->getAtom(), AssignedTrue))));
}

TYPED_TEST_P(SolverTest, DeepConflict) {
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
  EXPECT_THAT(this->solve({XOrY, NotXOrY, NotXOrNotY, XOrNotY}), unsat());
}

TYPED_TEST_P(SolverTest, IffIsEquivalentToDNF) {
  ConstraintContext Ctx;
  auto X = Ctx.atom();
  auto Y = Ctx.atom();
  auto NotX = Ctx.neg(X);
  auto NotY = Ctx.neg(Y);
  auto XIffY = Ctx.iff(X, Y);
  auto XIffYDNF = Ctx.disj(Ctx.conj(X, Y), Ctx.conj(NotX, NotY));
  auto NotEquivalent = Ctx.neg(Ctx.iff(XIffY, XIffYDNF));

  // !((X <=> Y) <=> ((X ^ Y) v (!X ^ !Y)))
  EXPECT_THAT(this->solve({NotEquivalent}), unsat());
}

TYPED_TEST_P(SolverTest, IffSameVars) {
  ConstraintContext Ctx;
  auto X = Ctx.atom();
  auto XEqX = Ctx.iff(X, X);

  // X <=> X
  EXPECT_THAT(this->solve({XEqX}), sat(_));
}

TYPED_TEST_P(SolverTest, IffDistinctVars) {
  ConstraintContext Ctx;
  auto X = Ctx.atom();
  auto Y = Ctx.atom();
  auto XEqY = Ctx.iff(X, Y);

  // X <=> Y
  EXPECT_THAT(
      this->solve({XEqY}),
      sat(AnyOf(UnorderedElementsAre(Pair(X->getAtom(), AssignedTrue),
                                     Pair(Y->getAtom(), AssignedTrue)),
                UnorderedElementsAre(Pair(X->getAtom(), AssignedFalse),
                                     Pair(Y->getAtom(), AssignedFalse)))));
}

TYPED_TEST_P(SolverTest, IffWithUnits) {
  ConstraintContext Ctx;
  auto X = Ctx.atom();
  auto Y = Ctx.atom();
  auto XEqY = Ctx.iff(X, Y);

  // (X <=> Y) ^ X ^ Y
  EXPECT_THAT(this->solve({XEqY, X, Y}),
              sat(UnorderedElementsAre(Pair(X->getAtom(), AssignedTrue),
                                       Pair(Y->getAtom(), AssignedTrue))));
}

TYPED_TEST_P(SolverTest, IffWithUnitsConflict) {
  Arena A;
  auto Constraints = parseFormulas(A, R"(
     (V0 = V1)
     V0
     !V1
  )");
  EXPECT_THAT(this->solve(Constraints), unsat());
}

TYPED_TEST_P(SolverTest, IffTransitiveConflict) {
  Arena A;
  auto Constraints = parseFormulas(A, R"(
     (V0 = V1)
     (V1 = V2)
     V2
     !V0
  )");
  EXPECT_THAT(this->solve(Constraints), unsat());
}

TYPED_TEST_P(SolverTest, DeMorgan) {
  Arena A;
  auto Constraints = parseFormulas(A, R"(
     (!(V0 | V1) = (!V0 & !V1))
     (!(V2 & V3) = (!V2 | !V3))
  )");
  EXPECT_THAT(this->solve(Constraints), sat(_));
}

TYPED_TEST_P(SolverTest, RespectsAdditionalConstraints) {
  Arena A;
  auto Constraints = parseFormulas(A, R"(
     (V0 = V1)
     V0
     !V1
  )");
  EXPECT_THAT(this->solve(Constraints), unsat());
}

TYPED_TEST_P(SolverTest, ImplicationIsEquivalentToDNF) {
  Arena A;
  auto Constraints = parseFormulas(A, R"(
     !((V0 => V1) = (!V0 | V1))
  )");
  EXPECT_THAT(this->solve(Constraints), unsat());
}

TYPED_TEST_P(SolverTest, ImplicationConflict) {
  Arena A;
  auto Constraints = parseFormulas(A, R"(
     (V0 => V1)
     (V0 & !V1)
  )");
  EXPECT_THAT(this->solve(Constraints), unsat());
}

TYPED_TEST_P(SolverTest, ReachedLimitsReflectsTimeouts) {
  Arena A;
  auto Constraints = parseFormulas(A, R"(
     (!(V0 | V1) = (!V0 & !V1))
     (!(V2 & V3) = (!V2 & !V3))
  )");
  TypeParam solver = this->createSolverWithLowTimeout();
  ASSERT_EQ(solver.solve(Constraints).getStatus(),
            Solver::Result::Status::TimedOut);
  EXPECT_TRUE(solver.reachedLimit());
}

TYPED_TEST_P(SolverTest, SimpleButLargeContradiction) {
  // This test ensures that the solver takes a short-cut on known
  // contradictory inputs, without using max_iterations. At the time
  // this test is added, formulas that are easily recognized to be
  // contradictory at CNF construction time would lead to timeout.
  TypeParam solver = this->createSolverWithLowTimeout();
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

REGISTER_TYPED_TEST_SUITE_P(
    SolverTest, Var, NegatedVar, UnitConflict, DistinctVars, DoubleNegation,
    NegatedDisjunction, NegatedConjunction, DisjunctionSameVarWithNegation,
    DisjunctionSameVar, ConjunctionSameVarsConflict, ConjunctionSameVar,
    PureVar, MustAssumeVarIsFalse, DeepConflict, IffIsEquivalentToDNF,
    IffSameVars, IffDistinctVars, IffWithUnits, IffWithUnitsConflict,
    IffTransitiveConflict, DeMorgan, RespectsAdditionalConstraints,
    ImplicationIsEquivalentToDNF, ImplicationConflict,
    ReachedLimitsReflectsTimeouts, SimpleButLargeContradiction);

} // namespace clang::dataflow::test

#endif // LLVM_CLANG_ANALYSIS_FLOW_SENSITIVE_TESTING_SUPPORT_H_
