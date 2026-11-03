//===- unittests/Analysis/FlowSensitive/Z3SolverTest.cpp -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Analysis/FlowSensitive/Z3Solver.h"
#include "SolverTest.h"

namespace clang::dataflow::test {

#if LLVM_WITH_Z3
template <> Z3Solver SolverTest<Z3Solver>::createSolverWithLowTimeout() {
  return Z3Solver(10);
}

namespace {

INSTANTIATE_TYPED_TEST_SUITE_P(Z3SolverTest, SolverTest, Z3Solver, );

TEST(Z3SolverSpecificTest, ReuseSolverDoesNotOvercountCumulativeRlimit) {
  Arena A;
  auto Constraints = parseFormulas(A, R"(
     (!(V0 | V1) = (!V0 & !V1))
     (!(V2 & V3) = (!V2 | !V3))
  )");
  // Each solve uses ~800-1000 rlimit units. 10 solves use < 20000 rlimit units
  // in total (linear in N), whereas mistakenly summing cumulative
  // "rlimit count" stats across calls grows quadratically (~45x single-solve
  // cost) and would exceed 20000.
  Z3Solver Solver(20000);
  for (int I = 0; I < 10; ++I) {
    EXPECT_EQ(Solver.solve(Constraints).getStatus(),
              Solver::Result::Status::Satisfiable);
    EXPECT_FALSE(Solver.reachedLimit());
  }
}

TEST(Z3SolverSpecificTest,
     ExhaustingRlimitAcrossMultipleCallsSetsReachedLimit) {
  Arena A;
  auto Constraints = parseFormulas(A, R"(
     (!(V0 | V1) = (!V0 & !V1))
     (!(V2 & V3) = (!V2 | !V3))
  )");
  Z3Solver Solver(3000);
  bool SawSatisfiable = false;
  bool SawTimeout = false;
  for (int I = 0; I < 10; ++I) {
    Solver::Result Res = Solver.solve(Constraints);
    if (Res.getStatus() == Solver::Result::Status::Satisfiable) {
      SawSatisfiable = true;
      EXPECT_FALSE(SawTimeout);
      EXPECT_FALSE(Solver.reachedLimit());
    } else {
      EXPECT_EQ(Res.getStatus(), Solver::Result::Status::TimedOut);
      SawTimeout = true;
      EXPECT_TRUE(Solver.reachedLimit());
    }
  }
  EXPECT_TRUE(SawSatisfiable);
  EXPECT_TRUE(SawTimeout);
}

} // namespace

#endif // LLVM_WITH_Z3

} // namespace clang::dataflow::test
