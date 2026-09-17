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

#ifdef LLVM_WITH_Z3
template <> Z3Solver SolverTest<Z3Solver>::createSolverWithLowTimeout() {
  return Z3Solver(1);
}

namespace {

INSTANTIATE_TYPED_TEST_SUITE_P(Z3SolverTest, SolverTest, Z3Solver, );

} // namespace

#endif // LLVM_WITH_Z3

} // namespace clang::dataflow::test
