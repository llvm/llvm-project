//===- Z3Solver.h -----------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  This file defines a Z3-based SAT solver implementation that can be used by
//  dataflow analyses.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_ANALYSIS_FLOWSENSITIVE_Z3SOLVER_H
#define LLVM_CLANG_ANALYSIS_FLOWSENSITIVE_Z3SOLVER_H

#include "clang/Analysis/FlowSensitive/Formula.h"
#include "clang/Analysis/FlowSensitive/Solver.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/SMTAPI.h"

namespace clang {
namespace dataflow {

/// Wrapper around Z3 to use it as a SAT solver (not full SMT).
class Z3Solver : public Solver {

public:
  Z3Solver() : Solver(llvm::CreateZ3Solver()) {
    Solver->setBoolParam("model", true);
  }

  // Creates a Z3-based solver with the given "rlimit" (0 is unlimited).
  // Rlimit is a count of operations, which may be more deterministic
  // than time. However, it could change across Z3 versions:
  // https://stackoverflow.com/questions/45457131/what-is-the-relation-between-options-rlimit-and-timeout/45458651#45458651
  explicit Z3Solver(std::uint32_t Rlimit)
      : Rlimit(Rlimit), Solver(llvm::CreateZ3Solver()) {
    Solver->setBoolParam("model", true);
  }

  Result solve(llvm::ArrayRef<const Formula *> Vals) override;

  bool reachedLimit() const override { return EverTimedOut; }

private:
  std::uint32_t Rlimit = 0;     // 0 is unlimited
  std::uint32_t RlimitUsed = 0; // track rlimit used (if Rlimit is non-zero)
  bool EverTimedOut = false;
  llvm::SMTSolverRef Solver;

  void updateRlimitUsed(const llvm::SMTSolverRef &Solver);
};

} // namespace dataflow
} // namespace clang

#endif // LLVM_CLANG_ANALYSIS_FLOWSENSITIVE_Z3SOLVER_H
