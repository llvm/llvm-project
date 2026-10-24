//===- Z3Solver.cpp ---------------------------------------------*- C++ -*-===//
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

#include <cassert>
#include <vector>

#include "clang/Analysis/FlowSensitive/CNFFormula.h"
#include "clang/Analysis/FlowSensitive/Formula.h"
#include "clang/Analysis/FlowSensitive/Solver.h"
#include "clang/Analysis/FlowSensitive/Z3Solver.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/SMTAPI.h"

namespace clang {
namespace dataflow {

static llvm::DenseMap<Atom, Solver::Result::Assignment>
buildSolution(const llvm::SMTSolverRef &Solver,
              const llvm::DenseMap<Variable, Atom> &Atomics,
              llvm::ArrayRef<llvm::SMTExprRef> BoolVars) {
  llvm::DenseMap<Atom, Solver::Result::Assignment> Solution;

  for (auto &Atomic : Atomics) {
    llvm::SMTExprRef Var = BoolVars[Atomic.first];
    llvm::APSInt Result = llvm::APSInt(1);
    if (!Solver->getInterpretation(Var, Result)) {
      // Model doesn't assign a value -- just pick one.
      Solution[Atomic.second] = Solver::Result::Assignment::AssignedTrue;
    } else {
      Solution[Atomic.second] = Result.getBoolValue()
                                    ? Solver::Result::Assignment::AssignedTrue
                                    : Solver::Result::Assignment::AssignedFalse;
    }
  }

  return Solution;
}

void Z3Solver::updateRlimitUsed(const llvm::SMTSolverRef &Solver) {
  if (Rlimit > 0) {
    std::uint32_t NextRlimitUsed =
        Solver->getStatistics()->getUnsigned("rlimit count");
    std::uint32_t Next = RlimitUsed + NextRlimitUsed;
    // If overflow, clamp
    if (Next < RlimitUsed)
      RlimitUsed = Rlimit;
    // If going past Rlimit, clamp
    else if (Next > Rlimit)
      RlimitUsed = Rlimit;
    else
      RlimitUsed = Next;
  }
}

Solver::Result Z3Solver::solve(llvm::ArrayRef<const Formula *> Constraints) {
  if (Constraints.empty()) {
    return Result::Satisfiable(llvm::DenseMap<Atom, Result::Assignment>());
  }
  if (Rlimit > 0 && RlimitUsed == Rlimit) {
    return Result::TimedOut();
  }

  llvm::DenseMap<Variable, Atom> Atomics;
  CNFFormula CNF = buildCNF(Constraints, Atomics);

  if (CNF.knownContradictory())
    return Result::Unsatisfiable();

  Solver->reset();
  Solver->setBoolParam("model", true);
  if (Rlimit != 0) {
    std::uint32_t Remaining = Rlimit - RlimitUsed;
    Solver->setUnsignedParam("rlimit", Remaining);
  }

  std::vector<llvm::SMTExprRef> BoolVars;
  BoolVars.resize(CNF.largestVar() + 1);
  llvm::SmallString<16> Str;
  llvm::raw_svector_ostream OS(Str);
  llvm::SMTSortRef BoolSort = Solver->getBoolSort();
  for (Variable V = 1; V <= CNF.largestVar(); ++V) {
    OS << "v" << V;
    BoolVars[V] = Solver->mkSymbol(Str.c_str(), BoolSort);
    Str.clear();
  }

  for (size_t ClauseId = 1; ClauseId <= CNF.numClauses(); ++ClauseId) {
    llvm::SMTExprRef Clause = nullptr;
    for (Literal Lit : CNF.clauseLiterals(ClauseId)) {
      llvm::SMTExprRef Var = BoolVars[var(Lit)];
      if (isNegLit(Lit))
        Var = Solver->mkNot(Var);
      if (Clause == nullptr) {
        Clause = Var;
      } else {
        Clause = Solver->mkOr(Clause, Var);
      }
    }
    if (Clause == nullptr) {
      Solver->addConstraint(Solver->mkBoolean(false));
    } else {
      Solver->addConstraint(Clause);
    }
  }

  std::optional<bool> IsSat = Solver->check();

  // Update rlimit
  updateRlimitUsed(Solver);

  if (IsSat.has_value()) {
    if (IsSat.value()) {
      return Result::Satisfiable(buildSolution(Solver, Atomics, BoolVars));
    }
    return Result::Unsatisfiable();
  }
  EverTimedOut = true;
  return Result::TimedOut();
}

} // namespace dataflow
} // namespace clang
