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
#include <cstddef>
#include <cstdint>
#include <optional>
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
#include "llvm/Support/raw_ostream.h"

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

Solver::Result Z3Solver::solve(llvm::ArrayRef<const Formula *> Constraints) {
  if (Constraints.empty()) {
    return Result::Satisfiable(llvm::DenseMap<Atom, Result::Assignment>());
  }
  if (EverTimedOut) {
    return Result::TimedOut();
  }

  llvm::DenseMap<Variable, Atom> Atomics;
  CNFFormula CNF = buildCNF(Constraints, Atomics);

  if (CNF.knownContradictory())
    return Result::Unsatisfiable();

  if (Solver == nullptr) {
    // Lazily create the solver.
    Solver = llvm::CreateZ3Solver();
    Solver->setBoolParam("model", true);
    // `buildCNF` already lowers formulas into unit-simplified propositional
    // 3-CNF with no SMT theories, so disable Z3's general-purpose SMT
    // preprocessors and relevancy tracking to avoid per-check overhead.
    Solver->setUnsignedParam("relevancy", 0);
    Solver->setBoolParam("propagate_values", false);
    Solver->setBoolParam("elim_unconstrained", false);
    Solver->setBoolParam("solve_eqs", false);
    Solver->setBoolParam("bound_simplifier", false);
    // Cache sort and false
    BoolSort = Solver->getBoolSort();
    FalseExpr = Solver->mkBoolean(false);
    // BoolVars numbering starts at 1
    BoolVars.push_back(nullptr);
    NegatedBoolVars.push_back(nullptr);
  } else {
    Solver->reset();
  }

  llvm::SmallString<16> Str;
  llvm::raw_svector_ostream OS(Str);
  for (Variable V = BoolVars.size(); V <= CNF.largestVar(); ++V) {
    OS << "v" << V;
    BoolVars.push_back(Solver->mkSymbol(Str.c_str(), BoolSort));
    Str.clear();
  }
  if (NegatedBoolVars.size() <= CNF.largestVar()) {
    NegatedBoolVars.resize(CNF.largestVar() + 1, nullptr);
  }

  for (size_t ClauseId = 1; ClauseId <= CNF.numClauses(); ++ClauseId) {
    llvm::SMTExprRef Clause = nullptr;
    for (Literal Lit : CNF.clauseLiterals(ClauseId)) {
      Variable V = var(Lit);
      llvm::SMTExprRef Var = BoolVars[V];
      if (isNegLit(Lit)) {
        llvm::SMTExprRef &NegVar = NegatedBoolVars[V];
        if (NegVar == nullptr) {
          NegVar = Solver->mkNot(Var);
        }
        Var = NegVar;
      }
      if (Clause == nullptr) {
        Clause = Var;
      } else {
        Clause = Solver->mkOr(Clause, Var);
      }
    }
    if (Clause == nullptr) {
      Solver->addConstraint(FalseExpr);
    } else {
      Solver->addConstraint(Clause);
    }
  }

  if (Rlimit > 0) {
    // NOTE: Z3's "rlimit count" statistic is cumulative across
    // `Solver->reset()` calls on the same Z3 context, and `addConstraint`
    // also increments the counter, so query it right before `check()`.
    std::uint32_t RlimitUsed =
        Solver->getStatistics()->getUnsigned("rlimit count");
    if (RlimitUsed >= Rlimit) {
      EverTimedOut = true;
      return Result::TimedOut();
    }
    Solver->setUnsignedParam("rlimit", Rlimit - RlimitUsed);
  }

  std::optional<bool> IsSat = Solver->check();

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
