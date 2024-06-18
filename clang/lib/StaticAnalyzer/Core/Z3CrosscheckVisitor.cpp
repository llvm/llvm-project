//===- Z3CrosscheckVisitor.cpp - Crosscheck reports with Z3 -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  This file declares the visitor and utilities around it for Z3 report
//  refutation.
//
//===----------------------------------------------------------------------===//

#include "clang/StaticAnalyzer/Core/BugReporter/Z3CrosscheckVisitor.h"
#include "clang/StaticAnalyzer/Core/BugReporter/BugReporter.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/SMTConv.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Support/SMTAPI.h"

#define DEBUG_TYPE "Z3CrosscheckOracle"

STATISTIC(NumZ3QueriesDone, "Number of Z3 queries done");
STATISTIC(NumTimesZ3TimedOut, "Number of times Z3 query timed out");

STATISTIC(NumTimesZ3QueryAcceptsReport,
          "Number of Z3 queries accepting a report");
STATISTIC(NumTimesZ3QueryRejectReport,
          "Number of Z3 queries rejecting a report");

using namespace clang;
using namespace ento;

Z3CrosscheckVisitor::Z3CrosscheckVisitor(Z3CrosscheckVisitor::Z3Result &Result)
    : Constraints(ConstraintMap::Factory().getEmptyMap()), Result(Result) {}

void Z3CrosscheckVisitor::finalizeVisitor(BugReporterContext &BRC,
                                          const ExplodedNode *EndPathNode,
                                          PathSensitiveBugReport &BR) {
  // Collect new constraints
  addConstraints(EndPathNode, /*OverwriteConstraintsOnExistingSyms=*/true);

  // Create a refutation manager
  llvm::SMTSolverRef RefutationSolver = llvm::CreateZ3Solver();
  RefutationSolver->setBoolParam("model", true);        // Enable model finding
  RefutationSolver->setUnsignedParam("timeout", 15000); // ms

  ASTContext &Ctx = BRC.getASTContext();

  // Add constraints to the solver
  for (const auto &[Sym, Range] : Constraints) {
    auto RangeIt = Range.begin();

    llvm::SMTExprRef SMTConstraints = SMTConv::getRangeExpr(
        RefutationSolver, Ctx, Sym, RangeIt->From(), RangeIt->To(),
        /*InRange=*/true);
    while ((++RangeIt) != Range.end()) {
      SMTConstraints = RefutationSolver->mkOr(
          SMTConstraints, SMTConv::getRangeExpr(RefutationSolver, Ctx, Sym,
                                                RangeIt->From(), RangeIt->To(),
                                                /*InRange=*/true));
    }
    RefutationSolver->addConstraint(SMTConstraints);
  }

  // And check for satisfiability
  std::optional<bool> IsSAT = RefutationSolver->check();
  Result = Z3Result{IsSAT};
}

void Z3CrosscheckVisitor::addConstraints(
    const ExplodedNode *N, bool OverwriteConstraintsOnExistingSyms) {
  // Collect new constraints
  ConstraintMap NewCs = getConstraintMap(N->getState());
  ConstraintMap::Factory &CF = N->getState()->get_context<ConstraintMap>();

  // Add constraints if we don't have them yet
  for (auto const &[Sym, Range] : NewCs) {
    if (!Constraints.contains(Sym)) {
      // This symbol is new, just add the constraint.
      Constraints = CF.add(Constraints, Sym, Range);
    } else if (OverwriteConstraintsOnExistingSyms) {
      // Overwrite the associated constraint of the Symbol.
      Constraints = CF.remove(Constraints, Sym);
      Constraints = CF.add(Constraints, Sym, Range);
    }
  }
}

PathDiagnosticPieceRef
Z3CrosscheckVisitor::VisitNode(const ExplodedNode *N, BugReporterContext &,
                               PathSensitiveBugReport &) {
  addConstraints(N, /*OverwriteConstraintsOnExistingSyms=*/false);
  return nullptr;
}

void Z3CrosscheckVisitor::Profile(llvm::FoldingSetNodeID &ID) const {
  static int Tag = 0;
  ID.AddPointer(&Tag);
}

Z3CrosscheckOracle::Z3Decision Z3CrosscheckOracle::interpretQueryResult(
    const Z3CrosscheckVisitor::Z3Result &Query) {
  ++NumZ3QueriesDone;

  if (!Query.IsSAT.has_value()) {
    // For backward compatibility, let's accept the first timeout.
    ++NumTimesZ3TimedOut;
    return AcceptReport;
  }

  if (Query.IsSAT.value()) {
    ++NumTimesZ3QueryAcceptsReport;
    return AcceptReport; // sat
  }

  ++NumTimesZ3QueryRejectReport;
  return RejectReport; // unsat
}
