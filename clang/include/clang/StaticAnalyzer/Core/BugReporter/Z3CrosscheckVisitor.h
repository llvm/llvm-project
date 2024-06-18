//===- Z3CrosscheckVisitor.h - Crosscheck reports with Z3 -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  This file defines the visitor and utilities around it for Z3 report
//  refutation.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_STATICANALYZER_CORE_BUGREPORTER_Z3CROSSCHECKVISITOR_H
#define LLVM_CLANG_STATICANALYZER_CORE_BUGREPORTER_Z3CROSSCHECKVISITOR_H

#include "clang/StaticAnalyzer/Core/BugReporter/BugReporterVisitors.h"

namespace clang::ento {

/// The bug visitor will walk all the nodes in a path and collect all the
/// constraints. When it reaches the root node, will create a refutation
/// manager and check if the constraints are satisfiable.
class Z3CrosscheckVisitor final : public BugReporterVisitor {
public:
  struct Z3Result {
    std::optional<bool> IsSAT = std::nullopt;
  };
  explicit Z3CrosscheckVisitor(Z3CrosscheckVisitor::Z3Result &Result);

  void Profile(llvm::FoldingSetNodeID &ID) const override;

  PathDiagnosticPieceRef VisitNode(const ExplodedNode *N,
                                   BugReporterContext &BRC,
                                   PathSensitiveBugReport &BR) override;

  void finalizeVisitor(BugReporterContext &BRC, const ExplodedNode *EndPathNode,
                       PathSensitiveBugReport &BR) override;

private:
  void addConstraints(const ExplodedNode *N,
                      bool OverwriteConstraintsOnExistingSyms);

  /// Holds the constraints in a given path.
  ConstraintMap Constraints;
  Z3Result &Result;
};

/// The oracle will decide if a report should be accepted or rejected based on
/// the results of the Z3 solver.
class Z3CrosscheckOracle {
public:
  enum Z3Decision {
    AcceptReport, // The report was SAT.
    RejectReport, // The report was UNSAT or UNDEF.
  };

  /// Makes a decision for accepting or rejecting the report based on the
  /// result of the corresponding Z3 query.
  static Z3Decision
  interpretQueryResult(const Z3CrosscheckVisitor::Z3Result &Query);
};

} // namespace clang::ento

#endif // LLVM_CLANG_STATICANALYZER_CORE_BUGREPORTER_Z3CROSSCHECKVISITOR_H
