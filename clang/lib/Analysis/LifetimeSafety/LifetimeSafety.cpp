//===- LifetimeSafety.cpp - C++ Lifetime Safety Analysis -*--------- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the main LifetimeSafetyAnalysis class, which coordinates
// the various components (fact generation, loan propagation, live origins
// analysis, and checking) to detect lifetime safety violations in C++ code.
//
//===----------------------------------------------------------------------===//
#include "clang/Analysis/Analyses/LifetimeSafety/LifetimeSafety.h"
#include "clang/AST/Decl.h"
#include "clang/AST/Expr.h"
#include "clang/AST/Type.h"
#include "clang/Analysis/Analyses/LifetimeSafety/Checker.h"
#include "clang/Analysis/Analyses/LifetimeSafety/Dataflow.h"
#include "clang/Analysis/Analyses/LifetimeSafety/Facts.h"
#include "clang/Analysis/Analyses/LifetimeSafety/FactsGenerator.h"
#include "clang/Analysis/Analyses/LifetimeSafety/LiveOrigins.h"
#include "clang/Analysis/Analyses/LifetimeSafety/LoanPropagation.h"
#include "clang/Analysis/Analyses/LifetimeSafety/Loans.h"
#include "clang/Analysis/Analyses/LifetimeSafety/Origins.h"
#include "clang/Analysis/AnalysisDeclContext.h"
#include "clang/Analysis/CFG.h"
#include "llvm/ADT/FoldingSet.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/TimeProfiler.h"
#include <memory>
#include <optional>

namespace clang::lifetimes {
namespace internal {

// We need this here for unique_ptr with forward declared class.
LifetimeSafetyAnalysis::~LifetimeSafetyAnalysis() = default;

LifetimeSafetyAnalysis::LifetimeSafetyAnalysis(AnalysisDeclContext &AC,
                                               LifetimeSafetyReporter *Reporter)
    : AC(AC), Reporter(Reporter) {}

void LifetimeSafetyAnalysis::run() {
  llvm::TimeTraceScope TimeProfile("LifetimeSafetyAnalysis");

  const CFG &Cfg = *AC.getCFG();
  DEBUG_WITH_TYPE("PrintCFG", Cfg.dump(AC.getASTContext().getLangOpts(),
                                       /*ShowColors=*/true));

  FactsGenerator FactGen(FactMgr, AC);
  FactGen.run();
  DEBUG_WITH_TYPE("LifetimeFacts", FactMgr.dump(Cfg, AC));

  /// TODO(opt): Consider optimizing individual blocks before running the
  /// dataflow analysis.
  /// 1. Expression Origins: These are assigned once and read at most once,
  ///    forming simple chains. These chains can be compressed into a single
  ///    assignment.
  /// 2. Block-Local Loans: Origins of expressions are never read by other
  ///    blocks; only Decls are visible.  Therefore, loans in a block that
  ///    never reach an Origin associated with a Decl can be safely dropped by
  ///    the analysis.
  /// 3. Collapse ExpireFacts belonging to same source location into a single
  ///    Fact.
  LoanPropagation = std::make_unique<LoanPropagationAnalysis>(
      Cfg, AC, FactMgr, Factory.OriginMapFactory, Factory.LoanSetFactory);
  LoanPropagation->run();

  LiveOrigins = std::make_unique<LiveOriginAnalysis>(
      Cfg, AC, FactMgr, Factory.LivenessMapFactory);
  LiveOrigins->run();
  DEBUG_WITH_TYPE("LiveOrigins",
                  LiveOrigins->dump(llvm::dbgs(), getTestPoints()));

  runLifetimeChecker(*LoanPropagation, *LiveOrigins, FactMgr, AC, Reporter);
}

LoanSet LifetimeSafetyAnalysis::getLoansAtPoint(OriginID OID,
                                                ProgramPoint PP) const {
  assert(LoanPropagation && "Analysis has not been run.");
  return LoanPropagation->getLoans(OID, PP);
}

std::optional<OriginID>
LifetimeSafetyAnalysis::getOriginIDForDecl(const ValueDecl *D) const {
  // This assumes the OriginManager's `get` can find an existing origin.
  // We might need a `find` method on OriginManager to avoid `getOrCreate` logic
  // in a const-query context if that becomes an issue.
  return const_cast<OriginManager &>(FactMgr.getOriginMgr()).get(*D);
}

std::vector<LoanID>
LifetimeSafetyAnalysis::getLoanIDForVar(const VarDecl *VD) const {
  std::vector<LoanID> Result;
  for (const Loan &L : FactMgr.getLoanMgr().getLoans())
    if (L.Path.D == VD)
      Result.push_back(L.ID);
  return Result;
}

std::vector<std::pair<OriginID, LivenessKind>>
LifetimeSafetyAnalysis::getLiveOriginsAtPoint(ProgramPoint PP) const {
  assert(LiveOrigins && "LiveOriginAnalysis has not been run.");
  std::vector<std::pair<OriginID, LivenessKind>> Result;
  for (auto &[OID, Info] : LiveOrigins->getLiveOrigins(PP))
    Result.push_back({OID, Info.Kind});
  return Result;
}

llvm::StringMap<ProgramPoint> LifetimeSafetyAnalysis::getTestPoints() const {
  llvm::StringMap<ProgramPoint> AnnotationToPointMap;
  for (const CFGBlock *Block : *AC.getCFG()) {
    for (const Fact *F : FactMgr.getFacts(Block)) {
      if (const auto *TPF = F->getAs<TestPointFact>()) {
        StringRef PointName = TPF->getAnnotation();
        assert(AnnotationToPointMap.find(PointName) ==
                   AnnotationToPointMap.end() &&
               "more than one test points with the same name");
        AnnotationToPointMap[PointName] = F;
      }
    }
  }
  return AnnotationToPointMap;
}
} // namespace internal

void runLifetimeSafetyAnalysis(AnalysisDeclContext &AC,
                               LifetimeSafetyReporter *Reporter) {
  internal::LifetimeSafetyAnalysis Analysis(AC, Reporter);
  Analysis.run();
}
} // namespace clang::lifetimes
