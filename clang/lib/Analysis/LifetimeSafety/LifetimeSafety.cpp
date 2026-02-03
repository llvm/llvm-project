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
#include "clang/Analysis/Analyses/LifetimeSafety/Facts.h"
#include "clang/Analysis/Analyses/LifetimeSafety/FactsGenerator.h"
#include "clang/Analysis/Analyses/LifetimeSafety/LifetimeStats.h"
#include "clang/Analysis/Analyses/LifetimeSafety/LiveOrigins.h"
#include "clang/Analysis/Analyses/LifetimeSafety/LoanPropagation.h"
#include "clang/Analysis/Analyses/LifetimeSafety/Origins.h"
#include "clang/Analysis/AnalysisDeclContext.h"
#include "clang/Analysis/CFG.h"
#include "llvm/ADT/FoldingSet.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/TimeProfiler.h"
#include <memory>

namespace clang::lifetimes {
namespace internal {

#ifndef NDEBUG
static void DebugOnlyFunction(AnalysisDeclContext &AC, const CFG &Cfg,
                              FactManager &FactMgr) {
  std::string Name;
  if (const Decl *D = AC.getDecl()) {
    if (const auto *ND = dyn_cast<NamedDecl>(D))
      Name = ND->getQualifiedNameAsString();
  };
  DEBUG_WITH_TYPE(Name.c_str(), AC.getDecl()->dumpColor());
  DEBUG_WITH_TYPE(Name.c_str(), Cfg.dump(AC.getASTContext().getLangOpts(),
                                         /*ShowColors=*/true));
  DEBUG_WITH_TYPE(Name.c_str(), FactMgr.dump(Cfg, AC));
}
#endif

LifetimeSafetyAnalysis::LifetimeSafetyAnalysis(
    AnalysisDeclContext &AC, LifetimeSafetySemaHelper *SemaHelper)
    : AC(AC), SemaHelper(SemaHelper) {}

void LifetimeSafetyAnalysis::run() {
  llvm::TimeTraceScope TimeProfile("LifetimeSafetyAnalysis");

  const CFG &Cfg = *AC.getCFG();
  DEBUG_WITH_TYPE("PrintCFG", Cfg.dump(AC.getASTContext().getLangOpts(),
                                       /*ShowColors=*/true));

  FactMgr = std::make_unique<FactManager>(AC, Cfg);

  FactsGenerator FactGen(*FactMgr, AC);
  FactGen.run();

  DEBUG_WITH_TYPE("LifetimeFacts", FactMgr->dump(Cfg, AC));

  // Debug print facts for a specific function using
  // -debug-only=EnableFilterByFunctionName,YourFunctionNameFoo
  DEBUG_WITH_TYPE("EnableFilterByFunctionName",
                  DebugOnlyFunction(AC, Cfg, *FactMgr));

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
      Cfg, AC, *FactMgr, Factory.OriginMapFactory, Factory.LoanSetFactory);

  LiveOrigins = std::make_unique<LiveOriginsAnalysis>(
      Cfg, AC, *FactMgr, Factory.LivenessMapFactory);
  DEBUG_WITH_TYPE("LiveOrigins",
                  LiveOrigins->dump(llvm::dbgs(), FactMgr->getTestPoints()));

  runLifetimeChecker(*LoanPropagation, *LiveOrigins, *FactMgr, AC, SemaHelper);
}

void collectLifetimeStats(AnalysisDeclContext &AC, OriginManager &OM,
                          LifetimeSafetyStats &Stats) {
  Stmt *FunctionBody = AC.getBody();
  if (FunctionBody == nullptr)
    return;
  OM.collectMissingOrigins(*FunctionBody, Stats);
}
} // namespace internal

void runLifetimeSafetyAnalysis(AnalysisDeclContext &AC,
                               LifetimeSafetySemaHelper *SemaHelper,
                               LifetimeSafetyStats &Stats, bool CollectStats) {
  internal::LifetimeSafetyAnalysis Analysis(AC, SemaHelper);
  Analysis.run();
  if (CollectStats)
    collectLifetimeStats(AC, Analysis.getFactManager().getOriginMgr(), Stats);
}
} // namespace clang::lifetimes
