//===- LifetimeSafety.h - C++ Lifetime Safety Analysis -*----------- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the main entry point and orchestrator for the C++ Lifetime
// Safety Analysis. It coordinates the entire analysis pipeline: fact
// generation, loan propagation, live origins analysis, and enforcement of
// lifetime safety policy.
//
// The analysis is based on the concepts of "origins" and "loans" to track
// pointer lifetimes and detect issues like use-after-free and dangling
// pointers. See the RFC for more details:
// https://discourse.llvm.org/t/rfc-intra-procedural-lifetime-analysis-in-clang/86291
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_CLANG_ANALYSIS_ANALYSES_LIFETIMESAFETY_H
#define LLVM_CLANG_ANALYSIS_ANALYSES_LIFETIMESAFETY_H

#include "clang/AST/Decl.h"
#include "clang/Analysis/Analyses/LifetimeSafety/Facts.h"
#include "clang/Analysis/Analyses/LifetimeSafety/LifetimeStats.h"
#include "clang/Analysis/Analyses/LifetimeSafety/LiveOrigins.h"
#include "clang/Analysis/Analyses/LifetimeSafety/LoanPropagation.h"
#include "clang/Analysis/Analyses/LifetimeSafety/MovedLoans.h"
#include "clang/Analysis/Analyses/LifetimeSafety/Origins.h"
#include "clang/Analysis/AnalysisDeclContext.h"
#include <cstddef>
#include <memory>

namespace clang::lifetimes {

struct LifetimeSafetyOpts {
  /// Maximum number of CFG blocks to analyze. Functions with larger CFGs will
  /// be skipped.
  size_t MaxCFGBlocks;
};

/// Enum to track functions visible across or within TU.
enum class SuggestionScope {
  CrossTU, // For suggestions on declarations visible across Translation Units.
  IntraTU  // For suggestions on definitions local to a Translation Unit.
};

/// Abstract interface for operations requiring Sema access.
///
/// This class exists to break a circular dependency: the LifetimeSafety
/// analysis target cannot directly depend on clangSema (which would create the
/// cycle: clangSema -> clangAnalysis -> clangAnalysisLifetimeSafety ->
/// clangSema).
///
/// Instead, this interface is implemented in AnalysisBasedWarnings.cpp (part of
/// clangSema), allowing the analysis to report diagnostics and modify the AST
/// through Sema without introducing a circular dependency.
class LifetimeSafetySemaHelper {
public:
  LifetimeSafetySemaHelper() = default;
  virtual ~LifetimeSafetySemaHelper() = default;

  virtual void reportUseAfterFree(const Expr *IssueExpr, const Expr *UseExpr,
                                  const Expr *MovedExpr,
                                  SourceLocation FreeLoc) {}

  virtual void reportUseAfterReturn(const Expr *IssueExpr,
                                    const Expr *ReturnExpr,
                                    const Expr *MovedExpr,
                                    SourceLocation ExpiryLoc) {}

  virtual void reportDanglingField(const Expr *IssueExpr,
                                   const FieldDecl *Field,
                                   const Expr *MovedExpr,
                                   SourceLocation ExpiryLoc) {}

  // Reports when a reference/iterator is used after the container operation
  // that invalidated it.
  virtual void reportUseAfterInvalidation(const Expr *IssueExpr,
                                          const Expr *UseExpr,
                                          const Expr *InvalidationExpr) {}
  virtual void reportUseAfterInvalidation(const ParmVarDecl *PVD,
                                          const Expr *UseExpr,
                                          const Expr *InvalidationExpr) {}

  // Suggests lifetime bound annotations for function paramters.
  virtual void suggestLifetimeboundToParmVar(SuggestionScope Scope,
                                             const ParmVarDecl *ParmToAnnotate,
                                             const Expr *EscapeExpr) {}

  // Reports misuse of [[clang::noescape]] when parameter escapes through return
  virtual void reportNoescapeViolation(const ParmVarDecl *ParmWithNoescape,
                                       const Expr *EscapeExpr) {}
  // Reports misuse of [[clang::noescape]] when parameter escapes through field
  virtual void reportNoescapeViolation(const ParmVarDecl *ParmWithNoescape,
                                       const FieldDecl *EscapeField) {}

  // Suggests lifetime bound annotations for implicit this.
  virtual void suggestLifetimeboundToImplicitThis(SuggestionScope Scope,
                                                  const CXXMethodDecl *MD,
                                                  const Expr *EscapeExpr) {}

  // Adds inferred lifetime bound attribute for implicit this to its
  // TypeSourceInfo.
  virtual void addLifetimeBoundToImplicitThis(const CXXMethodDecl *MD) {}
};

/// The main entry point for the analysis.
void runLifetimeSafetyAnalysis(AnalysisDeclContext &AC,
                               LifetimeSafetySemaHelper *SemaHelper,
                               LifetimeSafetyStats &Stats, bool CollectStats);

namespace internal {

void collectLifetimeStats(AnalysisDeclContext &AC, OriginManager &OM,
                          LifetimeSafetyStats &Stats);

/// An object to hold the factories for immutable collections, ensuring
/// that all created states share the same underlying memory management.
struct LifetimeFactory {
  OriginLoanMap::Factory OriginMapFactory{/*canonicalize=*/false};
  LoanSet::Factory LoanSetFactory{/*canonicalize=*/false};
  MovedLoansMap::Factory MovedLoansMapFactory{/*canonicalize=*/false};
  LivenessMap::Factory LivenessMapFactory{/*canonicalize=*/false};
};

/// Running the lifetime safety analysis and querying its results. It
/// encapsulates the various dataflow analyses.
class LifetimeSafetyAnalysis {
public:
  LifetimeSafetyAnalysis(AnalysisDeclContext &AC,
                         LifetimeSafetySemaHelper *SemaHelper,
                         const LifetimeSafetyOpts &LSOpts);

  void run();

  /// \note These are provided only for testing purposes.
  LoanPropagationAnalysis &getLoanPropagation() const {
    return *LoanPropagation;
  }
  LiveOriginsAnalysis &getLiveOrigins() const { return *LiveOrigins; }
  FactManager &getFactManager() { return *FactMgr; }

private:
  AnalysisDeclContext &AC;
  LifetimeSafetySemaHelper *SemaHelper;
  const LifetimeSafetyOpts LSOpts;
  LifetimeFactory Factory;
  std::unique_ptr<FactManager> FactMgr;
  std::unique_ptr<LiveOriginsAnalysis> LiveOrigins;
  std::unique_ptr<LoanPropagationAnalysis> LoanPropagation;
  std::unique_ptr<MovedLoansAnalysis> MovedLoans;
};
} // namespace internal
} // namespace clang::lifetimes

#endif // LLVM_CLANG_ANALYSIS_ANALYSES_LIFETIMESAFETY_H
