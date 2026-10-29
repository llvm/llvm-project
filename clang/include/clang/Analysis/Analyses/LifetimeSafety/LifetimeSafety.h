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
#include "llvm/ADT/PointerUnion.h"
#include <cstddef>
#include <memory>

namespace clang::lifetimes {

struct LifetimeSafetyOpts {
  /// Maximum number of CFG blocks to analyze. Functions with larger CFGs will
  /// be skipped.
  size_t MaxCFGBlocks = 0;

  /// Whether to suggest lifetime annotations.
  bool SuggestAnnotations = false;

  bool CheckNoescapeViolations = false;
  bool CheckLifetimeboundViolations = false;
  bool CheckMisplacedLifetimebound = false;
  bool CheckInapplicableLifetimebound = false;
  bool CheckFieldOriginEscape = false;
  /// Whether to verify methods of classes annotated with [[clang::places]] or
  /// [[clang::invalidates]] against their place contracts.
  bool CheckPlaceContracts = false;
};

/// Enum to track functions visible across or within TU.
enum class WarningScope {
  CrossTU, // For warnings on declarations visible across Translation Units.
  IntraTU  // For warnings on functions local to a Translation Unit.
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

  virtual void reportUseAfterScope(const Expr *IssueExpr, const Expr *UseExpr,
                                   const Expr *MovedExpr,
                                   SourceLocation FreeLoc,
                                   llvm::ArrayRef<const Expr *> ExprChain) {}

  // Overload for a use with only a location and no expression (e.g. a borrow
  // captured into the object and still held at the capturing method's exit).
  virtual void reportUseAfterScope(const Expr *IssueExpr, SourceLocation UseLoc,
                                   const Expr *MovedExpr,
                                   SourceLocation FreeLoc,
                                   llvm::ArrayRef<const Expr *> ExprChain) {}

  // TODO: Report where the object was destroyed when that happens before the
  // return (inner scopes, temporaries).
  virtual void reportUseAfterReturn(const Expr *IssueExpr,
                                    const Expr *ReturnExpr,
                                    const Expr *MovedExpr,
                                    llvm::ArrayRef<const Expr *> ExprChain) {}

  virtual void reportDanglingField(const Expr *IssueExpr,
                                   const FieldDecl *Field,
                                   const Expr *MovedExpr,
                                   bool IsCapturedByLambda,
                                   SourceLocation ExpiryLoc) {}

  virtual void reportDanglingGlobal(const Expr *IssueExpr,
                                    const VarDecl *DanglingGlobal,
                                    const Expr *MovedExpr,
                                    SourceLocation ExpiryLoc,
                                    bool IsMain = false) {}

  // Reports when a reference/iterator is used after the container operation
  // that invalidated it.
  virtual void
  reportUseAfterInvalidation(const Expr *IssueExpr, const Expr *UseExpr,
                             const Expr *InvalidationExpr,
                             llvm::ArrayRef<const Expr *> ExprChain) {}
  virtual void
  reportUseAfterInvalidation(const ParmVarDecl *PVD, const Expr *UseExpr,
                             const Expr *InvalidationExpr,
                             llvm::ArrayRef<const Expr *> ExprChain) {}
  virtual void
  reportUseAfterInvalidation(const Expr *IssueExpr, SourceLocation UseLoc,
                             const Expr *InvalidationExpr,
                             llvm::ArrayRef<const Expr *> ExprChain) {}
  virtual void
  reportUseAfterInvalidation(const ParmVarDecl *PVD, SourceLocation UseLoc,
                             const Expr *InvalidationExpr,
                             llvm::ArrayRef<const Expr *> ExprChain) {}
  // The invalidated storage is reached through the implicit object of the
  // method `MD` being analyzed (`this`); `Field` is the sub-object of `*this`
  // it belongs to, if any (e.g. `auto it = c.begin(); c.push_back(1); *it;`
  // in a method of a class with a field `c`).
  virtual void
  reportUseAfterInvalidation(const CXXMethodDecl *MD, const FieldDecl *Field,
                             const Expr *UseExpr, const Expr *InvalidationExpr,
                             llvm::ArrayRef<const Expr *> ExprChain) {}
  virtual void reportInvalidatedField(const Expr *IssueExpr,
                                      const FieldDecl *Field,
                                      const Expr *InvalidationExpr) {}
  virtual void reportInvalidatedField(const ParmVarDecl *PVD,
                                      const FieldDecl *Field,
                                      const Expr *InvalidationExpr) {}
  virtual void reportInvalidatedGlobal(const Expr *IssueExpr,
                                       const VarDecl *Global,
                                       const Expr *InvalidationExpr) {}
  virtual void reportInvalidatedGlobal(const ParmVarDecl *PVD,
                                       const VarDecl *Global,
                                       const Expr *InvalidationExpr) {}

  using EscapingTarget =
      llvm::PointerUnion<const Expr *, const FieldDecl *, const VarDecl *>;

  // Suggests lifetime bound annotations for function parameters.
  virtual void suggestLifetimeboundToParmVar(WarningScope Scope,
                                             const ParmVarDecl *ParmToAnnotate,
                                             EscapingTarget Target) {}

  // Reports misuse of [[clang::noescape]] when parameter escapes through return
  virtual void reportNoescapeViolation(const ParmVarDecl *ParmWithNoescape,
                                       const Expr *EscapeExpr) {}
  // Reports misuse of [[clang::noescape]] when parameter escapes through field
  virtual void reportNoescapeViolation(const ParmVarDecl *ParmWithNoescape,
                                       const FieldDecl *EscapeField) {}
  // Reports misuse of [[clang::noescape]] when parameter escapes through
  // assignment to a global variable
  virtual void reportNoescapeViolation(const ParmVarDecl *ParmWithNoescape,
                                       const VarDecl *EscapeGlobal) {}

  // Reports misuse of [[clang::lifetimebound]] when parameter doesn't escape
  // through return.
  virtual void
  reportLifetimeboundViolation(const ParmVarDecl *ParmWithLifetimebound) {}

  // Reports misuse of [[clang::lifetimebound]] when implicit this parameter
  // doesn't escape through return.
  virtual void
  reportLifetimeboundViolation(const CXXMethodDecl *MDWithLifetimebound) {}

  // Reports a member function definition that has [[clang::lifetimebound]] on
  // the implicit this parameter when the canonical declaration does not.
  virtual void reportMisplacedLifetimebound(WarningScope Scope,
                                            const CXXMethodDecl *FDef,
                                            const CXXMethodDecl *FDecl) {}

  // Reports a function definition parameter that has [[clang::lifetimebound]]
  // when the corresponding parameter in the canonical declaration.
  virtual void reportMisplacedLifetimebound(WarningScope Scope,
                                            const ParmVarDecl *PVDDef,
                                            const ParmVarDecl *PVDDecl) {}

  virtual void reportInapplicableLifetimebound(const ParmVarDecl *PVD) {}

  // Suggests lifetime bound annotations for implicit this.
  virtual void suggestLifetimeboundToImplicitThis(WarningScope Scope,
                                                  const CXXMethodDecl *MD,
                                                  const Expr *EscapeExpr) {}

  // Reports when a member function returns an origin of a field.
  virtual void reportFieldOriginEscape(const CXXMethodDecl *MD,
                                       const FieldDecl *FD,
                                       const Expr *EscapeExpr) {}

  // Place contracts (see [[clang::places]]). What was invalidated or borrowed
  // is identified by the first projection below `*this`: a named `Place`, a
  // `Field` of `*this`, or neither (the anonymous interior / the object).

  // Reports that method `MD` invalidates storage of `*this` that its
  // [[clang::invalidates]] annotation (if any) does not declare.
  virtual void reportUndeclaredInvalidation(const CXXMethodDecl *MD,
                                            const Expr *InvalidationExpr,
                                            const IdentifierInfo *Place,
                                            const FieldDecl *Field) {}

  // Reports that method `MD`, annotated `[[clang::lifetimebound(...)]]` on its
  // implicit object parameter, returns a borrow of a different place of
  // `*this`. `IsObject` means the object itself (`*this`) is returned.
  virtual void reportPlaceBorrowViolation(const CXXMethodDecl *MD,
                                          const Expr *ReturnExpr,
                                          const IdentifierInfo *Place,
                                          const FieldDecl *Field,
                                          bool IsObject) {}

  // Adds inferred lifetime bound attribute for implicit this to its
  // TypeSourceInfo.
  virtual void addLifetimeBoundToImplicitThis(const CXXMethodDecl *MD) {}
};

/// The main entry point for the analysis.
void runLifetimeSafetyAnalysis(AnalysisDeclContext &AC,
                               LifetimeSafetySemaHelper *SemaHelper,
                               const LifetimeSafetyOpts &Opts,
                               LifetimeSafetyStats &Stats, bool CollectStats);

namespace internal {

void collectLifetimeStats(AnalysisDeclContext &AC, OriginManager &OM,
                          LifetimeSafetyStats &Stats);

/// An object to hold the factories for immutable collections, ensuring
/// that all created states share the same underlying memory management.
struct LifetimeFactory {
  OriginLoanMap::Factory OriginMapFactory;
  LoanSet::Factory LoanSetFactory;
  MovedLoansMap::Factory MovedLoansMapFactory;
  LivenessMap::Factory LivenessMapFactory;
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
