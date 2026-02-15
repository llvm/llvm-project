//===- Checker.cpp - C++ Lifetime Safety Checker ----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the LifetimeChecker, which detects use-after-free
// errors by checking if live origins hold loans that have expired.
//
//===----------------------------------------------------------------------===//

#include "clang/Analysis/Analyses/LifetimeSafety/Checker.h"
#include "clang/AST/Decl.h"
#include "clang/AST/Expr.h"
#include "clang/Analysis/Analyses/LifetimeSafety/Facts.h"
#include "clang/Analysis/Analyses/LifetimeSafety/LifetimeAnnotations.h"
#include "clang/Analysis/Analyses/LifetimeSafety/LiveOrigins.h"
#include "clang/Analysis/Analyses/LifetimeSafety/LoanPropagation.h"
#include "clang/Analysis/Analyses/LifetimeSafety/Loans.h"
#include "clang/Analysis/Analyses/PostOrderCFGView.h"
#include "clang/Analysis/AnalysisDeclContext.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Basic/SourceManager.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/TimeProfiler.h"

namespace clang::lifetimes::internal {

static Confidence livenessKindToConfidence(LivenessKind K) {
  switch (K) {
  case LivenessKind::Must:
    return Confidence::Definite;
  case LivenessKind::Maybe:
    return Confidence::Maybe;
  case LivenessKind::Dead:
    return Confidence::None;
  }
  llvm_unreachable("unknown liveness kind");
}

namespace {

/// Struct to store the complete context for a potential lifetime violation.
struct PendingWarning {
  SourceLocation ExpiryLoc; // Where the loan expired.
  llvm::PointerUnion<const UseFact *, const OriginEscapesFact *> CausingFact;
  const Expr *MovedExpr;
  const Expr *InvalidatedByExpr;
  Confidence ConfidenceLevel;
};

using AnnotationTarget =
    llvm::PointerUnion<const ParmVarDecl *, const CXXMethodDecl *>;
using EscapingTarget = llvm::PointerUnion<const Expr *, const FieldDecl *>;

class LifetimeChecker {
private:
  llvm::DenseMap<LoanID, PendingWarning> FinalWarningsMap;
  llvm::DenseMap<AnnotationTarget, const Expr *> AnnotationWarningsMap;
  llvm::DenseMap<const ParmVarDecl *, EscapingTarget> NoescapeWarningsMap;
  const LoanPropagationAnalysis &LoanPropagation;
  const MovedLoansAnalysis &MovedLoans;
  const LiveOriginsAnalysis &LiveOrigins;
  FactManager &FactMgr;
  LifetimeSafetySemaHelper *SemaHelper;
  ASTContext &AST;

public:
  LifetimeChecker(const LoanPropagationAnalysis &LoanPropagation,
                  const MovedLoansAnalysis &MovedLoans,
                  const LiveOriginsAnalysis &LiveOrigins, FactManager &FM,
                  AnalysisDeclContext &ADC,
                  LifetimeSafetySemaHelper *SemaHelper)
      : LoanPropagation(LoanPropagation), MovedLoans(MovedLoans),
        LiveOrigins(LiveOrigins), FactMgr(FM), SemaHelper(SemaHelper),
        AST(ADC.getASTContext()) {
    for (const CFGBlock *B : *ADC.getAnalysis<PostOrderCFGView>())
      for (const Fact *F : FactMgr.getFacts(B))
        if (const auto *EF = F->getAs<ExpireFact>())
          checkExpiry(EF);
        else if (const auto *IOF = F->getAs<InvalidateOriginFact>())
          checkInvalidation(IOF);
        else if (const auto *OEF = F->getAs<OriginEscapesFact>())
          checkAnnotations(OEF);
    issuePendingWarnings();
    suggestAnnotations();
    reportNoescapeViolations();
    //  Annotation inference is currently guarded by a frontend flag. In the
    //  future, this might be replaced by a design that differentiates between
    //  explicit and inferred findings with separate warning groups.
    if (AST.getLangOpts().EnableLifetimeSafetyInference)
      inferAnnotations();
  }

  /// Checks if an escaping origin holds a placeholder loan, indicating a
  /// missing [[clang::lifetimebound]] annotation or a violation of
  /// [[clang::noescape]].
  void checkAnnotations(const OriginEscapesFact *OEF) {
    OriginID EscapedOID = OEF->getEscapedOriginID();
    LoanSet EscapedLoans = LoanPropagation.getLoans(EscapedOID, OEF);
    auto CheckParam = [&](const ParmVarDecl *PVD) {
      // NoEscape param should not escape.
      if (PVD->hasAttr<NoEscapeAttr>()) {
        if (auto *ReturnEsc = dyn_cast<ReturnEscapeFact>(OEF))
          NoescapeWarningsMap.try_emplace(PVD, ReturnEsc->getReturnExpr());
        if (auto *FieldEsc = dyn_cast<FieldEscapeFact>(OEF))
          NoescapeWarningsMap.try_emplace(PVD, FieldEsc->getFieldDecl());
        return;
      }
      // Suggest lifetimebound for parameter escaping through return.
      if (!PVD->hasAttr<LifetimeBoundAttr>())
        if (auto *ReturnEsc = dyn_cast<ReturnEscapeFact>(OEF))
          AnnotationWarningsMap.try_emplace(PVD, ReturnEsc->getReturnExpr());
      // TODO: Suggest lifetime_capture_by(this) for parameter escaping to a
      // field!
    };
    auto CheckImplicitThis = [&](const CXXMethodDecl *MD) {
      if (!implicitObjectParamIsLifetimeBound(MD))
        if (auto *ReturnEsc = dyn_cast<ReturnEscapeFact>(OEF))
          AnnotationWarningsMap.try_emplace(MD, ReturnEsc->getReturnExpr());
    };
    for (LoanID LID : EscapedLoans) {
      const Loan *L = FactMgr.getLoanMgr().getLoan(LID);
      const PlaceholderBase *PB = L->getAccessPath().getAsPlaceholderBase();
      if (!PB)
        continue;
      if (const auto *PVD = PB->getParmVarDecl())
        CheckParam(PVD);
      else if (const auto *MD = PB->getMethodDecl())
        CheckImplicitThis(MD);
    }
  }

  /// Checks for use-after-free & use-after-return errors when an access path
  /// expires (e.g., a variable goes out of scope).
  ///
  /// When a path expires, all loans prefixed by that path expire. For example,
  /// if `x` expires, loans to `x`, `x.field`, and `x.field.*` all expire.
  /// This method examines all live origins and reports warnings for loans they
  /// hold that are prefixed by the expired path.
  void checkExpiry(const ExpireFact *EF) {
    const AccessPath &ExpiredPath = EF->getAccessPath();

    LivenessMap Origins = LiveOrigins.getLiveOriginsAt(EF);

    for (auto &[OID, LiveInfo] : Origins) {
      LoanSet HeldLoans = LoanPropagation.getLoans(OID, EF);
      for (LoanID HeldLoanID : HeldLoans) {
        const Loan *HeldLoan = FactMgr.getLoanMgr().getLoan(HeldLoanID);
        if (ExpiredPath.isPrefixOf(HeldLoan->getAccessPath())) {
          // HeldLoan is expired because its base or itself is expired.
          const Expr *MovedExpr = nullptr;
          if (auto *ME = MovedLoans.getMovedLoans(EF).lookup(HeldLoanID))
            MovedExpr = *ME;

          Confidence NewConfidence = livenessKindToConfidence(LiveInfo.Kind);
          Confidence LastConf =
              FinalWarningsMap.lookup(HeldLoanID).ConfidenceLevel;
          if (LastConf >= NewConfidence)
            continue;

          FinalWarningsMap[HeldLoanID] = {EF->getExpiryLoc(),
                                          LiveInfo.CausingFact, MovedExpr,
                                          nullptr, NewConfidence};
        }
      }
    }
  }

  /// Checks for use-after-invalidation errors when a container is modified.
  ///
  /// When a container is invalidated, loans pointing into its interior are
  /// invalidated. For example, if container `v` is invalidated, iterators with
  /// loans to `v.*` are invalidated. This method finds live origins holding
  /// such loans and reports warnings. A loan is invalidated if its path extends
  /// an invalidated container's path (e.g., `v.*` extends `v`).
  void checkInvalidation(const InvalidateOriginFact *IOF) {
    OriginID InvalidatedOrigin = IOF->getInvalidatedOrigin();
    /// Get loans directly pointing to the invalidated container
    LoanSet DirectlyInvalidatedLoans =
        LoanPropagation.getLoans(InvalidatedOrigin, IOF);
    auto IsInvalidated = [&](const Loan *L) {
      for (LoanID InvalidID : DirectlyInvalidatedLoans) {
        const Loan *InvalidL = FactMgr.getLoanMgr().getLoan(InvalidID);
        if (InvalidL->getAccessPath().isStrictPrefixOf(L->getAccessPath()))
          return true;
      }
      return false;
    };
    // For each live origin, check if it holds an invalidated loan and report.
    LivenessMap Origins = LiveOrigins.getLiveOriginsAt(IOF);
    for (auto &[OID, LiveInfo] : Origins) {
      LoanSet HeldLoans = LoanPropagation.getLoans(OID, IOF);
      for (LoanID LiveLoanID : HeldLoans)
        if (IsInvalidated(FactMgr.getLoanMgr().getLoan(LiveLoanID))) {
          Confidence CurConfidence = livenessKindToConfidence(LiveInfo.Kind);
          Confidence LastConf =
              FinalWarningsMap.lookup(LiveLoanID).ConfidenceLevel;
          if (LastConf < CurConfidence) {
            FinalWarningsMap[LiveLoanID] = {
                /*ExpiryLoc=*/{},
                /*CausingFact=*/LiveInfo.CausingFact,
                /*MovedExpr=*/nullptr,
                /*InvalidatedByExpr=*/IOF->getInvalidationExpr(),
                /*ConfidenceLevel=*/CurConfidence};
          }
        }
    }
  }

  void issuePendingWarnings() {
    if (!SemaHelper)
      return;
    for (const auto &[LID, Warning] : FinalWarningsMap) {
      const Loan *L = FactMgr.getLoanMgr().getLoan(LID);

      const Expr *IssueExpr = L->getIssueExpr();
      const ParmVarDecl *InvalidatedPVD = nullptr;
      if (const PlaceholderBase *PB = L->getAccessPath().getAsPlaceholderBase())
        InvalidatedPVD = PB->getParmVarDecl();
      llvm::PointerUnion<const UseFact *, const OriginEscapesFact *>
          CausingFact = Warning.CausingFact;
      Confidence Confidence = Warning.ConfidenceLevel;
      const Expr *MovedExpr = Warning.MovedExpr;
      SourceLocation ExpiryLoc = Warning.ExpiryLoc;

      if (const auto *UF = CausingFact.dyn_cast<const UseFact *>()) {
        if (Warning.InvalidatedByExpr) {
          // Use-after-invalidation of an object on stack.
          if (IssueExpr)
            SemaHelper->reportUseAfterInvalidation(IssueExpr, UF->getUseExpr(),
                                                   Warning.InvalidatedByExpr);
          // Use-after-invalidation of a parameter.
          if (InvalidatedPVD) {
            SemaHelper->reportUseAfterInvalidation(
                InvalidatedPVD, UF->getUseExpr(), Warning.InvalidatedByExpr);
          }
        } else {
          // Scope-based expiry (use-after-scope).
          SemaHelper->reportUseAfterFree(IssueExpr, UF->getUseExpr(), MovedExpr,
                                         ExpiryLoc, Confidence);
        }
      } else if (const auto *OEF =
                     CausingFact.dyn_cast<const OriginEscapesFact *>()) {
        // Return stack address.
        if (const auto *RetEscape = dyn_cast<ReturnEscapeFact>(OEF))
          SemaHelper->reportUseAfterReturn(IssueExpr,
                                           RetEscape->getReturnExpr(),
                                           MovedExpr, ExpiryLoc, Confidence);
        // Dangling field.
        else if (const auto *FieldEscape = dyn_cast<FieldEscapeFact>(OEF))
          SemaHelper->reportDanglingField(
              IssueExpr, FieldEscape->getFieldDecl(), MovedExpr, ExpiryLoc);
        else
          llvm_unreachable("Unhandled OriginEscapesFact type");
      } else
        llvm_unreachable("Unhandled CausingFact type");
    }
  }

  /// Returns the declaration of a function that is visible across translation
  /// units, if such a declaration exists and is different from the definition.
  static const FunctionDecl *getCrossTUDecl(const FunctionDecl &FD,
                                            SourceManager &SM) {
    if (!FD.isExternallyVisible())
      return nullptr;
    const FileID DefinitionFile = SM.getFileID(FD.getLocation());
    for (const FunctionDecl *Redecl : FD.redecls())
      if (SM.getFileID(Redecl->getLocation()) != DefinitionFile)
        return Redecl;

    return nullptr;
  }

  static const FunctionDecl *getCrossTUDecl(const ParmVarDecl &PVD,
                                            SourceManager &SM) {
    if (const auto *FD = dyn_cast<FunctionDecl>(PVD.getDeclContext()))
      return getCrossTUDecl(*FD, SM);
    return nullptr;
  }

  static void suggestWithScopeForParmVar(LifetimeSafetySemaHelper *SemaHelper,
                                         const ParmVarDecl *PVD,
                                         SourceManager &SM,
                                         const Expr *EscapeExpr) {
    if (const FunctionDecl *CrossTUDecl = getCrossTUDecl(*PVD, SM))
      SemaHelper->suggestLifetimeboundToParmVar(
          SuggestionScope::CrossTU,
          CrossTUDecl->getParamDecl(PVD->getFunctionScopeIndex()), EscapeExpr);
    else
      SemaHelper->suggestLifetimeboundToParmVar(SuggestionScope::IntraTU, PVD,
                                                EscapeExpr);
  }

  static void
  suggestWithScopeForImplicitThis(LifetimeSafetySemaHelper *SemaHelper,
                                  const CXXMethodDecl *MD, SourceManager &SM,
                                  const Expr *EscapeExpr) {
    if (const FunctionDecl *CrossTUDecl = getCrossTUDecl(*MD, SM))
      SemaHelper->suggestLifetimeboundToImplicitThis(
          SuggestionScope::CrossTU, cast<CXXMethodDecl>(CrossTUDecl),
          EscapeExpr);
    else
      SemaHelper->suggestLifetimeboundToImplicitThis(SuggestionScope::IntraTU,
                                                     MD, EscapeExpr);
  }

  void suggestAnnotations() {
    if (!SemaHelper)
      return;
    SourceManager &SM = AST.getSourceManager();
    for (auto [Target, EscapeExpr] : AnnotationWarningsMap) {
      if (const auto *PVD = Target.dyn_cast<const ParmVarDecl *>())
        suggestWithScopeForParmVar(SemaHelper, PVD, SM, EscapeExpr);
      else if (const auto *MD = Target.dyn_cast<const CXXMethodDecl *>())
        suggestWithScopeForImplicitThis(SemaHelper, MD, SM, EscapeExpr);
    }
  }

  void reportNoescapeViolations() {
    for (auto [PVD, EscapeTarget] : NoescapeWarningsMap) {
      if (const auto *E = EscapeTarget.dyn_cast<const Expr *>())
        SemaHelper->reportNoescapeViolation(PVD, E);
      else if (const auto *FD = EscapeTarget.dyn_cast<const FieldDecl *>())
        SemaHelper->reportNoescapeViolation(PVD, FD);
      else
        llvm_unreachable("Unhandled EscapingTarget type");
    }
  }

  void inferAnnotations() {
    for (auto [Target, EscapeExpr] : AnnotationWarningsMap) {
      if (const auto *MD = Target.dyn_cast<const CXXMethodDecl *>()) {
        if (!implicitObjectParamIsLifetimeBound(MD))
          SemaHelper->addLifetimeBoundToImplicitThis(cast<CXXMethodDecl>(MD));
      } else if (const auto *PVD = Target.dyn_cast<const ParmVarDecl *>()) {
        const auto *FD = dyn_cast<FunctionDecl>(PVD->getDeclContext());
        if (!FD)
          continue;
        // Propagates inferred attributes via the most recent declaration to
        // ensure visibility for callers in post-order analysis.
        FD = getDeclWithMergedLifetimeBoundAttrs(FD);
        ParmVarDecl *InferredPVD = const_cast<ParmVarDecl *>(
            FD->getParamDecl(PVD->getFunctionScopeIndex()));
        if (!InferredPVD->hasAttr<LifetimeBoundAttr>())
          InferredPVD->addAttr(
              LifetimeBoundAttr::CreateImplicit(AST, PVD->getLocation()));
      }
    }
  }
};
} // namespace

void runLifetimeChecker(const LoanPropagationAnalysis &LP,
                        const MovedLoansAnalysis &MovedLoans,
                        const LiveOriginsAnalysis &LO, FactManager &FactMgr,
                        AnalysisDeclContext &ADC,
                        LifetimeSafetySemaHelper *SemaHelper) {
  llvm::TimeTraceScope TimeProfile("LifetimeChecker");
  LifetimeChecker Checker(LP, MovedLoans, LO, FactMgr, ADC, SemaHelper);
}

} // namespace clang::lifetimes::internal
