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
  Confidence ConfidenceLevel;
};

using AnnotationTarget =
    llvm::PointerUnion<const ParmVarDecl *, const CXXMethodDecl *>;

class LifetimeChecker {
private:
  llvm::DenseMap<LoanID, PendingWarning> FinalWarningsMap;
  llvm::DenseMap<AnnotationTarget, const Expr *> AnnotationWarningsMap;
  llvm::DenseMap<const ParmVarDecl *, const Expr *> NoescapeWarningsMap;
  const LoanPropagationAnalysis &LoanPropagation;
  const LiveOriginsAnalysis &LiveOrigins;
  const FactManager &FactMgr;
  LifetimeSafetySemaHelper *SemaHelper;
  ASTContext &AST;

public:
  LifetimeChecker(const LoanPropagationAnalysis &LoanPropagation,
                  const LiveOriginsAnalysis &LiveOrigins, const FactManager &FM,
                  AnalysisDeclContext &ADC,
                  LifetimeSafetySemaHelper *SemaHelper)
      : LoanPropagation(LoanPropagation), LiveOrigins(LiveOrigins), FactMgr(FM),
        SemaHelper(SemaHelper), AST(ADC.getASTContext()) {
    for (const CFGBlock *B : *ADC.getAnalysis<PostOrderCFGView>())
      for (const Fact *F : FactMgr.getFacts(B))
        if (const auto *EF = F->getAs<ExpireFact>())
          checkExpiry(EF);
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
    for (LoanID LID : EscapedLoans) {
      const Loan *L = FactMgr.getLoanMgr().getLoan(LID);
      if (const auto *PL = dyn_cast<PlaceholderLoan>(L)) {
        if (const auto *PVD = PL->getParmVarDecl()) {
          if (PVD->hasAttr<NoEscapeAttr>()) {
            NoescapeWarningsMap.try_emplace(PVD, OEF->getEscapeExpr());
            continue;
          }
          if (PVD->hasAttr<LifetimeBoundAttr>())
            continue;
          AnnotationWarningsMap.try_emplace(PVD, OEF->getEscapeExpr());
        } else if (const auto *MD = PL->getMethodDecl()) {
          if (!implicitObjectParamIsLifetimeBound(MD))
            AnnotationWarningsMap.try_emplace(MD, OEF->getEscapeExpr());
        }
      }
    }
  }

  /// Checks for use-after-free & use-after-return errors when a loan expires.
  ///
  /// This method examines all live origins at the expiry point and determines
  /// if any of them hold the expiring loan. If so, it creates a pending
  /// warning with the appropriate confidence level based on the liveness
  /// information. The confidence reflects whether the origin is definitely
  /// or maybe live at this point.
  ///
  /// Note: This implementation considers only the confidence of origin
  /// liveness. Future enhancements could also consider the confidence of loan
  /// propagation (e.g., a loan may only be held on some execution paths).
  void checkExpiry(const ExpireFact *EF) {
    LoanID ExpiredLoan = EF->getLoanID();
    LivenessMap Origins = LiveOrigins.getLiveOriginsAt(EF);
    Confidence CurConfidence = Confidence::None;
    // The UseFact or OriginEscapesFact most indicative of a lifetime error,
    // prioritized by earlier source location.
    llvm::PointerUnion<const UseFact *, const OriginEscapesFact *>
        BestCausingFact = nullptr;

    for (auto &[OID, LiveInfo] : Origins) {
      LoanSet HeldLoans = LoanPropagation.getLoans(OID, EF);
      if (!HeldLoans.contains(ExpiredLoan))
        continue;
      // Loan is defaulted.
      Confidence NewConfidence = livenessKindToConfidence(LiveInfo.Kind);
      if (CurConfidence < NewConfidence) {
        CurConfidence = NewConfidence;
        BestCausingFact = LiveInfo.CausingFact;
      }
    }
    if (!BestCausingFact)
      return;
    // We have a use-after-free.
    Confidence LastConf = FinalWarningsMap.lookup(ExpiredLoan).ConfidenceLevel;
    if (LastConf >= CurConfidence)
      return;
    FinalWarningsMap[ExpiredLoan] = {/*ExpiryLoc=*/EF->getExpiryLoc(),
                                     /*BestCausingFact=*/BestCausingFact,
                                     /*ConfidenceLevel=*/CurConfidence};
  }

  void issuePendingWarnings() {
    if (!SemaHelper)
      return;
    for (const auto &[LID, Warning] : FinalWarningsMap) {
      const Loan *L = FactMgr.getLoanMgr().getLoan(LID);
      const auto *BL = cast<PathLoan>(L);
      const Expr *IssueExpr = BL->getIssueExpr();
      llvm::PointerUnion<const UseFact *, const OriginEscapesFact *>
          CausingFact = Warning.CausingFact;
      Confidence Confidence = Warning.ConfidenceLevel;
      SourceLocation ExpiryLoc = Warning.ExpiryLoc;

      if (const auto *UF = CausingFact.dyn_cast<const UseFact *>())
        SemaHelper->reportUseAfterFree(IssueExpr, UF->getUseExpr(), ExpiryLoc,
                                       Confidence);
      else if (const auto *OEF =
                   CausingFact.dyn_cast<const OriginEscapesFact *>())
        SemaHelper->reportUseAfterReturn(IssueExpr, OEF->getEscapeExpr(),
                                         ExpiryLoc, Confidence);
      else
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
    for (auto [PVD, EscapeExpr] : NoescapeWarningsMap)
      SemaHelper->reportNoescapeViolation(PVD, EscapeExpr);
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
                        const LiveOriginsAnalysis &LO,
                        const FactManager &FactMgr, AnalysisDeclContext &ADC,
                        LifetimeSafetySemaHelper *SemaHelper) {
  llvm::TimeTraceScope TimeProfile("LifetimeChecker");
  LifetimeChecker Checker(LP, LO, FactMgr, ADC, SemaHelper);
}

} // namespace clang::lifetimes::internal
