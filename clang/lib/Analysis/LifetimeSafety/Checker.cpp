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

static bool causingFactDominatesExpiry(LivenessKind K) {
  switch (K) {
  case LivenessKind::Must:
    return true;
  case LivenessKind::Maybe:
  case LivenessKind::Dead:
    return false;
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
  bool CausingFactDominatesExpiry;
};

using AnnotationTarget =
    llvm::PointerUnion<const ParmVarDecl *, const CXXMethodDecl *>;
using EscapingTarget = LifetimeSafetySemaHelper::EscapingTarget;

class LifetimeChecker {
private:
  llvm::DenseMap<LoanID, PendingWarning> FinalWarningsMap;
  llvm::DenseMap<AnnotationTarget, EscapingTarget> AnnotationWarningsMap;
  llvm::DenseMap<const ParmVarDecl *, EscapingTarget> NoescapeWarningsMap;
  llvm::DenseSet<const Decl *> VerifiedLiftimeboundEscapes;
  const LoanPropagationAnalysis &LoanPropagation;
  const MovedLoansAnalysis &MovedLoans;
  const LiveOriginsAnalysis &LiveOrigins;
  FactManager &FactMgr;
  LifetimeSafetySemaHelper *SemaHelper;
  ASTContext &AST;
  const Decl *FD;

  static SourceLocation
  GetFactLoc(llvm::PointerUnion<const UseFact *, const OriginEscapesFact *> F) {
    if (const auto *UF = F.dyn_cast<const UseFact *>())
      return UF->getUseExpr()->getExprLoc();
    if (const auto *OEF = F.dyn_cast<const OriginEscapesFact *>()) {
      if (auto *ReturnEsc = dyn_cast<ReturnEscapeFact>(OEF))
        return ReturnEsc->getReturnExpr()->getExprLoc();
      if (auto *FieldEsc = dyn_cast<FieldEscapeFact>(OEF))
        return FieldEsc->getFieldDecl()->getLocation();
    }
    llvm_unreachable("unhandled causing fact in PointerUnion");
  }

public:
  LifetimeChecker(const LoanPropagationAnalysis &LoanPropagation,
                  const MovedLoansAnalysis &MovedLoans,
                  const LiveOriginsAnalysis &LiveOrigins, FactManager &FM,
                  AnalysisDeclContext &ADC,
                  LifetimeSafetySemaHelper *SemaHelper)
      : LoanPropagation(LoanPropagation), MovedLoans(MovedLoans),
        LiveOrigins(LiveOrigins), FactMgr(FM), SemaHelper(SemaHelper),
        AST(ADC.getASTContext()), FD(ADC.getDecl()) {
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
    reportLifetimeboundViolations();
    reportMisplacedLifetimebound();
    reportMeaninglessLifetimebound();
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
    auto CheckParam = [&](const ParmVarDecl *PVD, bool IsMoved) {
      // NoEscape param should not escape.
      if (PVD->hasAttr<NoEscapeAttr>()) {
        if (auto *ReturnEsc = dyn_cast<ReturnEscapeFact>(OEF))
          NoescapeWarningsMap.try_emplace(PVD, ReturnEsc->getReturnExpr());
        if (auto *FieldEsc = dyn_cast<FieldEscapeFact>(OEF))
          NoescapeWarningsMap.try_emplace(PVD, FieldEsc->getFieldDecl());
        if (auto *GlobalEsc = dyn_cast<GlobalEscapeFact>(OEF))
          NoescapeWarningsMap.try_emplace(PVD, GlobalEsc->getGlobal());
        return;
      }
      // Skip annotation suggestion for moved loans, as ownership transfer
      // obscures the lifetime relationship (e.g., shared_ptr from unique_ptr).
      if (IsMoved)
        return;
      if (PVD->hasAttr<LifetimeBoundAttr>()) {
        // Track that this lifetimebound parameter correctly escapes.
        VerifiedLiftimeboundEscapes.insert(PVD);
      } else {
        // Otherwise, suggest lifetimebound for parameter escaping through
        // return or a field in constructor.
        if (auto *ReturnEsc = dyn_cast<ReturnEscapeFact>(OEF))
          AnnotationWarningsMap.try_emplace(PVD, ReturnEsc->getReturnExpr());
        else if (auto *FieldEsc = dyn_cast<FieldEscapeFact>(OEF);
                 FieldEsc && isa<CXXConstructorDecl>(FD))
          AnnotationWarningsMap.try_emplace(PVD, FieldEsc->getFieldDecl());
      }
      // TODO: Suggest lifetime_capture_by(this) for parameter escaping to a
      // field!
    };
    auto CheckImplicitThis = [&](const CXXMethodDecl *MD) {
      if (implicitObjectParamIsLifetimeBound(MD))
        VerifiedLiftimeboundEscapes.insert(MD);
      else if (auto *ReturnEsc = dyn_cast<ReturnEscapeFact>(OEF))
        AnnotationWarningsMap.try_emplace(MD, ReturnEsc->getReturnExpr());
    };
    auto MovedAtEscape = MovedLoans.getMovedLoans(OEF);
    for (LoanID LID : EscapedLoans) {
      const Loan *L = FactMgr.getLoanMgr().getLoan(LID);
      const AccessPath &AP = L->getAccessPath();
      if (const auto *PVD = AP.getAsPlaceholderParam())
        CheckParam(PVD, /*IsMoved=*/MovedAtEscape.lookup(LID));
      else if (const auto *MD = AP.getAsPlaceholderThis())
        CheckImplicitThis(MD);
    }
  }

  /// Checks for use-after-free & use-after-return errors when an access path
  /// expires (e.g., a variable goes out of scope).
  ///
  /// When a path expires, all loans having this path expires.
  /// This method examines all live origins and reports warnings for loans they
  /// hold that are prefixed by the expired path.
  void checkExpiry(const ExpireFact *EF) {
    const AccessPath &ExpiredPath = EF->getAccessPath();
    LivenessMap Origins = LiveOrigins.getLiveOriginsAt(EF);
    for (auto &[OID, LiveInfo] : Origins) {
      LoanSet HeldLoans = LoanPropagation.getLoans(OID, EF);
      for (LoanID HeldLoanID : HeldLoans) {
        const Loan *HeldLoan = FactMgr.getLoanMgr().getLoan(HeldLoanID);
        if (ExpiredPath != HeldLoan->getAccessPath())
          continue;
        // HeldLoan is expired because its AccessPath is expired.
        PendingWarning &CurWarning = FinalWarningsMap[HeldLoan->getID()];
        const Expr *MovedExpr = nullptr;
        if (auto *ME = MovedLoans.getMovedLoans(EF).lookup(HeldLoanID))
          MovedExpr = *ME;
        // Skip if we already have a dominating causing fact.
        if (CurWarning.CausingFactDominatesExpiry)
          continue;
        if (causingFactDominatesExpiry(LiveInfo.Kind))
          CurWarning.CausingFactDominatesExpiry = true;
        CurWarning.CausingFact = LiveInfo.CausingFact;
        CurWarning.ExpiryLoc = EF->getExpiryLoc();
        CurWarning.MovedExpr = MovedExpr;
        CurWarning.InvalidatedByExpr = nullptr;
      }
    }
  }

  /// Checks for use-after-invalidation errors when a container is modified.
  ///
  /// This method identifies origins that are live at the point of invalidation
  /// and checks if they hold loans that are invalidated by the operation
  /// (e.g., iterators into a vector that is being pushed to).
  void checkInvalidation(const InvalidateOriginFact *IOF) {
    OriginID InvalidatedOrigin = IOF->getInvalidatedOrigin();
    /// Get loans directly pointing to the invalidated container
    LoanSet DirectlyInvalidatedLoans =
        LoanPropagation.getLoans(InvalidatedOrigin, IOF);
    auto IsInvalidated = [&](const Loan *L) {
      for (LoanID InvalidID : DirectlyInvalidatedLoans) {
        const Loan *InvalidL = FactMgr.getLoanMgr().getLoan(InvalidID);
        if (InvalidL->getAccessPath() == L->getAccessPath())
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
          bool CurDomination = causingFactDominatesExpiry(LiveInfo.Kind);
          bool LastDomination =
              FinalWarningsMap.lookup(LiveLoanID).CausingFactDominatesExpiry;
          if (!LastDomination) {
            FinalWarningsMap[LiveLoanID] = {
                /*ExpiryLoc=*/{},
                /*CausingFact=*/LiveInfo.CausingFact,
                /*MovedExpr=*/nullptr,
                /*InvalidatedByExpr=*/IOF->getInvalidationExpr(),
                /*CausingFactDominatesExpiry=*/CurDomination};
          }
        }
    }
  }

  void issuePendingWarnings() {
    if (!SemaHelper)
      return;
    for (const auto &[LID, Warning] : FinalWarningsMap) {
      const Loan *L = FactMgr.getLoanMgr().getLoan(LID);
      const Expr *IssueExpr = L->getIssuingExpr();
      llvm::PointerUnion<const UseFact *, const OriginEscapesFact *>
          CausingFact = Warning.CausingFact;
      const ParmVarDecl *InvalidatedPVD =
          L->getAccessPath().getAsPlaceholderParam();
      const Expr *MovedExpr = Warning.MovedExpr;
      SourceLocation ExpiryLoc = Warning.ExpiryLoc;

      if (const auto *UF = CausingFact.dyn_cast<const UseFact *>()) {
        if (Warning.InvalidatedByExpr) {
          if (IssueExpr)
            // Use-after-invalidation of an object on stack.
            SemaHelper->reportUseAfterInvalidation(IssueExpr, UF->getUseExpr(),
                                                   Warning.InvalidatedByExpr);
          else if (InvalidatedPVD)
            // Use-after-invalidation of a parameter.
            SemaHelper->reportUseAfterInvalidation(
                InvalidatedPVD, UF->getUseExpr(), Warning.InvalidatedByExpr);

        } else
          // Scope-based expiry (use-after-scope).
          SemaHelper->reportUseAfterScope(IssueExpr, UF->getUseExpr(),
                                          MovedExpr, ExpiryLoc);
      } else if (const auto *OEF =
                     CausingFact.dyn_cast<const OriginEscapesFact *>()) {
        if (Warning.InvalidatedByExpr) {
          if (const auto *FieldEscape = dyn_cast<FieldEscapeFact>(OEF)) {
            // Invalidated object escapes to a field.
            if (IssueExpr)
              // Invalidated object on stack escapes to a field.
              SemaHelper->reportInvalidatedField(IssueExpr,
                                                 FieldEscape->getFieldDecl(),
                                                 Warning.InvalidatedByExpr);
            else if (InvalidatedPVD)
              // Invalidated parameter escapes to a field.
              SemaHelper->reportInvalidatedField(InvalidatedPVD,
                                                 FieldEscape->getFieldDecl(),
                                                 Warning.InvalidatedByExpr);
          } else if (const auto *GlobalEscape =
                         dyn_cast<GlobalEscapeFact>(OEF)) {
            // Invalidated object escapes to global or static storage.
            if (IssueExpr)
              // Invalidated object on stack escapes to global or static
              // storage.
              SemaHelper->reportInvalidatedGlobal(IssueExpr,
                                                  GlobalEscape->getGlobal(),
                                                  Warning.InvalidatedByExpr);
            else if (InvalidatedPVD)
              // Invalidated parameter escapes to global or static storage.
              SemaHelper->reportInvalidatedGlobal(InvalidatedPVD,
                                                  GlobalEscape->getGlobal(),
                                                  Warning.InvalidatedByExpr);
          } else if (isa<ReturnEscapeFact>(OEF)) {
            // FIXME: Diagnose invalidated return escapes separately.
          } else
            llvm_unreachable("Unhandled OriginEscapesFact type");
        } else if (const auto *RetEscape = dyn_cast<ReturnEscapeFact>(OEF))
          // Return stack address.
          SemaHelper->reportUseAfterReturn(
              IssueExpr, RetEscape->getReturnExpr(), MovedExpr);
        else if (const auto *FieldEscape = dyn_cast<FieldEscapeFact>(OEF))
          // Dangling field.
          SemaHelper->reportDanglingField(
              IssueExpr, FieldEscape->getFieldDecl(), MovedExpr, ExpiryLoc);
        else if (const auto *GlobalEscape = dyn_cast<GlobalEscapeFact>(OEF))
          // Global escape.
          SemaHelper->reportDanglingGlobal(IssueExpr, GlobalEscape->getGlobal(),
                                           MovedExpr, ExpiryLoc);
        else
          llvm_unreachable("Unhandled OriginEscapesFact type");
      } else
        llvm_unreachable("Unhandled CausingFact type");
    }
  }

  // Returns declarations that should be annotated with lifetime attributes
  // in order to annotate FDef: the canonical declaration and the earliest
  // redeclarations in each other file. This defines the placement policy for
  // lifetime annotations. Each target is paired with its corresponding warning
  // scope.
  llvm::SmallVector<std::pair<const FunctionDecl *, WarningScope>, 2>
  getTargetDeclsForAttr(const FunctionDecl *FDef) {
    if (!FDef)
      return {};

    assert(FDef->isThisDeclarationADefinition() &&
           "Expected FunctionDecl to be a definition");

    const auto &SM = FDef->getASTContext().getSourceManager();

    auto GetFile = [&SM](const FunctionDecl *FD) {
      return SM.getFileID(SM.getExpansionLoc(FD->getLocation()));
    };

    const FileID DefFile = GetFile(FDef);
    const FunctionDecl *CanonicalDecl = FDef->getCanonicalDecl();
    llvm::SmallVector<std::pair<const FunctionDecl *, WarningScope>, 2> Targets{
        {CanonicalDecl, GetFile(CanonicalDecl) == DefFile
                            ? WarningScope::IntraTU
                            : WarningScope::CrossTU}};

    // Find the earliest redeclaration in each file other than the definition
    // file.
    auto AddCrossTUDecl = [&](const FunctionDecl *FD) {
      FileID File = GetFile(FD);
      if (File == DefFile)
        return;
      for (auto [SeenFD, _] : Targets)
        if (GetFile(SeenFD) == File)
          return;
      Targets.push_back({FD, WarningScope::CrossTU});
    };

    // We iterate in reverse order (from most recent to oldest) to find
    // the first declaration in each file.

    // Store in temporary variable to manually extend lifetime
    auto redecls = llvm::to_vector(FDef->redecls());

    for (const FunctionDecl *Redecl : llvm::reverse(redecls))
      AddCrossTUDecl(Redecl);

    return Targets;
  }

  void suggestWithScopeForParmVar(const ParmVarDecl *PVD,
                                  EscapingTarget EscapeTarget) {
    if (llvm::isa<const VarDecl *>(EscapeTarget))
      return;

    for (auto [Decl, Scope] : getTargetDeclsForAttr(cast<FunctionDecl>(FD))) {
      const auto *ParmToAnnotate =
          Decl->getParamDecl(PVD->getFunctionScopeIndex());
      SemaHelper->suggestLifetimeboundToParmVar(Scope, ParmToAnnotate,
                                                EscapeTarget);
    }
  }

  void suggestWithScopeForImplicitThis(const CXXMethodDecl *MD,
                                       const Expr *EscapeExpr) {
    for (auto [Decl, Scope] : getTargetDeclsForAttr(MD)) {
      SemaHelper->suggestLifetimeboundToImplicitThis(
          Scope, cast<CXXMethodDecl>(Decl), EscapeExpr);
    }
  }

  void suggestAnnotations() {
    if (!SemaHelper)
      return;
    for (auto [Target, EscapeTarget] : AnnotationWarningsMap) {
      if (const auto *PVD = Target.dyn_cast<const ParmVarDecl *>())
        suggestWithScopeForParmVar(PVD, EscapeTarget);
      else if (const auto *MD = Target.dyn_cast<const CXXMethodDecl *>()) {
        if (const auto *EscapeExpr = EscapeTarget.dyn_cast<const Expr *>())
          suggestWithScopeForImplicitThis(MD, EscapeExpr);
        else
          llvm_unreachable("Implicit this can only escape via Expr (return)");
      }
    }
  }

  void reportNoescapeViolations() {
    for (auto [PVD, EscapeTarget] : NoescapeWarningsMap) {
      if (const auto *E = EscapeTarget.dyn_cast<const Expr *>())
        SemaHelper->reportNoescapeViolation(PVD, E);
      else if (const auto *FD = EscapeTarget.dyn_cast<const FieldDecl *>())
        SemaHelper->reportNoescapeViolation(PVD, FD);
      else if (const auto *G = EscapeTarget.dyn_cast<const VarDecl *>())
        SemaHelper->reportNoescapeViolation(PVD, G);
      else
        llvm_unreachable("Unhandled EscapingTarget type");
    }
  }

  void reportLifetimeboundViolations() {
    if (!isa<FunctionDecl>(FD))
      return;
    if (const auto *MD = dyn_cast<CXXMethodDecl>(FD);
        MD && getImplicitObjectParamLifetimeBoundAttr(MD) &&
        !VerifiedLiftimeboundEscapes.contains(MD))
      SemaHelper->reportLifetimeboundViolation(MD);
    for (const ParmVarDecl *PVD : cast<FunctionDecl>(FD)->parameters()) {
      if (!PVD->hasAttr<LifetimeBoundAttr>())
        continue;
      bool isImplicit = PVD->getAttr<LifetimeBoundAttr>()->isImplicit();
      bool Escapes = VerifiedLiftimeboundEscapes.contains(PVD);
      assert((!isImplicit || Escapes || isInStlNamespace(FD)) &&
             "Implicit lifetimebound parameters "
             "should escape through return");
      if (!isImplicit && !Escapes)
        SemaHelper->reportLifetimeboundViolation(PVD);
    }
  }

  // Reports lifetimebound attributes that are placed on a function definition
  // but not on the corresponding declaration.
  void reportMisplacedLifetimebound() {
    const FunctionDecl *FDef = dyn_cast<FunctionDecl>(FD);
    if (!FDef)
      return;

    auto TargetDecls = getTargetDeclsForAttr(FDef);
    // Check if implicit 'this' has lifetimebound on definition but not on
    // declaration.
    if (const auto *MDef = dyn_cast<CXXMethodDecl>(FDef);
        MDef && getDirectImplicitObjectLifetimeBoundAttr(MDef))
      for (auto [Decl, Scope] : TargetDecls) {
        const auto *MDecl = cast<CXXMethodDecl>(Decl);
        if (!getDirectImplicitObjectLifetimeBoundAttr(MDecl))
          SemaHelper->reportMisplacedLifetimebound(Scope, MDef, MDecl);
      }

    // Check each parameter for explicit lifetimebound on definition but not on
    // declaration.
    for (const auto *PDef : FDef->parameters()) {
      const auto *Attr = PDef->getAttr<LifetimeBoundAttr>();
      if (!Attr || Attr->isImplicit())
        continue;
      for (auto [Decl, Scope] : TargetDecls) {
        const auto *PDecl = Decl->getParamDecl(PDef->getFunctionScopeIndex());
        if (!PDecl->hasAttr<LifetimeBoundAttr>())
          SemaHelper->reportMisplacedLifetimebound(Scope, PDef, PDecl);
      }
    }
  }

  static bool canBeLifetimebound(QualType QT) {
    if (QT->isReferenceType() || QT->isPointerType())
      return true;
    if (isGslOwnerType(QT) || QT->isScalarType())
      return false;
    return true;
  }

  void reportMeaninglessLifetimebound() {
    for (const auto &PVD : cast<FunctionDecl>(FD)->parameters()) {
      if (!PVD->hasAttr<LifetimeBoundAttr>())
        continue;
      if (!canBeLifetimebound(PVD->getType()))
        SemaHelper->reportMeaninglessLifetimebound(PVD);
    }
  }

  void inferAnnotations() {
    for (auto [Target, EscapeTarget] : AnnotationWarningsMap) {
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
