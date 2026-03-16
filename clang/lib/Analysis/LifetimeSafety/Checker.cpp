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

struct AliasAssignmentSearchResult {
  const llvm::SmallVector<AssignmentPair> Payload;
  bool SearchComplete;
  const ValueDecl *LastDestDecl;
  std::optional<OriginID> LastOrigin;
};

class LifetimeChecker {
private:
  llvm::DenseMap<LoanID, PendingWarning> FinalWarningsMap;
  llvm::DenseMap<AnnotationTarget, EscapingTarget> AnnotationWarningsMap;
  llvm::DenseMap<const ParmVarDecl *, EscapingTarget> NoescapeWarningsMap;
  const LoanPropagationAnalysis &LoanPropagation;
  const MovedLoansAnalysis &MovedLoans;
  const LiveOriginsAnalysis &LiveOrigins;
  FactManager &FactMgr;
  LifetimeSafetySemaHelper *SemaHelper;
  ASTContext &AST;
  const Decl *FD;
  AnalysisDeclContext &ADC;

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
        AST(ADC.getASTContext()), FD(ADC.getDecl()), ADC(ADC) {
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
        if (auto *GlobalEsc = dyn_cast<GlobalEscapeFact>(OEF))
          NoescapeWarningsMap.try_emplace(PVD, GlobalEsc->getGlobal());
        return;
      }
      // Suggest lifetimebound for parameter escaping through return or a field
      // in constructor.
      if (!PVD->hasAttr<LifetimeBoundAttr>()) {
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
      if (!implicitObjectParamIsLifetimeBound(MD))
        if (auto *ReturnEsc = dyn_cast<ReturnEscapeFact>(OEF))
          AnnotationWarningsMap.try_emplace(MD, ReturnEsc->getReturnExpr());
    };
    for (LoanID LID : EscapedLoans) {
      const Loan *L = FactMgr.getLoanMgr().getLoan(LID);
      const AccessPath &AP = L->getAccessPath();
      if (const auto *PVD = AP.getAsPlaceholderParam())
        CheckParam(PVD);
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

  std::optional<llvm::SmallVector<AssignmentPair>>
  getAliasListInMultiBlock(const CFGBlock *StartBlock, const LoanID EndLoanID,
                           OriginID *StartOID) {
    const ValueDecl *LastDestDecl = nullptr;
    llvm::SmallVector<const CFGBlock *> PendingBlocks;
    std::optional<AssignmentPair> StartStmt = std::nullopt;
    std::optional<AssignmentPair> EndStmt = std::nullopt;
    std::optional<OriginID> LastOriginID = std::nullopt;
    llvm::SmallPtrSet<const CFGBlock *, 32> VistedBlocks;
    llvm::DenseMap<AssignmentPair, AssignmentPair> VistedExprs;

    const auto AliasStmtFilter = [&VistedExprs](const AssignmentPair StartStmt,
                                                const AssignmentPair EndStmt) {
      llvm::SmallVector<AssignmentPair> AliasStmts;
      for (auto Stmt = StartStmt; Stmt != EndStmt;
           Stmt = VistedExprs.at(Stmt)) {
        AliasStmts.push_back(Stmt);
      }
      AliasStmts.push_back(EndStmt);
      return AliasStmts;
    };

    PendingBlocks.push_back(StartBlock);

    for (size_t i = 0; i < PendingBlocks.size(); ++i) {
      const CFGBlock *CurrBlock = PendingBlocks[i];

      const auto [BlockAliasList, Success, CurrLastDestDecl, CurrLastOriginID] =
          getAliasListCore(CurrBlock, EndLoanID, StartOID, LastDestDecl,
                           LastOriginID);
      if (CurrLastDestDecl)
        LastDestDecl = CurrLastDestDecl;
      if (CurrLastOriginID.has_value())
        LastOriginID = CurrLastOriginID;

      if (!BlockAliasList.empty()) {
        if (VistedExprs.empty()) {
          StartStmt = BlockAliasList[0];
        }

        for (size_t i = 0; i < BlockAliasList.size() - 1; ++i) {
          VistedExprs.insert({BlockAliasList[i], BlockAliasList[i + 1]});
        }

        if (EndStmt.has_value())
          VistedExprs.insert({EndStmt.value(), BlockAliasList[0]});

        EndStmt = BlockAliasList[BlockAliasList.size() - 1];
      }

      if (Success && StartStmt.has_value() && EndStmt.has_value()) {
        return AliasStmtFilter(StartStmt.value(), EndStmt.value());
      }

      for (const auto Block : CurrBlock->preds()) {
        if (Block && VistedBlocks.insert(Block).second)
          PendingBlocks.push_back(Block);
      }

      if (VistedBlocks.size() >= 32 && StartStmt.has_value() &&
          EndStmt.has_value()) {
        return AliasStmtFilter(StartStmt.value(), EndStmt.value());
      }
    }

    if (StartStmt.has_value() && EndStmt.has_value()) {
      return AliasStmtFilter(StartStmt.value(), EndStmt.value());
    }

    return std::nullopt;
  }

  std::optional<OriginSrcExpr> GetPureSrcExpr(const Expr *TargetExpr) {
    if (!TargetExpr)
      return std::nullopt;
    const Expr *SExpr = TargetExpr->IgnoreParenCasts();
    if (!SExpr)
      return std::nullopt;

    if (const auto *SDRExpr = llvm::dyn_cast<DeclRefExpr>(SExpr)) {
      return SDRExpr;
    }
    if (const auto *STMExpr = llvm::dyn_cast<CXXTemporaryObjectExpr>(SExpr)) {
      return STMExpr;
    }
    if (const auto *SCExpr = llvm::dyn_cast<CallExpr>(SExpr)) {
      return SCExpr;
    }

    if (const auto *SCCExpr = llvm::dyn_cast<CXXConstructExpr>(SExpr)) {
      if (SCCExpr->getNumArgs() > 0)
        return GetPureSrcExpr(SCCExpr->getArg(0));
    }
    if (const auto *SUOExpr = llvm::dyn_cast<UnaryOperator>(SExpr)) {
      return GetPureSrcExpr(SUOExpr->getSubExpr());
    }
    if (const auto *SCBExpr = llvm::dyn_cast<CXXBindTemporaryExpr>(SExpr)) {
      return GetPureSrcExpr(SCBExpr->getSubExpr());
    }

    return std::nullopt;
  }

  /// Retrieves a list of assignment chains for use-after-scope analysis.
  ///
  /// To help users understand the data flow, we track where the problematic
  /// address originated.
  AliasAssignmentSearchResult
  getAliasListCore(const CFGBlock *Block, const LoanID EndLoanID,
                   OriginID *TargetOID, const ValueDecl *LastDestDecl = nullptr,
                   const std::optional<OriginID> LastOriginID = std::nullopt) {
    llvm::SmallVector<AssignmentPair> AliasStmts;
    const ValueDecl *DestDecl = LastDestDecl;
    const auto Facts = FactMgr.getFacts(Block);
    bool FetchLoan = false;
    auto IssueOriginID = LastOriginID;

    for (auto F = Facts.rbegin(); F != Facts.rend(); ++F) {
      if (const auto *OFF = (*F)->getAs<OriginFlowFact>()) {
        if (IssueOriginID.has_value() &&
            OFF->getDestOriginID() == IssueOriginID.value()) {
          FetchLoan = true;
        }
        if (OFF->getDestOriginID() == *TargetOID) {
          const auto HeldLoans =
              LoanPropagation.getLoans(OFF->getSrcOriginID(), OFF);

          if (HeldLoans.contains(EndLoanID)) {
            const auto TargetOrigin =
                FactMgr.getOriginMgr().getOrigin(OFF->getDestOriginID());

            if (DestDecl == nullptr) {
              if (const ValueDecl *DDecl = TargetOrigin.getDecl()) {
                DestDecl = DDecl;
              }
            } else {
              auto SExpr = GetPureSrcExpr(TargetOrigin.getExpr());
              if (!SExpr.has_value()) {
                const auto SrcOrigin =
                    FactMgr.getOriginMgr().getOrigin(OFF->getSrcOriginID());
                SExpr = GetPureSrcExpr(SrcOrigin.getExpr());
              }

              if (SExpr.has_value()) {
                AliasStmts.push_back({SExpr.value(), DestDecl});
                DestDecl = nullptr;
              }
            }
            *TargetOID = OFF->getSrcOriginID();
          }
        }
      } else if (const auto *IF = (*F)->getAs<IssueFact>()) {
        if (IF->getLoanID() == EndLoanID) {
          IssueOriginID = IF->getOriginID();
        }
      }

      if (FetchLoan) {
        return {AliasStmts, true, DestDecl, IssueOriginID};
      }
    }
    return {AliasStmts, false, DestDecl, IssueOriginID};
  }

  std::optional<llvm::SmallVector<AssignmentPair>>
  getAliasList(const UseFact *UF, const LoanID End, const bool InOneBlock) {
    const CFGBlock *IssueBlock =
        ADC.getCFGStmtMap()->getBlock(UF->getUseExpr());
    assert(IssueBlock && "Searching CFGBlock failed");

    for (const OriginList *Cur = UF->getUsedOrigins(); Cur;
         Cur = Cur->peelOuterOrigin()) {
      auto TargetOID = Cur->getOuterOriginID();
      if (InOneBlock) {
        AliasAssignmentSearchResult Result =
            getAliasListCore(IssueBlock, End, &TargetOID);
        if (!Result.Payload.empty())
          return Result.Payload;
      } else {
        auto Result = getAliasListInMultiBlock(IssueBlock, End, &TargetOID);
        if (Result.has_value())
          return Result.value();
      }
    }

    return std::nullopt;
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

        } else {
          // Scope-based expiry (use-after-scope).
          const CFGStmtMap *CurrCFGStmtMap = ADC.getCFGStmtMap();
          const auto AliasExprs =
              getAliasList(UF, LID,
                           CurrCFGStmtMap->getBlock(UF->getUseExpr()) ==
                               CurrCFGStmtMap->getBlock(IssueExpr));
          if (!AliasExprs.has_value()) {
            llvm::dbgs() << "Search variable assignment chain failed\n";
          }

          SemaHelper->reportUseAfterFree(IssueExpr, UF->getUseExpr(), MovedExpr,
                                         AliasExprs.value_or({}), ExpiryLoc);
        }
      } else if (const auto *OEF =
                     CausingFact.dyn_cast<const OriginEscapesFact *>()) {
        if (const auto *RetEscape = dyn_cast<ReturnEscapeFact>(OEF))
          // Return stack address.
          SemaHelper->reportUseAfterReturn(
              IssueExpr, RetEscape->getReturnExpr(), MovedExpr, ExpiryLoc);
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
                                         EscapingTarget EscapeTarget) {
    if (llvm::isa<const VarDecl *>(EscapeTarget))
      return;

    if (const FunctionDecl *CrossTUDecl = getCrossTUDecl(*PVD, SM))
      SemaHelper->suggestLifetimeboundToParmVar(
          SuggestionScope::CrossTU,
          CrossTUDecl->getParamDecl(PVD->getFunctionScopeIndex()),
          EscapeTarget);
    else
      SemaHelper->suggestLifetimeboundToParmVar(SuggestionScope::IntraTU, PVD,
                                                EscapeTarget);
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
    for (auto [Target, EscapeTarget] : AnnotationWarningsMap) {
      if (const auto *PVD = Target.dyn_cast<const ParmVarDecl *>())
        suggestWithScopeForParmVar(SemaHelper, PVD, SM, EscapeTarget);
      else if (const auto *MD = Target.dyn_cast<const CXXMethodDecl *>()) {
        if (const auto *EscapeExpr = EscapeTarget.dyn_cast<const Expr *>())
          suggestWithScopeForImplicitThis(SemaHelper, MD, SM, EscapeExpr);
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
