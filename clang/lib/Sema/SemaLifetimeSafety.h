//===--- SemaLifetimeSafety.h - Sema support for lifetime safety =---------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  This file defines the Sema-specific implementation for lifetime safety
//  analysis. It provides diagnostic reporting and helper functions that bridge
//  the lifetime safety analysis framework with Sema's diagnostic engine.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LIB_SEMA_SEMALIFETIMESAFETY_H
#define LLVM_CLANG_LIB_SEMA_SEMALIFETIMESAFETY_H

#include "clang/Analysis/Analyses/LifetimeSafety/LifetimeSafety.h"
#include "clang/Analysis/Analyses/LifetimeSafety/Loans.h"
#include "clang/Basic/DiagnosticSema.h"
#include "clang/Lex/Lexer.h"
#include "clang/Sema/Sema.h"

namespace clang::lifetimes {

inline bool IsLifetimeSafetyDiagnosticEnabled(Sema &S, const Decl *D) {
  DiagnosticsEngine &Diags = S.getDiagnostics();
  return !Diags.isIgnored(diag::warn_lifetime_safety_use_after_scope,
                          D->getBeginLoc()) ||
         !Diags.isIgnored(diag::warn_lifetime_safety_use_after_scope_moved,
                          D->getBeginLoc()) ||
         !Diags.isIgnored(diag::warn_lifetime_safety_return_stack_addr,
                          D->getBeginLoc()) ||
         !Diags.isIgnored(diag::warn_lifetime_safety_return_stack_addr_moved,
                          D->getBeginLoc()) ||
         !Diags.isIgnored(diag::warn_lifetime_safety_invalidation,
                          D->getBeginLoc()) ||
         !Diags.isIgnored(diag::warn_lifetime_safety_noescape_escapes,
                          D->getBeginLoc());
}

inline __attribute__((always_inline)) void
formatLHSValueDeclForSema(const ValueDecl *TargetValue,
                          llvm::SmallVectorImpl<char> &LHSMsg) {
  if (TargetValue) {
    const llvm::StringRef PrefixStr = "variable '";
    const llvm::StringRef TargetName = TargetValue->getName();
    LHSMsg.append(PrefixStr.begin(), PrefixStr.end());
    LHSMsg.append(TargetName.begin(), TargetName.end());
    LHSMsg.push_back('\'');
  }
}

inline void reportAssignmentImpl(Sema &S, LoanEntity IssueEntity,
                                 const ValueDecl *LHS, const Expr *RHS,
                                 const SourceLocation LHSExploc) {
  llvm::SmallString<32> IssueMsg;
  llvm::SmallString<32> LHSMsg;
  llvm::SmallVector<ExprPrintingResult> SrcMsgList;
  formatLoanEntityForSema(IssueEntity, IssueMsg);
  formatLHSValueDeclForSema(LHS, LHSMsg);
  formatSrcExprForSema(RHS, SrcMsgList);

  if (SrcMsgList.size() == 1 &&
      llvm::isa<DeclRefExpr>(SrcMsgList[0].CurrExpr)) {
    S.Diag(LHSExploc, diag::note_lifetime_safety_note_alias_chain)
        << LHSMsg << IssueMsg;
  } else {
    for (const ExprPrintingResult &SrcMsg : llvm::reverse(SrcMsgList))
      S.Diag(SrcMsg.CurrExpr->getExprLoc(),
             diag::note_lifetime_safety_note_alias_chain)
          << SrcMsg.CurrExpr->getSourceRange() << SrcMsg.Str << IssueMsg;
    S.Diag(LHSExploc, diag::note_lifetime_safety_note_alias_chain)
        << LHSMsg << IssueMsg;
  }
}

inline void reportAssignment(Sema &S, LoanEntity IssueEntity,
                             const OriginDestExpr &LHS, const Expr *RHS) {
  if (!LHS || !RHS)
    return;

  if (const DeclRefExpr *LDExpr = llvm::dyn_cast<const DeclRefExpr *>(LHS)) {
    reportAssignmentImpl(S, IssueEntity, LDExpr->getDecl(), RHS,
                         LDExpr->getExprLoc());
  } else if (const ValueDecl *LVDecl = llvm::dyn_cast<const ValueDecl *>(LHS)) {
    reportAssignmentImpl(S, IssueEntity, LVDecl, RHS, LVDecl->getLocation());
  } else if (const MemberExpr *LVDecl =
                 llvm::dyn_cast<const MemberExpr *>(LHS)) {
    reportAssignmentImpl(S, IssueEntity, LVDecl->getMemberDecl(), RHS,
                         LVDecl->getExprLoc());
  }
}

class LifetimeSafetySemaHelperImpl : public LifetimeSafetySemaHelper {

public:
  LifetimeSafetySemaHelperImpl(Sema &S) : S(S) {}

  void reportUseAfterFree(const Expr *IssueExpr, const Expr *UseExpr,
                          const Expr *MovedExpr,
                          llvm::ArrayRef<AssignmentPair> AliasList,
                          SourceLocation FreeLoc) override {
    S.Diag(IssueExpr->getExprLoc(),
           MovedExpr ? diag::warn_lifetime_safety_use_after_scope_moved
                     : diag::warn_lifetime_safety_use_after_scope)
        << IssueExpr->getSourceRange();
    if (MovedExpr)
      S.Diag(MovedExpr->getExprLoc(), diag::note_lifetime_safety_moved_here)
          << MovedExpr->getSourceRange();
    S.Diag(FreeLoc, diag::note_lifetime_safety_destroyed_here);

    for (const AssignmentPair &AliasStmt : AliasList)
      reportAssignment(S, IssueExpr, AliasStmt.first, AliasStmt.second);

    S.Diag(UseExpr->getExprLoc(), diag::note_lifetime_safety_used_here)
        << UseExpr->getSourceRange();
  }

  void reportUseAfterReturn(const Expr *IssueExpr, const Expr *ReturnExpr,
                            const Expr *MovedExpr,
                            llvm::ArrayRef<AssignmentPair> AliasList,
                            SourceLocation ExpiryLoc) override {
    S.Diag(IssueExpr->getExprLoc(),
           MovedExpr ? diag::warn_lifetime_safety_return_stack_addr_moved
                     : diag::warn_lifetime_safety_return_stack_addr)
        << IssueExpr->getSourceRange();
    if (MovedExpr)
      S.Diag(MovedExpr->getExprLoc(), diag::note_lifetime_safety_moved_here)
          << MovedExpr->getSourceRange();

    for (const AssignmentPair &AliasStmt : AliasList)
      reportAssignment(S, IssueExpr, AliasStmt.first, AliasStmt.second);

    S.Diag(ReturnExpr->getExprLoc(), diag::note_lifetime_safety_returned_here)
        << ReturnExpr->getSourceRange();
  }

  void reportDanglingField(const Expr *IssueExpr,
                           const FieldDecl *DanglingField,
                           const Expr *MovedExpr,
                           llvm::ArrayRef<AssignmentPair> AliasList,
                           SourceLocation ExpiryLoc) override {
    S.Diag(IssueExpr->getExprLoc(),
           MovedExpr ? diag::warn_lifetime_safety_dangling_field_moved
                     : diag::warn_lifetime_safety_dangling_field)
        << IssueExpr->getSourceRange();
    if (MovedExpr)
      S.Diag(MovedExpr->getExprLoc(), diag::note_lifetime_safety_moved_here)
          << MovedExpr->getSourceRange();

    for (const AssignmentPair &AliasStmt : AliasList)
      reportAssignment(S, IssueExpr, AliasStmt.first, AliasStmt.second);

    S.Diag(DanglingField->getLocation(),
           diag::note_lifetime_safety_dangling_field_here)
        << DanglingField->getEndLoc();
  }

  void reportDanglingGlobal(const Expr *IssueExpr,
                            const VarDecl *DanglingGlobal,
                            const Expr *MovedExpr,
                            llvm::ArrayRef<AssignmentPair> AliasList,
                            SourceLocation ExpiryLoc) override {
    S.Diag(IssueExpr->getExprLoc(),
           MovedExpr ? diag::warn_lifetime_safety_dangling_global_moved
                     : diag::warn_lifetime_safety_dangling_global)
        << IssueExpr->getSourceRange();
    if (MovedExpr)
      S.Diag(MovedExpr->getExprLoc(), diag::note_lifetime_safety_moved_here)
          << MovedExpr->getSourceRange();

    for (const AssignmentPair &AliasStmt : AliasList)
      reportAssignment(S, IssueExpr, AliasStmt.first, AliasStmt.second);

    if (DanglingGlobal->isStaticLocal() || DanglingGlobal->isStaticDataMember())
      S.Diag(DanglingGlobal->getLocation(),
             diag::note_lifetime_safety_dangling_static_here)
          << DanglingGlobal->getEndLoc();
    else
      S.Diag(DanglingGlobal->getLocation(),
             diag::note_lifetime_safety_dangling_global_here)
          << DanglingGlobal->getEndLoc();
  }

  void reportUseAfterInvalidation(const Expr *IssueExpr, const Expr *UseExpr,
                                  const Expr *InvalidationExpr) override {
    S.Diag(IssueExpr->getExprLoc(), diag::warn_lifetime_safety_invalidation)
        << false << IssueExpr->getSourceRange();
    S.Diag(InvalidationExpr->getExprLoc(),
           diag::note_lifetime_safety_invalidated_here)
        << InvalidationExpr->getSourceRange();
    S.Diag(UseExpr->getExprLoc(), diag::note_lifetime_safety_used_here)
        << UseExpr->getSourceRange();
  }
  void reportUseAfterInvalidation(const ParmVarDecl *PVD, const Expr *UseExpr,
                                  const Expr *InvalidationExpr) override {
    S.Diag(PVD->getSourceRange().getBegin(),
           diag::warn_lifetime_safety_invalidation)
        << true << PVD->getSourceRange();
    S.Diag(InvalidationExpr->getExprLoc(),
           diag::note_lifetime_safety_invalidated_here)
        << InvalidationExpr->getSourceRange();
    S.Diag(UseExpr->getExprLoc(), diag::note_lifetime_safety_used_here)
        << UseExpr->getSourceRange();
  }

  void suggestLifetimeboundToParmVar(SuggestionScope Scope,
                                     const ParmVarDecl *ParmToAnnotate,
                                     llvm::ArrayRef<AssignmentPair> AliasList,
                                     EscapingTarget Target) override {
    unsigned DiagID =
        (Scope == SuggestionScope::CrossTU)
            ? diag::warn_lifetime_safety_cross_tu_param_suggestion
            : diag::warn_lifetime_safety_intra_tu_param_suggestion;
    SourceLocation InsertionPoint = Lexer::getLocForEndOfToken(
        ParmToAnnotate->getEndLoc(), 0, S.getSourceManager(), S.getLangOpts());
    StringRef FixItText = " [[clang::lifetimebound]]";
    if (!ParmToAnnotate->getIdentifier()) {
      // For unnamed parameters, placing attributes after the type would be
      // parsed as a type attribute, not a parameter attribute.
      InsertionPoint = ParmToAnnotate->getBeginLoc();
      FixItText = "[[clang::lifetimebound]] ";
    }
    S.Diag(ParmToAnnotate->getBeginLoc(), DiagID)
        << ParmToAnnotate->getSourceRange()
        << FixItHint::CreateInsertion(InsertionPoint, FixItText);

    for (const AssignmentPair &AliasStmt : AliasList)
      reportAssignment(S, ParmToAnnotate, AliasStmt.first, AliasStmt.second);

    if (const auto *EscapeExpr = Target.dyn_cast<const Expr *>())
      S.Diag(EscapeExpr->getBeginLoc(),
             diag::note_lifetime_safety_suggestion_returned_here)
          << EscapeExpr->getSourceRange();
    else if (const auto *EscapeField = Target.dyn_cast<const FieldDecl *>())
      S.Diag(EscapeField->getLocation(),
             diag::note_lifetime_safety_escapes_to_field_here)
          << EscapeField->getSourceRange();
  }

  void
  suggestLifetimeboundToImplicitThis(SuggestionScope Scope,
                                     const CXXMethodDecl *MD,
                                     llvm::ArrayRef<AssignmentPair> AliasList,
                                     const Expr *EscapeExpr) override {
    unsigned DiagID = (Scope == SuggestionScope::CrossTU)
                          ? diag::warn_lifetime_safety_cross_tu_this_suggestion
                          : diag::warn_lifetime_safety_intra_tu_this_suggestion;
    const auto MDL = MD->getTypeSourceInfo()->getTypeLoc();
    SourceLocation InsertionPoint = Lexer::getLocForEndOfToken(
        MDL.getEndLoc(), 0, S.getSourceManager(), S.getLangOpts());
    if (const auto *FPT = MD->getType()->getAs<FunctionProtoType>();
        FPT && FPT->hasTrailingReturn()) {
      // For trailing return types, 'getEndLoc()' includes the return type
      // after '->', placing the attribute in an invalid position.
      // Instead use 'getLocalRangeEnd()' which gives the '->' location
      // for trailing returns, so find the last token before it.
      const auto FTL = MDL.getAs<FunctionTypeLoc>();
      assert(FTL);
      InsertionPoint = Lexer::getLocForEndOfToken(
          Lexer::findPreviousToken(FTL.getLocalRangeEnd(), S.getSourceManager(),
                                   S.getLangOpts(),
                                   /*IncludeComments=*/false)
              ->getLocation(),
          0, S.getSourceManager(), S.getLangOpts());
    }
    S.Diag(InsertionPoint, DiagID)
        << MD->getNameInfo().getSourceRange()
        << FixItHint::CreateInsertion(InsertionPoint,
                                      " [[clang::lifetimebound]]");

    for (const AssignmentPair &AliasStmt : AliasList)
      reportAssignment(S, MD, AliasStmt.first, AliasStmt.second);

    S.Diag(EscapeExpr->getBeginLoc(),
           diag::note_lifetime_safety_suggestion_returned_here)
        << EscapeExpr->getSourceRange();
  }

  void reportNoescapeViolation(const ParmVarDecl *ParmWithNoescape,
                               const Expr *EscapeExpr) override {
    S.Diag(ParmWithNoescape->getBeginLoc(),
           diag::warn_lifetime_safety_noescape_escapes)
        << ParmWithNoescape->getSourceRange();

    S.Diag(EscapeExpr->getBeginLoc(),
           diag::note_lifetime_safety_suggestion_returned_here)
        << EscapeExpr->getSourceRange();
  }

  void reportNoescapeViolation(const ParmVarDecl *ParmWithNoescape,
                               const FieldDecl *EscapeField) override {
    S.Diag(ParmWithNoescape->getBeginLoc(),
           diag::warn_lifetime_safety_noescape_escapes)
        << ParmWithNoescape->getSourceRange();

    S.Diag(EscapeField->getLocation(),
           diag::note_lifetime_safety_escapes_to_field_here)
        << EscapeField->getEndLoc();
  }

  void reportNoescapeViolation(const ParmVarDecl *ParmWithNoescape,
                               const VarDecl *EscapeGlobal) override {
    S.Diag(ParmWithNoescape->getBeginLoc(),
           diag::warn_lifetime_safety_noescape_escapes)
        << ParmWithNoescape->getSourceRange();
    if (EscapeGlobal->isStaticLocal() || EscapeGlobal->isStaticDataMember())
      S.Diag(EscapeGlobal->getLocation(),
             diag::note_lifetime_safety_escapes_to_static_storage_here)
          << EscapeGlobal->getEndLoc();
    else
      S.Diag(EscapeGlobal->getLocation(),
             diag::note_lifetime_safety_escapes_to_global_here)
          << EscapeGlobal->getEndLoc();
  }

  void addLifetimeBoundToImplicitThis(const CXXMethodDecl *MD) override {
    S.addLifetimeBoundToImplicitThis(const_cast<CXXMethodDecl *>(MD));
  }

private:
  Sema &S;
};

} // namespace clang::lifetimes

#endif // LLVM_CLANG_LIB_SEMA_SEMALIFETIMESAFETY_H
