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
#include "clang/Basic/DiagnosticSema.h"
#include "clang/Lex/Lexer.h"
#include "clang/Sema/Sema.h"

namespace clang::lifetimes {

inline bool IsLifetimeSafetyDiagnosticEnabled(Sema &S, const Decl *D) {
  DiagnosticsEngine &Diags = S.getDiagnostics();
  constexpr unsigned DiagIDs[] = {
      diag::warn_lifetime_safety_use_after_scope,
      diag::warn_lifetime_safety_use_after_scope_moved,
      diag::warn_lifetime_safety_use_after_free,
      diag::warn_lifetime_safety_return_stack_addr,
      diag::warn_lifetime_safety_return_stack_addr_moved,
      diag::warn_lifetime_safety_invalidation,
      diag::warn_lifetime_safety_dangling_field,
      diag::warn_lifetime_safety_dangling_field_moved,
      diag::warn_lifetime_safety_dangling_global,
      diag::warn_lifetime_safety_dangling_global_moved,
      diag::warn_lifetime_safety_noescape_escapes,
      diag::warn_lifetime_safety_param_lifetimebound_violation,
  };
  for (unsigned DiagID : DiagIDs)
    if (!Diags.isIgnored(DiagID, D->getBeginLoc()))
      return true;
  return false;
}

class LifetimeSafetySemaHelperImpl : public LifetimeSafetySemaHelper {

public:
  LifetimeSafetySemaHelperImpl(Sema &S) : S(S) {}

  void reportUseAfterScope(const Expr *IssueExpr, const Expr *UseExpr,
                           const Expr *MovedExpr,
                           SourceLocation FreeLoc) override {
    S.Diag(IssueExpr->getExprLoc(),
           MovedExpr ? diag::warn_lifetime_safety_use_after_scope_moved
                     : diag::warn_lifetime_safety_use_after_scope)
        << IssueExpr->getSourceRange();
    if (MovedExpr)
      S.Diag(MovedExpr->getExprLoc(), diag::note_lifetime_safety_moved_here)
          << MovedExpr->getSourceRange();
    S.Diag(FreeLoc, diag::note_lifetime_safety_destroyed_here);
    S.Diag(UseExpr->getExprLoc(), diag::note_lifetime_safety_used_here)
        << UseExpr->getSourceRange();
  }

  void reportUseAfterReturn(const Expr *IssueExpr, const Expr *ReturnExpr,
                            const Expr *MovedExpr,
                            SourceLocation ExpiryLoc) override {
    S.Diag(IssueExpr->getExprLoc(),
           MovedExpr ? diag::warn_lifetime_safety_return_stack_addr_moved
                     : diag::warn_lifetime_safety_return_stack_addr)
        << IssueExpr->getSourceRange();
    if (MovedExpr)
      S.Diag(MovedExpr->getExprLoc(), diag::note_lifetime_safety_moved_here)
          << MovedExpr->getSourceRange();
    S.Diag(ReturnExpr->getExprLoc(), diag::note_lifetime_safety_returned_here)
        << ReturnExpr->getSourceRange();
  }

  void reportDanglingField(const Expr *IssueExpr,
                           const FieldDecl *DanglingField,
                           const Expr *MovedExpr,
                           SourceLocation ExpiryLoc) override {
    S.Diag(IssueExpr->getExprLoc(),
           MovedExpr ? diag::warn_lifetime_safety_dangling_field_moved
                     : diag::warn_lifetime_safety_dangling_field)
        << IssueExpr->getSourceRange();
    if (MovedExpr)
      S.Diag(MovedExpr->getExprLoc(), diag::note_lifetime_safety_moved_here)
          << MovedExpr->getSourceRange();
    S.Diag(DanglingField->getLocation(),
           diag::note_lifetime_safety_dangling_field_here)
        << DanglingField->getEndLoc();
  }

  void reportDanglingGlobal(const Expr *IssueExpr,
                            const VarDecl *DanglingGlobal,
                            const Expr *MovedExpr,
                            SourceLocation ExpiryLoc) override {
    S.Diag(IssueExpr->getExprLoc(),
           MovedExpr ? diag::warn_lifetime_safety_dangling_global_moved
                     : diag::warn_lifetime_safety_dangling_global)
        << IssueExpr->getSourceRange();
    if (MovedExpr)
      S.Diag(MovedExpr->getExprLoc(), diag::note_lifetime_safety_moved_here)
          << MovedExpr->getSourceRange();
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
    auto WarnDiag = isa<CXXDeleteExpr>(InvalidationExpr)
                        ? diag::warn_lifetime_safety_use_after_free
                        : diag::warn_lifetime_safety_invalidation;
    auto UseDiag = isa<CXXDeleteExpr>(InvalidationExpr)
                       ? diag::note_lifetime_safety_freed_here
                       : diag::note_lifetime_safety_invalidated_here;
    S.Diag(IssueExpr->getExprLoc(), WarnDiag)
        << false << IssueExpr->getSourceRange();
    S.Diag(InvalidationExpr->getExprLoc(), UseDiag)
        << InvalidationExpr->getSourceRange();
    S.Diag(UseExpr->getExprLoc(), diag::note_lifetime_safety_used_here)
        << UseExpr->getSourceRange();
  }
  void reportUseAfterInvalidation(const ParmVarDecl *PVD, const Expr *UseExpr,
                                  const Expr *InvalidationExpr) override {

    auto WarnDiag = isa<CXXDeleteExpr>(InvalidationExpr)
                        ? diag::warn_lifetime_safety_use_after_free
                        : diag::warn_lifetime_safety_invalidation;
    auto UseDiag = isa<CXXDeleteExpr>(InvalidationExpr)
                       ? diag::note_lifetime_safety_freed_here
                       : diag::note_lifetime_safety_invalidated_here;

    S.Diag(PVD->getSourceRange().getBegin(), WarnDiag)
        << true << PVD->getSourceRange();
    S.Diag(InvalidationExpr->getExprLoc(), UseDiag)
        << InvalidationExpr->getSourceRange();
    S.Diag(UseExpr->getExprLoc(), diag::note_lifetime_safety_used_here)
        << UseExpr->getSourceRange();
  }

  void suggestLifetimeboundToParmVar(SuggestionScope Scope,
                                     const ParmVarDecl *ParmToAnnotate,
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

    if (const auto *EscapeExpr = Target.dyn_cast<const Expr *>())
      S.Diag(EscapeExpr->getBeginLoc(),
             diag::note_lifetime_safety_suggestion_returned_here)
          << EscapeExpr->getSourceRange();
    else if (const auto *EscapeField = Target.dyn_cast<const FieldDecl *>())
      S.Diag(EscapeField->getLocation(),
             diag::note_lifetime_safety_escapes_to_field_here)
          << EscapeField->getSourceRange();
  }

  void reportLifetimeboundViolation(
      const ParmVarDecl *ParmWithLifetimebound) override {
    const auto *Attr = ParmWithLifetimebound->getAttr<LifetimeBoundAttr>();
    StringRef ParamName = ParmWithLifetimebound->getName();
    bool HasName = ParamName.size() > 0;
    S.Diag(Attr->getLocation(),
           diag::warn_lifetime_safety_param_lifetimebound_violation)
        << HasName << ParamName << Attr->getRange();
  }

  void suggestLifetimeboundToImplicitThis(SuggestionScope Scope,
                                          const CXXMethodDecl *MD,
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
