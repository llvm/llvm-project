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

#include "clang/Analysis/Analyses/LifetimeSafety/LifetimeAnnotations.h"
#include "clang/Analysis/Analyses/LifetimeSafety/LifetimeSafety.h"
#include "clang/Basic/DiagnosticSema.h"
#include "clang/Lex/Lexer.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Sema/Sema.h"
#include <string>

namespace clang::lifetimes {

inline bool IsLifetimeSafetyEnabled(Sema &S, const Decl *D) {
  // TODO: Enable ObjectiveC later when we know it's stable enough.
  if (S.getLangOpts().ObjC)
    return false;

  // TODO: Default this flag to on in the future.
  if (!S.getLangOpts().CPlusPlus && !S.getLangOpts().EnableLifetimeSafetyInC)
    return false;

  // Translation-unit mode: whole-program analysis runs once on TU.
  // Individual function analysis is disabled when TU mode is enabled.
  if (S.getLangOpts().EnableLifetimeSafetyTUAnalysis)
    return isa<TranslationUnitDecl>(D);

  // Per-function mode: analysis runs on each function/method individually.
  // Skip TU-level calls when per-function mode is enabled.
  if (isa<TranslationUnitDecl>(D))
    return false;

  // Enable per-function mode via debug flag or specific diagnostics.
  if (S.getLangOpts().DebugRunLifetimeSafety)
    return true;
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
      diag::warn_lifetime_safety_lifetimebound_violation,
      diag::warn_lifetime_safety_cross_tu_misplaced_lifetimebound,
      diag::warn_lifetime_safety_intra_tu_misplaced_lifetimebound,
      diag::warn_lifetime_safety_invalidated_field,
      diag::warn_lifetime_safety_invalidated_global,
      diag::warn_lifetime_safety_cross_tu_param_suggestion,
      diag::warn_lifetime_safety_intra_tu_param_suggestion,
      diag::warn_lifetime_safety_cross_tu_ctor_param_suggestion,
      diag::warn_lifetime_safety_intra_tu_ctor_param_suggestion,
      diag::warn_lifetime_safety_cross_tu_this_suggestion,
      diag::warn_lifetime_safety_intra_tu_this_suggestion,
      diag::warn_lifetime_safety_inapplicable_lifetimebound};
  for (unsigned DiagID : DiagIDs)
    if (!Diags.isIgnored(DiagID, D->getBeginLoc()))
      return true;
  return false;
}

inline bool ShouldSuggestLifetimeAnnotations(Sema &S, const Decl *D) {
  DiagnosticsEngine &Diags = S.getDiagnostics();
  constexpr unsigned DiagIDs[] = {
      diag::warn_lifetime_safety_intra_tu_param_suggestion,
      diag::warn_lifetime_safety_cross_tu_param_suggestion,
      diag::warn_lifetime_safety_intra_tu_ctor_param_suggestion,
      diag::warn_lifetime_safety_cross_tu_ctor_param_suggestion,
      diag::warn_lifetime_safety_intra_tu_this_suggestion,
      diag::warn_lifetime_safety_cross_tu_this_suggestion};
  for (unsigned DiagID : DiagIDs)
    if (!Diags.isIgnored(DiagID, D->getBeginLoc()))
      return true;
  return false;
}

inline LifetimeSafetyOpts GetLifetimeSafetyOpts(Sema &S, const Decl *D) {
  LifetimeSafetyOpts LSOpts;
  LSOpts.MaxCFGBlocks = S.getLangOpts().LifetimeSafetyMaxCFGBlocks;
  LSOpts.SuggestAnnotations = ShouldSuggestLifetimeAnnotations(S, D);
  return LSOpts;
}

class LifetimeSafetySemaHelperImpl : public LifetimeSafetySemaHelper {

public:
  LifetimeSafetySemaHelperImpl(Sema &S) : S(S) {}

  void reportUseAfterScope(const Expr *IssueExpr, const Expr *UseExpr,
                           const Expr *MovedExpr, SourceLocation FreeLoc,
                           llvm::ArrayRef<const Expr *> ExprChain) override {
    unsigned DiagID = MovedExpr
                          ? diag::warn_lifetime_safety_use_after_scope_moved
                          : diag::warn_lifetime_safety_use_after_scope;
    std::string DestroyedSubject = getDiagSubjectDescription(IssueExpr);

    S.Diag(IssueExpr->getExprLoc(), DiagID)
        << DestroyedSubject << IssueExpr->getSourceRange();
    if (MovedExpr)
      S.Diag(MovedExpr->getExprLoc(), diag::note_lifetime_safety_moved_here)
          << MovedExpr->getSourceRange();
    S.Diag(FreeLoc, diag::note_lifetime_safety_destroyed_here)
        << DestroyedSubject;

    reportAliasingChain(ExprChain);

    S.Diag(UseExpr->getExprLoc(), diag::note_lifetime_safety_used_here)
        << UseExpr->getSourceRange();
  }

  void reportUseAfterReturn(const Expr *IssueExpr, const Expr *ReturnExpr,
                            const Expr *MovedExpr) override {
    unsigned DiagID = MovedExpr
                          ? diag::warn_lifetime_safety_return_stack_addr_moved
                          : diag::warn_lifetime_safety_return_stack_addr;

    S.Diag(IssueExpr->getExprLoc(), DiagID)
        << getDiagSubjectDescription(IssueExpr) << IssueExpr->getSourceRange();

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
    unsigned DiagID = MovedExpr
                          ? diag::warn_lifetime_safety_dangling_field_moved
                          : diag::warn_lifetime_safety_dangling_field;

    S.Diag(IssueExpr->getExprLoc(), DiagID)
        << getDiagSubjectDescription(IssueExpr)
        << getDiagSubjectDescription(DanglingField)
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
    unsigned DiagID = MovedExpr
                          ? diag::warn_lifetime_safety_dangling_global_moved
                          : diag::warn_lifetime_safety_dangling_global;

    S.Diag(IssueExpr->getExprLoc(), DiagID)
        << getDiagSubjectDescription(IssueExpr)
        << getDiagSubjectDescription(DanglingGlobal)
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
    std::string InvalidatedSubject = getDiagSubjectDescription(IssueExpr);
    S.Diag(IssueExpr->getExprLoc(), WarnDiag)
        << InvalidatedSubject << IssueExpr->getSourceRange();
    reportInvalidationSite(InvalidationExpr, InvalidatedSubject);
    S.Diag(UseExpr->getExprLoc(), diag::note_lifetime_safety_used_here)
        << UseExpr->getSourceRange();
  }
  void reportUseAfterInvalidation(const ParmVarDecl *PVD, const Expr *UseExpr,
                                  const Expr *InvalidationExpr) override {

    auto WarnDiag = isa<CXXDeleteExpr>(InvalidationExpr)
                        ? diag::warn_lifetime_safety_use_after_free
                        : diag::warn_lifetime_safety_invalidation;
    std::string InvalidatedSubject = getDiagSubjectDescription(PVD);

    S.Diag(PVD->getSourceRange().getBegin(), WarnDiag)
        << InvalidatedSubject << PVD->getSourceRange();
    reportInvalidationSite(InvalidationExpr, InvalidatedSubject);
    S.Diag(UseExpr->getExprLoc(), diag::note_lifetime_safety_used_here)
        << UseExpr->getSourceRange();
  }

  void reportInvalidatedField(const Expr *IssueExpr,
                              const FieldDecl *DanglingField,
                              const Expr *InvalidationExpr) override {
    std::string InvalidatedSubject = getDiagSubjectDescription(IssueExpr);
    S.Diag(IssueExpr->getExprLoc(),
           diag::warn_lifetime_safety_invalidated_field)
        << InvalidatedSubject << getDiagSubjectDescription(DanglingField)
        << IssueExpr->getSourceRange();
    reportInvalidationSite(InvalidationExpr, InvalidatedSubject);
    S.Diag(DanglingField->getLocation(),
           diag::note_lifetime_safety_dangling_field_here)
        << DanglingField->getEndLoc();
  }

  void reportInvalidatedField(const ParmVarDecl *PVD,
                              const FieldDecl *DanglingField,
                              const Expr *InvalidationExpr) override {
    std::string InvalidatedSubject = getDiagSubjectDescription(PVD);
    S.Diag(PVD->getSourceRange().getBegin(),
           diag::warn_lifetime_safety_invalidated_field)
        << InvalidatedSubject << getDiagSubjectDescription(DanglingField)
        << PVD->getSourceRange();
    reportInvalidationSite(InvalidationExpr, InvalidatedSubject);
    S.Diag(DanglingField->getLocation(),
           diag::note_lifetime_safety_dangling_field_here)
        << DanglingField->getEndLoc();
  }

  void reportInvalidatedGlobal(const Expr *IssueExpr,
                               const VarDecl *DanglingGlobal,
                               const Expr *InvalidationExpr) override {
    std::string InvalidatedSubject = getDiagSubjectDescription(IssueExpr);
    S.Diag(IssueExpr->getExprLoc(),
           diag::warn_lifetime_safety_invalidated_global)
        << InvalidatedSubject << getDiagSubjectDescription(DanglingGlobal)
        << IssueExpr->getSourceRange();
    reportInvalidationSite(InvalidationExpr, InvalidatedSubject);
    if (DanglingGlobal->isStaticLocal() || DanglingGlobal->isStaticDataMember())
      S.Diag(DanglingGlobal->getLocation(),
             diag::note_lifetime_safety_dangling_static_here)
          << DanglingGlobal->getEndLoc();
    else
      S.Diag(DanglingGlobal->getLocation(),
             diag::note_lifetime_safety_dangling_global_here)
          << DanglingGlobal->getEndLoc();
  }

  void reportInvalidatedGlobal(const ParmVarDecl *PVD,
                               const VarDecl *DanglingGlobal,
                               const Expr *InvalidationExpr) override {
    std::string InvalidatedSubject = getDiagSubjectDescription(PVD);
    S.Diag(PVD->getSourceRange().getBegin(),
           diag::warn_lifetime_safety_invalidated_global)
        << InvalidatedSubject << getDiagSubjectDescription(DanglingGlobal)
        << PVD->getSourceRange();
    reportInvalidationSite(InvalidationExpr, InvalidatedSubject);
    if (DanglingGlobal->isStaticLocal() || DanglingGlobal->isStaticDataMember())
      S.Diag(DanglingGlobal->getLocation(),
             diag::note_lifetime_safety_dangling_static_here)
          << DanglingGlobal->getEndLoc();
    else
      S.Diag(DanglingGlobal->getLocation(),
             diag::note_lifetime_safety_dangling_global_here)
          << DanglingGlobal->getEndLoc();
  }

  void suggestLifetimeboundToParmVar(WarningScope Scope,
                                     const ParmVarDecl *ParmToAnnotate,
                                     EscapingTarget Target) override {
    unsigned DiagID;
    if (isa<CXXConstructorDecl>(ParmToAnnotate->getDeclContext()))
      DiagID = (Scope == WarningScope::CrossTU)
                   ? diag::warn_lifetime_safety_cross_tu_ctor_param_suggestion
                   : diag::warn_lifetime_safety_intra_tu_ctor_param_suggestion;
    else
      DiagID = (Scope == WarningScope::CrossTU)
                   ? diag::warn_lifetime_safety_cross_tu_param_suggestion
                   : diag::warn_lifetime_safety_intra_tu_param_suggestion;

    auto [InsertionPoint, FixItText] = getLifetimeBoundFixIt(ParmToAnnotate);

    S.Diag(InsertionPoint, DiagID)
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
           diag::warn_lifetime_safety_lifetimebound_violation)
        << HasName << ParamName << Attr->getRange();
  }

  void reportLifetimeboundViolation(
      const CXXMethodDecl *MDWithLifetimebound) override {
    const auto *Attr =
        getImplicitObjectParamLifetimeBoundAttr(MDWithLifetimebound);
    assert(Attr && "Expected lifetimebound attribute");
    S.Diag(Attr->getLocation(),
           diag::warn_lifetime_safety_lifetimebound_violation)
        << 2 << "" << Attr->getRange();
  }

  void reportMisplacedLifetimebound(WarningScope Scope,
                                    const CXXMethodDecl *FDef,
                                    const CXXMethodDecl *FDecl) override {
    const auto *Attr = getDirectImplicitObjectLifetimeBoundAttr(FDef);
    assert(Attr && "Expected lifetimebound attribute");
    unsigned DiagID =
        Scope == WarningScope::CrossTU
            ? diag::warn_lifetime_safety_cross_tu_misplaced_lifetimebound
            : diag::warn_lifetime_safety_intra_tu_misplaced_lifetimebound;

    auto [InsertionPoint, FixItText] = getLifetimeBoundFixIt(FDecl);

    // Do not emit fix-its in macros or at invalid locations.
    bool IsMacro =
        FDecl->getBeginLoc().isMacroID() || InsertionPoint.isMacroID();

    if (IsMacro || InsertionPoint.isInvalid())
      S.Diag(FDecl->getLocation(), DiagID);
    else
      S.Diag(InsertionPoint, DiagID)
          << FixItHint::CreateInsertion(InsertionPoint, FixItText);

    S.Diag(Attr->getLocation(), diag::note_lifetime_safety_lifetimebound_here)
        << Attr->getRange();
  }

  void reportMisplacedLifetimebound(WarningScope Scope,
                                    const ParmVarDecl *PVDDef,
                                    const ParmVarDecl *PVDDecl) override {

    const auto *Attr = PVDDef->getAttr<LifetimeBoundAttr>();
    assert(Attr && "Expected lifetimebound attribute");
    unsigned DiagID =
        Scope == WarningScope::CrossTU
            ? diag::warn_lifetime_safety_cross_tu_misplaced_lifetimebound
            : diag::warn_lifetime_safety_intra_tu_misplaced_lifetimebound;

    auto [InsertionPoint, FixItText] = getLifetimeBoundFixIt(PVDDecl);

    // Do not emit fix-its in macros or at invalid locations.
    bool IsMacro =
        PVDDecl->getBeginLoc().isMacroID() || InsertionPoint.isMacroID();

    if (IsMacro || InsertionPoint.isInvalid())
      S.Diag(PVDDecl->getBeginLoc(), DiagID) << PVDDecl->getSourceRange();
    else
      S.Diag(InsertionPoint, DiagID)
          << PVDDecl->getSourceRange()
          << FixItHint::CreateInsertion(InsertionPoint, FixItText);

    S.Diag(Attr->getLocation(), diag::note_lifetime_safety_lifetimebound_here)
        << Attr->getRange();
  }

  void reportInapplicableLifetimebound(const ParmVarDecl *PVD) override {
    assert(PVD->hasAttr<LifetimeBoundAttr>() &&
           "Expected parameter to have lifetimebound attribute");
    const auto *Attr = PVD->getAttr<LifetimeBoundAttr>();
    S.Diag(Attr->getLocation(),
           diag::warn_lifetime_safety_inapplicable_lifetimebound)
        << PVD->getType() << Attr->getRange();
  }

  void suggestLifetimeboundToImplicitThis(WarningScope Scope,
                                          const CXXMethodDecl *MD,
                                          const Expr *EscapeExpr) override {
    unsigned DiagID = (Scope == WarningScope::CrossTU)
                          ? diag::warn_lifetime_safety_cross_tu_this_suggestion
                          : diag::warn_lifetime_safety_intra_tu_this_suggestion;

    auto [InsertionPoint, FixItText] = getLifetimeBoundFixIt(MD);

    S.Diag(InsertionPoint, DiagID)
        << MD->getNameInfo().getSourceRange()
        << FixItHint::CreateInsertion(InsertionPoint, FixItText);

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
  struct LifetimeBoundMacroCache {
    bool IsBuilt = false;
    SmallVector<const IdentifierInfo *> Candidates;
  };

  void buildLifetimeBoundMacroCache(LifetimeBoundMacroCache &Cache,
                                    ArrayRef<TokenValue> Tokens) {
    if (Cache.IsBuilt)
      return;

    const Preprocessor &PP = S.getPreprocessor();
    // Collect macro names that were ever defined as a lifetimebound attribute.
    for (const auto &M : PP.macros()) {
      const IdentifierInfo *II = M.first;
      const MacroDirective *MD = PP.getLocalMacroDirectiveHistory(II);
      if (!MD)
        continue;

      // Include earlier matching definitions to handle redefinitions.
      for (MacroDirective::DefInfo Def = MD->getDefinition(); Def;
           Def = Def.getPreviousDefinition()) {
        const MacroInfo *MI = Def.getMacroInfo();
        if (MI->isObjectLike() && Tokens.size() == MI->getNumTokens() &&
            std::equal(Tokens.begin(), Tokens.end(), MI->tokens_begin())) {
          Cache.Candidates.push_back(II);
          break;
        }
      }
    }
    Cache.IsBuilt = true;
  }

  StringRef getLastCachedMacroWithSpelling(SourceLocation Loc,
                                           llvm::ArrayRef<TokenValue> Tokens,
                                           LifetimeBoundMacroCache &Cache) {
    if (Loc.isInvalid())
      return {};

    buildLifetimeBoundMacroCache(Cache, Tokens);

    const Preprocessor &PP = S.getPreprocessor();
    const SourceManager &SM = S.getSourceManager();
    SourceLocation BestLocation;
    StringRef BestSpelling;
    for (const IdentifierInfo *II : Cache.Candidates) {
      const MacroDirective *MD = PP.getLocalMacroDirectiveHistory(II);
      const MacroDirective::DefInfo Def = MD->findDirectiveAtLoc(Loc, SM);
      if (!Def || !Def.getMacroInfo())
        continue;

      // Ensure the macro definition active at Loc still has this spelling.
      const MacroInfo *MI = Def.getMacroInfo();
      if (!MI->isObjectLike() || Tokens.size() != MI->getNumTokens() ||
          !std::equal(Tokens.begin(), Tokens.end(), MI->tokens_begin()))
        continue;

      // Choose the matching macro defined latest before Loc.
      SourceLocation Location = Def.getLocation();
      assert(Location.isInvalid() ||
             SM.isBeforeInTranslationUnit(Location, Loc));
      if (BestLocation.isInvalid() ||
          (Location.isValid() &&
           SM.isBeforeInTranslationUnit(BestLocation, Location))) {
        BestLocation = Location;
        BestSpelling = II->getName();
      }
    }
    return BestSpelling;
  }

  void reportInvalidationSite(const Expr *InvalidationExpr,
                              StringRef InvalidatedSubject) {
    auto Diag = isa<CXXDeleteExpr>(InvalidationExpr)
                    ? diag::note_lifetime_safety_freed_here
                    : diag::note_lifetime_safety_invalidated_here;
    S.Diag(InvalidationExpr->getExprLoc(), Diag)
        << InvalidatedSubject << InvalidationExpr->getSourceRange();
  }

  std::string getLifetimeBoundFixItText(SourceLocation Loc, bool LeadingSpace,
                                        bool AllowGNUAttrMacro = true) {
    StringRef Spelling = S.getLangOpts().LifetimeSafetyLifetimeBoundMacro;
    if (Spelling.empty() && Loc.isValid()) {
      const Preprocessor &PP = S.getPreprocessor();
      Spelling = getLastCachedMacroWithSpelling(
          Loc,
          {tok::l_square, tok::l_square, PP.getIdentifierInfo("clang"),
           tok::coloncolon, PP.getIdentifierInfo("lifetimebound"),
           tok::r_square, tok::r_square},
          ClangLifetimeBoundMacroCache);

      if (Spelling.empty() && AllowGNUAttrMacro)
        Spelling = getLastCachedMacroWithSpelling(
            Loc,
            {tok::kw___attribute, tok::l_paren, tok::l_paren,
             PP.getIdentifierInfo("lifetimebound"), tok::r_paren, tok::r_paren},
            GNULifetimeBoundMacroCache);
    }
    const std::string Text =
        Spelling.empty() ? "[[clang::lifetimebound]]" : Spelling.str();
    return LeadingSpace ? " " + Text : Text + " ";
  }

  std::pair<SourceLocation, std::string>
  getLifetimeBoundFixIt(const ParmVarDecl *Decl) {
    SourceLocation InsertionPoint = Lexer::getLocForEndOfToken(
        Decl->getEndLoc(), 0, S.getSourceManager(), S.getLangOpts());
    bool LeadingSpace = true;

    if (!Decl->getIdentifier()) {
      // For unnamed parameters, placing attributes after the type would be
      // parsed as a type attribute, not a parameter attribute.
      InsertionPoint = Decl->getBeginLoc();
      LeadingSpace = false;
    } else if (Decl->hasDefaultArg()) {
      // If the parameter has a default argument, place the attribute after the
      // named argument.
      InsertionPoint = Lexer::getLocForEndOfToken(
          Decl->getLocation(), 0, S.getSourceManager(), S.getLangOpts());
    }
    return {InsertionPoint,
            getLifetimeBoundFixItText(InsertionPoint, LeadingSpace)};
  }

  std::pair<SourceLocation, std::string>
  getLifetimeBoundFixIt(const CXXMethodDecl *MD) {
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
    return {InsertionPoint,
            getLifetimeBoundFixItText(InsertionPoint, /*LeadingSpace=*/true,
                                      /*AllowGNUAttrMacro=*/false)};
  }

  std::string getDiagSubjectDescription(const ValueDecl *VD) {
    std::string Res;
    llvm::raw_string_ostream OS(Res);
    if (isa<FieldDecl>(VD)) {
      OS << "field";
    } else if (isa<ParmVarDecl>(VD)) {
      OS << "parameter";
    } else if (const auto *Var = dyn_cast<VarDecl>(VD)) {
      if (Var->isStaticLocal() || Var->isStaticDataMember())
        OS << "static variable";
      else if (Var->hasGlobalStorage())
        OS << "global variable";
      else
        OS << "local variable";
    } else {
      OS << "variable";
    }
    OS << " '";
    VD->getNameForDiagnostic(OS, S.getPrintingPolicy(), /*Qualified=*/false);
    OS << "'";
    return Res;
  }

  std::string getDiagSubjectDescription(const Expr *E) {
    E = E->IgnoreImpCasts();
    if (isa<MaterializeTemporaryExpr>(E))
      return "temporary object";
    if (isa<CXXNewExpr>(E))
      return "allocated object";
    if (const auto *DRE = dyn_cast<DeclRefExpr>(E))
      return getDiagSubjectDescription(DRE->getDecl());

    if (const auto *CE = dyn_cast<CallExpr>(E)) {
      const auto *FD = CE->getDirectCallee();
      if (!FD)
        return "result of call";
      if (FD->isOverloadedOperator() || isa<CXXConversionDecl>(FD))
        return "expression";
      std::string Name;
      llvm::raw_string_ostream OS(Name);
      FD->getNameForDiagnostic(OS, S.getPrintingPolicy(),
                               /*Qualified=*/false);
      return "result of call to '" + Name + "'";
    }

    // TODO: Handle other expression types.
    return "expression";
  }

  bool shouldShowInAliasChain(const Expr *CurrExpr, const Expr *LastExpr) {
    CurrExpr = CurrExpr->IgnoreImpCasts();
    LastExpr = LastExpr->IgnoreImpCasts();

    if (!isa<CallExpr, DeclRefExpr>(CurrExpr))
      return false;
    // Source ranges can be used to filter out many implicit expressions,
    // because operations between class objects often involve numerous implicit
    // conversions, yet they share the same source range.
    return CurrExpr->getSourceRange() != LastExpr->getSourceRange();
  }

  void reportAliasingChain(llvm::ArrayRef<const Expr *> OriginExprChain) {
    if (OriginExprChain.empty())
      return;

    const Expr *LastExpr = OriginExprChain.back();
    std::string IssueStr = getDiagSubjectDescription(LastExpr);

    for (const Expr *CurrExpr : reverse(OriginExprChain.drop_back())) {
      if (!shouldShowInAliasChain(CurrExpr, LastExpr))
        continue;
      S.Diag(CurrExpr->getBeginLoc(),
             diag::note_lifetime_safety_aliases_storage)
          << CurrExpr->getSourceRange() << getDiagSubjectDescription(CurrExpr)
          << IssueStr;
      LastExpr = CurrExpr;
    }
  }

  LifetimeBoundMacroCache ClangLifetimeBoundMacroCache;
  LifetimeBoundMacroCache GNULifetimeBoundMacroCache;
  Sema &S;
};

} // namespace clang::lifetimes

#endif // LLVM_CLANG_LIB_SEMA_SEMALIFETIMESAFETY_H
