//===--- WalkAST.cpp - Find declaration references in the AST -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AnalysisInternal.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/Expr.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/AST/TemplateName.h"
#include "clang/AST/Type.h"
#include "clang/AST/TypeLoc.h"
#include "clang/Basic/SourceLocation.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Casting.h"

namespace clang::include_cleaner {
namespace {
using DeclCallback = llvm::function_ref<void(SourceLocation, NamedDecl &)>;

class ASTWalker : public RecursiveASTVisitor<ASTWalker> {
  DeclCallback Callback;

  bool handleTemplateName(SourceLocation Loc, TemplateName TN) {
    // For using-templates, only mark the alias.
    if (auto *USD = TN.getAsUsingShadowDecl()) {
      report(Loc, USD);
      return true;
    }
    report(Loc, TN.getAsTemplateDecl());
    return true;
  }

  void report(SourceLocation Loc, NamedDecl *ND) {
    if (!ND || Loc.isInvalid())
      return;
    Callback(Loc, *cast<NamedDecl>(ND->getCanonicalDecl()));
  }

public:
  ASTWalker(DeclCallback Callback) : Callback(Callback) {}

  bool VisitDeclRefExpr(DeclRefExpr *DRE) {
    report(DRE->getLocation(), DRE->getFoundDecl());
    return true;
  }

  bool VisitMemberExpr(MemberExpr *E) {
    report(E->getMemberLoc(), E->getFoundDecl().getDecl());
    return true;
  }

  bool VisitCXXConstructExpr(CXXConstructExpr *E) {
    report(E->getLocation(), E->getConstructor());
    return true;
  }

  bool VisitOverloadExpr(OverloadExpr *E) {
    // Since we can't prove which overloads are used, report all of them.
    // FIXME: Provide caller with the ability to make a decision for such uses.
    llvm::for_each(E->decls(),
                   [this, E](NamedDecl *D) { report(E->getNameLoc(), D); });
    return true;
  }

  bool VisitUsingDecl(UsingDecl *UD) {
    // FIXME: Provide caller with the ability to tell apart used/non-used
    // targets.
    for (const auto *Shadow : UD->shadows())
      report(UD->getLocation(), Shadow->getTargetDecl());
    return true;
  }

  bool VisitFunctionDecl(FunctionDecl *FD) {
    // Mark declaration from definition as it needs type-checking.
    if (FD->isThisDeclarationADefinition())
      report(FD->getLocation(), FD);
    return true;
  }

  bool VisitEnumDecl(EnumDecl *D) {
    // Definition of an enum with an underlying type references declaration for
    // type-checking purposes.
    if (D->isThisDeclarationADefinition() && D->getIntegerTypeSourceInfo())
      report(D->getLocation(), D);
    return true;
  }

  // TypeLoc visitors.
  bool VisitUsingTypeLoc(UsingTypeLoc TL) {
    report(TL.getNameLoc(), TL.getFoundDecl());
    return true;
  }

  bool VisitTagTypeLoc(TagTypeLoc TTL) {
    report(TTL.getNameLoc(), TTL.getDecl());
    return true;
  }

  bool VisitTypedefTypeLoc(TypedefTypeLoc TTL) {
    report(TTL.getNameLoc(), TTL.getTypedefNameDecl());
    return true;
  }

  bool VisitTemplateSpecializationTypeLoc(TemplateSpecializationTypeLoc TL) {
    // FIXME: Handle explicit specializations.
    return handleTemplateName(TL.getTemplateNameLoc(),
                              TL.getTypePtr()->getTemplateName());
  }

  bool VisitDeducedTemplateSpecializationTypeLoc(
      DeducedTemplateSpecializationTypeLoc TL) {
    // FIXME: Handle specializations.
    return handleTemplateName(TL.getTemplateNameLoc(),
                              TL.getTypePtr()->getTemplateName());
  }

  bool TraverseTemplateArgumentLoc(const TemplateArgumentLoc &TL) {
    auto &Arg = TL.getArgument();
    // Template-template parameters require special attention, as there's no
    // TemplateNameLoc.
    if (Arg.getKind() == TemplateArgument::Template ||
        Arg.getKind() == TemplateArgument::TemplateExpansion)
      return handleTemplateName(TL.getLocation(),
                                Arg.getAsTemplateOrTemplatePattern());
    return RecursiveASTVisitor::TraverseTemplateArgumentLoc(TL);
  }
};

} // namespace

void walkAST(Decl &Root, DeclCallback Callback) {
  ASTWalker(Callback).TraverseDecl(&Root);
}

} // namespace clang::include_cleaner
