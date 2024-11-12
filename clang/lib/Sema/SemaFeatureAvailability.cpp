//===--- SemaFeatureAvailability.cpp - Availability attribute handling ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  This file processes the feature availability attribute.
//
//===----------------------------------------------------------------------===//

#include "clang/AST/Attr.h"
#include "clang/AST/Decl.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Sema/DelayedDiagnostic.h"
#include "clang/Sema/ScopeInfo.h"
#include "clang/Sema/Sema.h"
#include "llvm/ADT/SmallSet.h"
#include <utility>

using namespace clang;
using namespace sema;

static bool isFeatureUseGuarded(const DomainAvailabilityAttr *AA,
                                const Decl *ContextDecl, ASTContext &Ctx) {
  for (auto *Attr : ContextDecl->specific_attrs<DomainAvailabilityAttr>())
    if (AA->getDomain() == Attr->getDomain())
      return AA->getUnavailable() == Attr->getUnavailable();
  return false;
}

static void diagnoseDeclFeatureAvailability(const NamedDecl *D,
                                            SourceLocation Loc,
                                            Decl *ContextDecl, Sema &S) {
  for (auto *Attr : D->specific_attrs<DomainAvailabilityAttr>())
    if (!isFeatureUseGuarded(Attr, ContextDecl, S.Context))
      S.Diag(Loc, diag::err_unguarded_feature)
          << D << Attr->getDomain().str() << Attr->getUnavailable();
}

class DiagnoseUnguardedFeatureAvailability
    : public RecursiveASTVisitor<DiagnoseUnguardedFeatureAvailability> {

  typedef RecursiveASTVisitor<DiagnoseUnguardedFeatureAvailability> Base;

  Sema &SemaRef;
  const Decl *D;

  struct FeatureAvailInfo {
    StringRef Domain;
    bool Unavailable;
  };

  SmallVector<FeatureAvailInfo, 4> FeatureStack;

  bool isFeatureUseGuarded(const DomainAvailabilityAttr *Attr) const;

  bool isConditionallyGuardedByFeature() const;

public:
  DiagnoseUnguardedFeatureAvailability(Sema &SemaRef, const Decl *D,
                                       Decl *Ctx = nullptr)
      : SemaRef(SemaRef), D(D) {}

  void diagnoseDeclFeatureAvailability(const NamedDecl *D, SourceLocation Loc);

  bool TraverseIfStmt(IfStmt *If);

  bool VisitDeclRefExpr(DeclRefExpr *DRE) {
    diagnoseDeclFeatureAvailability(DRE->getDecl(), DRE->getBeginLoc());
    return true;
  }

  bool VisitMemberExpr(MemberExpr *ME) {
    diagnoseDeclFeatureAvailability(ME->getMemberDecl(), ME->getBeginLoc());
    return true;
  }

  bool VisitObjCMessageExpr(ObjCMessageExpr *OME) {
    diagnoseDeclFeatureAvailability(OME->getMethodDecl(), OME->getBeginLoc());
    return true;
  }

  bool VisitLabelStmt(LabelStmt *LS) {
    if (isConditionallyGuardedByFeature())
      SemaRef.Diag(LS->getBeginLoc(),
                   diag::err_label_in_conditionally_guarded_feature);
    return true;
  }

  bool VisitTypeLoc(TypeLoc Ty);

  void IssueDiagnostics() {
    if (auto *FD = dyn_cast<FunctionDecl>(D))
      TraverseStmt(FD->getBody());
    else if (auto *OMD = dyn_cast<ObjCMethodDecl>(D))
      TraverseStmt(OMD->getBody());
  }
};

static std::pair<StringRef, bool> extractFeatureExpr(const Expr *IfCond) {
  const auto *E = IfCond;
  bool IsNegated = false;
  while (true) {
    E = E->IgnoreParens();
    if (const auto *AE = dyn_cast<ObjCAvailabilityCheckExpr>(E)) {
      if (!AE->hasDomainName())
        return {};
      return {AE->getDomainName(), IsNegated};
    }

    const auto *UO = dyn_cast<UnaryOperator>(E);
    if (!UO || UO->getOpcode() != UO_LNot) {
      return {};
    }
    E = UO->getSubExpr();
    IsNegated = !IsNegated;
  }
}

bool DiagnoseUnguardedFeatureAvailability::isConditionallyGuardedByFeature()
    const {
  return FeatureStack.size();
}

bool DiagnoseUnguardedFeatureAvailability::TraverseIfStmt(IfStmt *If) {
  std::pair<StringRef, bool> IfCond;
  if (auto *Cond = If->getCond())
    IfCond = extractFeatureExpr(Cond);
  if (IfCond.first.empty()) {
    // This isn't an availability checking 'if', we can just continue.
    return Base::TraverseIfStmt(If);
  }

  StringRef FeatureStr = IfCond.first;
  auto *Guarded = If->getThen();
  auto *Unguarded = If->getElse();
  if (IfCond.second) {
    std::swap(Guarded, Unguarded);
  }

  FeatureStack.push_back({FeatureStr, false});
  bool ShouldContinue = TraverseStmt(Guarded);
  FeatureStack.pop_back();

  if (!ShouldContinue)
    return false;

  FeatureStack.push_back({FeatureStr, true});
  ShouldContinue = TraverseStmt(Unguarded);
  FeatureStack.pop_back();
  return ShouldContinue;
}

bool DiagnoseUnguardedFeatureAvailability::isFeatureUseGuarded(
    const DomainAvailabilityAttr *Attr) const {
  auto Domain = Attr->getDomain();
  for (auto &Info : FeatureStack)
    if (Info.Domain == Domain && Info.Unavailable == Attr->getUnavailable())
      return true;
  return ::isFeatureUseGuarded(Attr, D, SemaRef.Context);
}

void DiagnoseUnguardedFeatureAvailability::diagnoseDeclFeatureAvailability(
    const NamedDecl *D, SourceLocation Loc) {
  for (auto *Attr : D->specific_attrs<DomainAvailabilityAttr>()) {
    std::string FeatureUse = Attr->getDomain().str();
    if (!isFeatureUseGuarded(Attr))
      SemaRef.Diag(Loc, diag::err_unguarded_feature)
          << D << FeatureUse << Attr->getUnavailable();
  }
}

bool DiagnoseUnguardedFeatureAvailability::VisitTypeLoc(TypeLoc Ty) {
  const Type *TyPtr = Ty.getTypePtr();
  SourceLocation Loc = Ty.getBeginLoc();

  if (Loc.isInvalid())
    return true;

  if (const auto *TT = dyn_cast<TagType>(TyPtr)) {
    TagDecl *TD = TT->getDecl();
    diagnoseDeclFeatureAvailability(TD, Ty.getBeginLoc());
  } else if (const auto *TD = dyn_cast<TypedefType>(TyPtr)) {
    TypedefNameDecl *D = TD->getDecl();
    diagnoseDeclFeatureAvailability(D, Ty.getBeginLoc());
  } else if (const auto *ObjCO = dyn_cast<ObjCObjectType>(TyPtr)) {
    if (NamedDecl *D = ObjCO->getInterface())
      diagnoseDeclFeatureAvailability(D, Ty.getBeginLoc());
  }

  return true;
}

void Sema::handleDelayedFeatureAvailabilityCheck(DelayedDiagnostic &DD,
                                                 Decl *Ctx) {
  assert(DD.Kind == DelayedDiagnostic::FeatureAvailability &&
         "Expected a feature availability diagnostic here");

  DD.Triggered = true;
  diagnoseDeclFeatureAvailability(DD.getFeatureAvailabilityDecl(), DD.Loc, Ctx,
                                  *this);
}

void Sema::DiagnoseUnguardedFeatureAvailabilityViolations(Decl *D) {
  assert((D->getAsFunction() || isa<ObjCMethodDecl>(D)) &&
         "function or ObjC method decl expected");
  DiagnoseUnguardedFeatureAvailability(*this, D).IssueDiagnostics();
}

void Sema::DiagnoseFeatureAvailabilityOfDecl(NamedDecl *D,
                                             ArrayRef<SourceLocation> Locs) {
  if (!Context.hasFeatureAvailabilityAttr(D))
    return;

  if (FunctionScopeInfo *Context = getCurFunctionAvailabilityContext()) {
    Context->HasPotentialFeatureAvailabilityViolations = true;
    return;
  }

  if (DelayedDiagnostics.shouldDelayDiagnostics()) {
    DelayedDiagnostics.add(DelayedDiagnostic::makeFeatureAvailability(D, Locs));
    return;
  }

  Decl *Ctx = cast<Decl>(getCurLexicalContext());
  diagnoseDeclFeatureAvailability(D, Locs.front(), Ctx, *this);
}
