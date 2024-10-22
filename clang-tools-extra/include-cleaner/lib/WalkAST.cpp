//===--- WalkAST.cpp - Find declaration references in the AST -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AnalysisInternal.h"
#include "clang-include-cleaner/Types.h"
#include "clang/AST/ASTFwd.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclFriend.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/AST/Expr.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/NestedNameSpecifier.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/AST/TemplateBase.h"
#include "clang/AST/TemplateName.h"
#include "clang/AST/Type.h"
#include "clang/AST/TypeLoc.h"
#include "clang/Basic/IdentifierTable.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Basic/Specifiers.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"

namespace clang::include_cleaner {
namespace {
using DeclCallback =
    llvm::function_ref<void(SourceLocation, NamedDecl &, RefType)>;

class ASTWalker : public RecursiveASTVisitor<ASTWalker> {
  DeclCallback Callback;

  void report(SourceLocation Loc, NamedDecl *ND,
              RefType RT = RefType::Explicit) {
    if (!ND || Loc.isInvalid())
      return;
    Callback(Loc, *cast<NamedDecl>(ND->getCanonicalDecl()), RT);
  }

  NamedDecl *resolveTemplateName(TemplateName TN) {
    // For using-templates, only mark the alias.
    if (auto *USD = TN.getAsUsingShadowDecl())
      return USD;
    return TN.getAsTemplateDecl();
  }
  NamedDecl *getMemberProvider(QualType Base) {
    if (Base->isPointerType())
      return getMemberProvider(Base->getPointeeType());
    // Unwrap the sugar ElaboratedType.
    if (const auto *ElTy = dyn_cast<ElaboratedType>(Base))
      return getMemberProvider(ElTy->getNamedType());

    if (const auto *TT = dyn_cast<TypedefType>(Base))
      return TT->getDecl();
    if (const auto *UT = dyn_cast<UsingType>(Base))
      return UT->getFoundDecl();
    // A heuristic: to resolve a template type to **only** its template name.
    // We're only using this method for the base type of MemberExpr, in general
    // the template provides the member, and the critical case `unique_ptr<Foo>`
    // is supported (the base type is a Foo*).
    //
    // There are some exceptions that this heuristic could fail (dependent base,
    // dependent typealias), but we believe these are rare.
    if (const auto *TST = dyn_cast<TemplateSpecializationType>(Base))
      return resolveTemplateName(TST->getTemplateName());
    return Base->getAsRecordDecl();
  }
  // Templated as TemplateSpecializationType and
  // DeducedTemplateSpecializationType doesn't share a common base.
  template <typename T>
  // Picks the most specific specialization for a
  // (Deduced)TemplateSpecializationType, while prioritizing using-decls.
  NamedDecl *getMostRelevantTemplatePattern(const T *TST) {
    // In case of exported template names always prefer the using-decl. This
    // implies we'll point at the using-decl even when there's an explicit
    // specializaiton using the exported name, but that's rare.
    auto *ND = resolveTemplateName(TST->getTemplateName());
    if (llvm::isa_and_present<UsingShadowDecl, TypeAliasTemplateDecl>(ND))
      return ND;
    // This is the underlying decl used by TemplateSpecializationType, can be
    // null when type is dependent or not resolved to a pattern yet.
    // If so, fallback to primary template.
    CXXRecordDecl *TD = TST->getAsCXXRecordDecl();
    if (!TD || TD->getTemplateSpecializationKind() == TSK_Undeclared)
      return ND;
    // We ignore explicit instantiations. This might imply marking the wrong
    // declaration as used in specific cases, but seems like the right trade-off
    // in general (e.g. we don't want to include a custom library that has an
    // explicit specialization of a common type).
    if (auto *Pat = TD->getTemplateInstantiationPattern())
      return Pat;
    // For explicit specializations, use the specialized decl directly.
    return TD;
  }

public:
  ASTWalker(DeclCallback Callback) : Callback(Callback) {}

  // Operators are almost always ADL extension points and by design references
  // to them doesn't count as uses (generally the type should provide them, so
  // ignore them).
  // Unless we're using an operator defined as a member, in such cases treat
  // these as regular member references.
  bool TraverseCXXOperatorCallExpr(CXXOperatorCallExpr *S) {
    if (!WalkUpFromCXXOperatorCallExpr(S))
      return false;
    if (auto *CD = S->getCalleeDecl()) {
      if (llvm::isa<CXXMethodDecl>(CD)) {
        // Treat this as a regular member reference.
        report(S->getOperatorLoc(), getMemberProvider(S->getArg(0)->getType()),
               RefType::Implicit);
      } else {
        report(S->getOperatorLoc(), llvm::dyn_cast<NamedDecl>(CD),
               RefType::Implicit);
      }
    }
    for (auto *Arg : S->arguments())
      if (!TraverseStmt(Arg))
        return false;
    return true;
  }

  bool qualifierIsNamespaceOrNone(DeclRefExpr *DRE) {
    const auto *Qual = DRE->getQualifier();
    if (!Qual)
      return true;
    switch (Qual->getKind()) {
    case NestedNameSpecifier::Namespace:
    case NestedNameSpecifier::NamespaceAlias:
    case NestedNameSpecifier::Global:
      return true;
    case NestedNameSpecifier::TypeSpec:
    case NestedNameSpecifier::TypeSpecWithTemplate:
    case NestedNameSpecifier::Super:
    case NestedNameSpecifier::Identifier:
      return false;
    }
    llvm_unreachable("Unknown value for NestedNameSpecifierKind");
  }

  bool VisitDeclRefExpr(DeclRefExpr *DRE) {
    auto *FD = DRE->getFoundDecl();
    // Prefer the underlying decl if FoundDecl isn't a shadow decl, e.g:
    // - For templates, found-decl is always primary template, but we want the
    // specializaiton itself.
    if (!llvm::isa<UsingShadowDecl>(FD))
      FD = DRE->getDecl();
    // For refs to non-meber-like decls, use the found decl.
    // For member-like decls, we should have a reference from the qualifier to
    // the container decl instead, which is preferred as it'll handle
    // aliases/exports properly.
    if (!FD->isCXXClassMember() && !llvm::isa<EnumConstantDecl>(FD)) {
      report(DRE->getLocation(), FD);
      return true;
    }
    // If the ref is without a qualifier, and is a member, ignore it. As it is
    // available in current context due to some other construct (e.g. base
    // specifiers, using decls) that has to spell the name explicitly.
    //
    // If it's an enum constant, it must be due to prior decl. Report references
    // to it when qualifier isn't a type.
    if (llvm::isa<EnumConstantDecl>(FD) && qualifierIsNamespaceOrNone(DRE))
      report(DRE->getLocation(), FD);
    return true;
  }

  bool VisitMemberExpr(MemberExpr *E) {
    // Reporting a usage of the member decl would cause issues (e.g. force
    // including the base class for inherited members). Instead, we report a
    // usage of the base type of the MemberExpr, so that e.g. code
    // `returnFoo().bar` can keep #include "foo.h" (rather than inserting
    // "bar.h" for the underlying base type `Bar`).
    QualType Type = E->getBase()->IgnoreImpCasts()->getType();
    report(E->getMemberLoc(), getMemberProvider(Type), RefType::Implicit);
    return true;
  }
  bool VisitCXXDependentScopeMemberExpr(CXXDependentScopeMemberExpr *E) {
    report(E->getMemberLoc(), getMemberProvider(E->getBaseType()),
           RefType::Implicit);
    return true;
  }

  bool VisitCXXConstructExpr(CXXConstructExpr *E) {
    // Always treat consturctor calls as implicit. We'll have an explicit
    // reference for the constructor calls that mention the type-name (through
    // TypeLocs). This reference only matters for cases where there's no
    // explicit syntax at all or there're only braces.
    report(E->getLocation(), getMemberProvider(E->getType()),
           RefType::Implicit);
    return true;
  }

  bool VisitOverloadExpr(OverloadExpr *E) {
    // Since we can't prove which overloads are used, report all of them.
    for (NamedDecl *D : E->decls())
      report(E->getNameLoc(), D, RefType::Ambiguous);
    return true;
  }

  // Report all (partial) specializations of a class/var template decl.
  template <typename TemplateDeclType, typename ParitialDeclType>
  void reportSpecializations(SourceLocation Loc, NamedDecl *ND) {
    const auto *TD = llvm::dyn_cast<TemplateDeclType>(ND);
    if (!TD)
      return;

    for (auto *Spec : TD->specializations())
      report(Loc, Spec, RefType::Ambiguous);
    llvm::SmallVector<ParitialDeclType *> PartialSpecializations;
    TD->getPartialSpecializations(PartialSpecializations);
    for (auto *PartialSpec : PartialSpecializations)
      report(Loc, PartialSpec, RefType::Ambiguous);
  }
  bool VisitUsingDecl(UsingDecl *UD) {
    for (const auto *Shadow : UD->shadows()) {
      auto *TD = Shadow->getTargetDecl();
      // For function-decls, we might have overloads brought in due to
      // transitive dependencies. Hence we only want to report explicit
      // references for those if they're used.
      // But for record decls, spelling of the type always refers to primary
      // decl non-ambiguously. Hence spelling is already a use.
      auto IsUsed = TD->isUsed() || TD->isReferenced() || !TD->getAsFunction();
      report(UD->getLocation(), TD,
             IsUsed ? RefType::Explicit : RefType::Ambiguous);

      // All (partial) template specializations are visible via a using-decl,
      // However a using-decl only refers to the primary template (per C++ name
      // lookup). Thus, we need to manually report all specializations.
      reportSpecializations<ClassTemplateDecl,
                            ClassTemplatePartialSpecializationDecl>(
          UD->getLocation(), TD);
      reportSpecializations<VarTemplateDecl,
                            VarTemplatePartialSpecializationDecl>(
          UD->getLocation(), TD);
      if (const auto *FTD = llvm::dyn_cast<FunctionTemplateDecl>(TD))
        for (auto *Spec : FTD->specializations())
          report(UD->getLocation(), Spec, RefType::Ambiguous);
    }
    return true;
  }

  bool VisitFunctionDecl(FunctionDecl *FD) {
    // Mark declaration from definition as it needs type-checking.
    if (FD->isThisDeclarationADefinition())
      report(FD->getLocation(), FD);
    // Explicit specializaiton/instantiations of a function template requires
    // primary template.
    if (clang::isTemplateExplicitInstantiationOrSpecialization(
            FD->getTemplateSpecializationKind()))
      report(FD->getLocation(), FD->getPrimaryTemplate());
    return true;
  }
  bool VisitVarDecl(VarDecl *VD) {
    // Ignore the parameter decl itself (its children were handled elsewhere),
    // as they don't contribute to the main-file #include.
    if (llvm::isa<ParmVarDecl>(VD))
      return true;
    // Mark declaration from definition as it needs type-checking.
    if (VD->isThisDeclarationADefinition())
      report(VD->getLocation(), VD);
    return true;
  }

  bool VisitEnumDecl(EnumDecl *D) {
    // Definition of an enum with an underlying type references declaration for
    // type-checking purposes.
    if (D->isThisDeclarationADefinition() && D->getIntegerTypeSourceInfo())
      report(D->getLocation(), D);
    return true;
  }

  bool VisitFriendDecl(FriendDecl *D) {
    // We already visit the TypeLoc properly, but need to special case the decl
    // case.
    if (auto *FD = D->getFriendDecl())
      report(D->getLocation(), FD);
    return true;
  }

  bool VisitConceptReference(const ConceptReference *CR) {
    report(CR->getConceptNameLoc(), CR->getFoundDecl());
    return true;
  }

  // Report a reference from explicit specializations/instantiations to the
  // specialized template. Implicit ones are filtered out by RAV.
  bool
  VisitClassTemplateSpecializationDecl(ClassTemplateSpecializationDecl *CTSD) {
    if (clang::isTemplateExplicitInstantiationOrSpecialization(
            CTSD->getTemplateSpecializationKind()))
      report(CTSD->getLocation(),
             CTSD->getSpecializedTemplate()->getTemplatedDecl());
    return true;
  }
  bool VisitVarTemplateSpecializationDecl(VarTemplateSpecializationDecl *VTSD) {
    if (clang::isTemplateExplicitInstantiationOrSpecialization(
            VTSD->getTemplateSpecializationKind()))
      report(VTSD->getLocation(),
             VTSD->getSpecializedTemplate()->getTemplatedDecl());
    return true;
  }

  // TypeLoc visitors.
  void reportType(SourceLocation RefLoc, NamedDecl *ND) {
    // Reporting explicit references to types nested inside classes can cause
    // issues, e.g. a type accessed through a derived class shouldn't require
    // inclusion of the base.
    // Hence we report all such references as implicit. The code must spell the
    // outer type-location somewhere, which will trigger an explicit reference
    // and per IWYS, it's that spelling's responsibility to bring in necessary
    // declarations.
    RefType RT = llvm::isa<RecordDecl>(ND->getDeclContext())
                     ? RefType::Implicit
                     : RefType::Explicit;
    return report(RefLoc, ND, RT);
  }

  bool VisitUsingTypeLoc(UsingTypeLoc TL) {
    reportType(TL.getNameLoc(), TL.getFoundDecl());
    return true;
  }

  bool VisitTagTypeLoc(TagTypeLoc TTL) {
    reportType(TTL.getNameLoc(), TTL.getDecl());
    return true;
  }

  bool VisitTypedefTypeLoc(TypedefTypeLoc TTL) {
    reportType(TTL.getNameLoc(), TTL.getTypedefNameDecl());
    return true;
  }

  bool VisitTemplateSpecializationTypeLoc(TemplateSpecializationTypeLoc TL) {
    reportType(TL.getTemplateNameLoc(),
               getMostRelevantTemplatePattern(TL.getTypePtr()));
    return true;
  }

  bool VisitDeducedTemplateSpecializationTypeLoc(
      DeducedTemplateSpecializationTypeLoc TL) {
    reportType(TL.getTemplateNameLoc(),
               getMostRelevantTemplatePattern(TL.getTypePtr()));
    return true;
  }

  bool TraverseTemplateArgumentLoc(const TemplateArgumentLoc &TL) {
    auto &Arg = TL.getArgument();
    // Template-template parameters require special attention, as there's no
    // TemplateNameLoc.
    if (Arg.getKind() == TemplateArgument::Template ||
        Arg.getKind() == TemplateArgument::TemplateExpansion) {
      report(TL.getLocation(),
             resolveTemplateName(Arg.getAsTemplateOrTemplatePattern()));
      return true;
    }
    return RecursiveASTVisitor::TraverseTemplateArgumentLoc(TL);
  }

  bool VisitCXXStdInitializerListExpr(CXXStdInitializerListExpr *E) {
    // Reliance on initializer_lists requires std::initializer_list to be
    // visible per standard. So report a reference to it, otherwise include of
    // `<initializer_list>` might not receive any use.
    report(E->getExprLoc(),
           const_cast<CXXRecordDecl *>(E->getBestDynamicClassType()),
           RefType::Implicit);
    return true;
  }

  bool VisitCXXNewExpr(CXXNewExpr *E) {
    report(E->getExprLoc(), E->getOperatorNew(), RefType::Ambiguous);
    return true;
  }
  bool VisitCXXDeleteExpr(CXXDeleteExpr *E) {
    report(E->getExprLoc(), E->getOperatorDelete(), RefType::Ambiguous);
    return true;
  }
};

} // namespace

void walkAST(Decl &Root, DeclCallback Callback) {
  ASTWalker(Callback).TraverseDecl(&Root);
}

} // namespace clang::include_cleaner
