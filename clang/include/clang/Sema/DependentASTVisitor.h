//===--- DependentASTVisitor.h - Helper for dependent nodes -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  This file defines the DependentASTVisitor RecursiveASTVisitor layer, which
//  is responsible for visiting unresolved symbol references.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_AST_DEPENDENT_SEMA_VISITOR_H
#define LLVM_CLANG_AST_DEPENDENT_SEMA_VISITOR_H

#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/AST/Type.h"
#include "clang/Sema/HeuristicResolver.h"

namespace clang {

// TODO: Use in the indexer.
template <typename Derived>
class DependentASTVisitor : public RecursiveASTVisitor<Derived> {
private:
  bool visitDependentReference(
      const Type *T, const DeclarationName &Name, SourceLocation Loc,
      llvm::function_ref<bool(const NamedDecl *ND)> Filter) {
    if (!T)
      return true;
    const TemplateSpecializationType *TST =
        T->getAs<TemplateSpecializationType>();
    if (!TST)
      return true;
    TemplateName TN = TST->getTemplateName();
    const ClassTemplateDecl *TD =
        dyn_cast_or_null<ClassTemplateDecl>(TN.getAsTemplateDecl());
    if (!TD)
      return true;
    CXXRecordDecl *RD = TD->getTemplatedDecl();
    if (!RD->hasDefinition())
      return true;
    RD = RD->getDefinition();
    std::vector<const NamedDecl *> Symbols =
        HeuristicResolver(RD->getASTContext())
            .lookupDependentName(RD, Name, Filter);
    // FIXME: Improve overload handling.
    if (Symbols.size() != 1)
      return true;
    if (Loc.isInvalid())
      return true;
    return RecursiveASTVisitor<Derived>::getDerived()
        .VisitDependentSymbolReference(Symbols[0], Loc);
  }

public:
  bool VisitCXXDependentScopeMemberExpr(CXXDependentScopeMemberExpr *E) {
    const DeclarationNameInfo &Info = E->getMemberNameInfo();
    return visitDependentReference(
        E->getBaseType().getTypePtrOrNull(), Info.getName(), Info.getLoc(),
        [](const NamedDecl *D) { return D->isCXXInstanceMember(); });
  }

  bool VisitDependentScopeDeclRefExpr(DependentScopeDeclRefExpr *E) {
    const DeclarationNameInfo &Info = E->getNameInfo();
    const NestedNameSpecifier *NNS = E->getQualifier();
    return visitDependentReference(
        NNS->getAsType(), Info.getName(), Info.getLoc(),
        [](const NamedDecl *D) { return !D->isCXXInstanceMember(); });
  }

  bool VisitDependentNameTypeLoc(DependentNameTypeLoc TL) {
    const DependentNameType *DNT = TL.getTypePtr();
    const NestedNameSpecifier *NNS = DNT->getQualifier();
    DeclarationName Name(DNT->getIdentifier());
    return visitDependentReference(
        NNS->getAsType(), Name, TL.getNameLoc(),
        [](const NamedDecl *ND) { return isa<TypeDecl>(ND); });
  }

  bool VisitDependentSymbolReference(const NamedDecl *Symbol,
                                     SourceLocation SymbolNameLoc) {
    return true;
  }
};

} // end namespace clang

#endif // LLVM_CLANG_AST_DEPENDENT_SEMA_VISITOR_H
