//===--- HeuristicResolver.cpp ---------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "HeuristicResolver.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/AST/ExprCXX.h"

namespace clang {
namespace clangd {

// Convenience lambdas for use as the 'Filter' parameter of
// HeuristicResolver::resolveDependentMember().
const auto NoFilter = [](const NamedDecl *D) { return true; };
const auto NonStaticFilter = [](const NamedDecl *D) {
  return D->isCXXInstanceMember();
};
const auto StaticFilter = [](const NamedDecl *D) {
  return !D->isCXXInstanceMember();
};
const auto ValueFilter = [](const NamedDecl *D) { return isa<ValueDecl>(D); };
const auto TypeFilter = [](const NamedDecl *D) { return isa<TypeDecl>(D); };
const auto TemplateFilter = [](const NamedDecl *D) {
  return isa<TemplateDecl>(D);
};

// Helper function for HeuristicResolver::resolveDependentMember()
// which takes a possibly-dependent type `T` and heuristically
// resolves it to a CXXRecordDecl in which we can try name lookup.
CXXRecordDecl *resolveTypeToRecordDecl(const Type *T) {
  assert(T);

  if (const auto *RT = T->getAs<RecordType>())
    return dyn_cast<CXXRecordDecl>(RT->getDecl());

  if (const auto *ICNT = T->getAs<InjectedClassNameType>())
    T = ICNT->getInjectedSpecializationType().getTypePtrOrNull();
  if (!T)
    return nullptr;

  const auto *TST = T->getAs<TemplateSpecializationType>();
  if (!TST)
    return nullptr;

  const ClassTemplateDecl *TD = dyn_cast_or_null<ClassTemplateDecl>(
      TST->getTemplateName().getAsTemplateDecl());
  if (!TD)
    return nullptr;

  return TD->getTemplatedDecl();
}

const Type *HeuristicResolver::getPointeeType(const Type *T) const {
  if (!T)
    return nullptr;

  if (T->isPointerType())
    return T->castAs<PointerType>()->getPointeeType().getTypePtrOrNull();

  // Try to handle smart pointer types.

  // Look up operator-> in the primary template. If we find one, it's probably a
  // smart pointer type.
  auto ArrowOps = resolveDependentMember(
      T, Ctx.DeclarationNames.getCXXOperatorName(OO_Arrow), NonStaticFilter);
  if (ArrowOps.empty())
    return nullptr;

  // Getting the return type of the found operator-> method decl isn't useful,
  // because we discarded template arguments to perform lookup in the primary
  // template scope, so the return type would just have the form U* where U is a
  // template parameter type.
  // Instead, just handle the common case where the smart pointer type has the
  // form of SmartPtr<X, ...>, and assume X is the pointee type.
  auto *TST = T->getAs<TemplateSpecializationType>();
  if (!TST)
    return nullptr;
  if (TST->getNumArgs() == 0)
    return nullptr;
  const TemplateArgument &FirstArg = TST->getArg(0);
  if (FirstArg.getKind() != TemplateArgument::Type)
    return nullptr;
  return FirstArg.getAsType().getTypePtrOrNull();
}

std::vector<const NamedDecl *> HeuristicResolver::resolveMemberExpr(
    const CXXDependentScopeMemberExpr *ME) const {
  // If the expression has a qualifier, first try resolving the member
  // inside the qualifier's type.
  // Note that we cannot use a NonStaticFilter in either case, for a couple
  // of reasons:
  //   1. It's valid to access a static member using instance member syntax,
  //      e.g. `instance.static_member`.
  //   2. We can sometimes get a CXXDependentScopeMemberExpr for static
  //      member syntax too, e.g. if `X::static_member` occurs inside
  //      an instance method, it's represented as a CXXDependentScopeMemberExpr
  //      with `this` as the base expression as `X` as the qualifier
  //      (which could be valid if `X` names a base class after instantiation).
  if (NestedNameSpecifier *NNS = ME->getQualifier()) {
    if (const Type *QualifierType = resolveNestedNameSpecifierToType(NNS)) {
      auto Decls =
          resolveDependentMember(QualifierType, ME->getMember(), NoFilter);
      if (!Decls.empty())
        return Decls;
    }
  }

  // If that didn't yield any results, try resolving the member inside
  // the expression's base type.
  const Type *BaseType = ME->getBaseType().getTypePtrOrNull();
  if (ME->isArrow()) {
    BaseType = getPointeeType(BaseType);
  }
  if (!BaseType)
    return {};
  if (const auto *BT = BaseType->getAs<BuiltinType>()) {
    // If BaseType is the type of a dependent expression, it's just
    // represented as BultinType::Dependent which gives us no information. We
    // can get further by analyzing the depedent expression.
    Expr *Base = ME->isImplicitAccess() ? nullptr : ME->getBase();
    if (Base && BT->getKind() == BuiltinType::Dependent) {
      BaseType = resolveExprToType(Base);
    }
  }
  return resolveDependentMember(BaseType, ME->getMember(), NoFilter);
}

std::vector<const NamedDecl *> HeuristicResolver::resolveDeclRefExpr(
    const DependentScopeDeclRefExpr *RE) const {
  return resolveDependentMember(RE->getQualifier()->getAsType(),
                                RE->getDeclName(), StaticFilter);
}

std::vector<const NamedDecl *>
HeuristicResolver::resolveTypeOfCallExpr(const CallExpr *CE) const {
  const auto *CalleeType = resolveExprToType(CE->getCallee());
  if (!CalleeType)
    return {};
  if (const auto *FnTypePtr = CalleeType->getAs<PointerType>())
    CalleeType = FnTypePtr->getPointeeType().getTypePtr();
  if (const FunctionType *FnType = CalleeType->getAs<FunctionType>()) {
    if (const auto *D =
            resolveTypeToRecordDecl(FnType->getReturnType().getTypePtr())) {
      return {D};
    }
  }
  return {};
}

std::vector<const NamedDecl *>
HeuristicResolver::resolveCalleeOfCallExpr(const CallExpr *CE) const {
  if (const auto *ND = dyn_cast_or_null<NamedDecl>(CE->getCalleeDecl())) {
    return {ND};
  }

  return resolveExprToDecls(CE->getCallee());
}

std::vector<const NamedDecl *> HeuristicResolver::resolveUsingValueDecl(
    const UnresolvedUsingValueDecl *UUVD) const {
  return resolveDependentMember(UUVD->getQualifier()->getAsType(),
                                UUVD->getNameInfo().getName(), ValueFilter);
}

std::vector<const NamedDecl *> HeuristicResolver::resolveDependentNameType(
    const DependentNameType *DNT) const {
  return resolveDependentMember(
      resolveNestedNameSpecifierToType(DNT->getQualifier()),
      DNT->getIdentifier(), TypeFilter);
}

std::vector<const NamedDecl *>
HeuristicResolver::resolveTemplateSpecializationType(
    const DependentTemplateSpecializationType *DTST) const {
  return resolveDependentMember(
      resolveNestedNameSpecifierToType(DTST->getQualifier()),
      DTST->getIdentifier(), TemplateFilter);
}

const Type *resolveDeclsToType(const std::vector<const NamedDecl *> &Decls) {
  if (Decls.size() != 1) // Names an overload set -- just bail.
    return nullptr;
  if (const auto *TD = dyn_cast<TypeDecl>(Decls[0])) {
    return TD->getTypeForDecl();
  }
  if (const auto *VD = dyn_cast<ValueDecl>(Decls[0])) {
    return VD->getType().getTypePtrOrNull();
  }
  return nullptr;
}

std::vector<const NamedDecl *>
HeuristicResolver::resolveExprToDecls(const Expr *E) const {
  if (const auto *ME = dyn_cast<CXXDependentScopeMemberExpr>(E)) {
    return resolveMemberExpr(ME);
  }
  if (const auto *RE = dyn_cast<DependentScopeDeclRefExpr>(E)) {
    return resolveDeclRefExpr(RE);
  }
  if (const auto *OE = dyn_cast<OverloadExpr>(E)) {
    return {OE->decls_begin(), OE->decls_end()};
  }
  if (const auto *CE = dyn_cast<CallExpr>(E)) {
    return resolveTypeOfCallExpr(CE);
  }
  if (const auto *ME = dyn_cast<MemberExpr>(E))
    return {ME->getMemberDecl()};

  return {};
}

const Type *HeuristicResolver::resolveExprToType(const Expr *E) const {
  std::vector<const NamedDecl *> Decls = resolveExprToDecls(E);
  if (!Decls.empty())
    return resolveDeclsToType(Decls);

  return E->getType().getTypePtr();
}

const Type *HeuristicResolver::resolveNestedNameSpecifierToType(
    const NestedNameSpecifier *NNS) const {
  if (!NNS)
    return nullptr;

  // The purpose of this function is to handle the dependent (Kind ==
  // Identifier) case, but we need to recurse on the prefix because
  // that may be dependent as well, so for convenience handle
  // the TypeSpec cases too.
  switch (NNS->getKind()) {
  case NestedNameSpecifier::TypeSpec:
  case NestedNameSpecifier::TypeSpecWithTemplate:
    return NNS->getAsType();
  case NestedNameSpecifier::Identifier: {
    return resolveDeclsToType(resolveDependentMember(
        resolveNestedNameSpecifierToType(NNS->getPrefix()),
        NNS->getAsIdentifier(), TypeFilter));
  }
  default:
    break;
  }
  return nullptr;
}

std::vector<const NamedDecl *> HeuristicResolver::resolveDependentMember(
    const Type *T, DeclarationName Name,
    llvm::function_ref<bool(const NamedDecl *ND)> Filter) const {
  if (!T)
    return {};
  if (auto *ET = T->getAs<EnumType>()) {
    auto Result = ET->getDecl()->lookup(Name);
    return {Result.begin(), Result.end()};
  }
  if (auto *RD = resolveTypeToRecordDecl(T)) {
    if (!RD->hasDefinition())
      return {};
    RD = RD->getDefinition();
    return RD->lookupDependentName(Name, Filter);
  }
  return {};
}

} // namespace clangd
} // namespace clang
