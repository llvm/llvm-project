//===--- HeuristicResolver.cpp ---------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "HeuristicResolver.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/CXXInheritance.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/Type.h"

namespace clang {
namespace clangd {

namespace {

// Helper class for implementing HeuristicResolver.
// Unlike HeuristicResolver which is a long-lived class,
// a new instance of this class is created for every external
// call into a HeuristicResolver operation. That allows this
// class to store state that's local to such a top-level call,
// particularly "recursion protection sets" that keep track of
// nodes that have already been seen to avoid infinite recursion.
class HeuristicResolverImpl {
public:
  HeuristicResolverImpl(ASTContext &Ctx) : Ctx(Ctx) {}

  // These functions match the public interface of HeuristicResolver
  // (but aren't const since they may modify the recursion protection sets).
  std::vector<const NamedDecl *>
  resolveMemberExpr(const CXXDependentScopeMemberExpr *ME);
  std::vector<const NamedDecl *>
  resolveDeclRefExpr(const DependentScopeDeclRefExpr *RE);
  std::vector<const NamedDecl *> resolveTypeOfCallExpr(const CallExpr *CE);
  std::vector<const NamedDecl *> resolveCalleeOfCallExpr(const CallExpr *CE);
  std::vector<const NamedDecl *>
  resolveUsingValueDecl(const UnresolvedUsingValueDecl *UUVD);
  std::vector<const NamedDecl *>
  resolveDependentNameType(const DependentNameType *DNT);
  std::vector<const NamedDecl *> resolveTemplateSpecializationType(
      const DependentTemplateSpecializationType *DTST);
  const Type *resolveNestedNameSpecifierToType(const NestedNameSpecifier *NNS);
  const Type *getPointeeType(const Type *T);

private:
  ASTContext &Ctx;

  // Recursion protection sets
  llvm::SmallSet<const DependentNameType *, 4> SeenDependentNameTypes;

  // Given a tag-decl type and a member name, heuristically resolve the
  // name to one or more declarations.
  // The current heuristic is simply to look up the name in the primary
  // template. This is a heuristic because the template could potentially
  // have specializations that declare different members.
  // Multiple declarations could be returned if the name is overloaded
  // (e.g. an overloaded method in the primary template).
  // This heuristic will give the desired answer in many cases, e.g.
  // for a call to vector<T>::size().
  std::vector<const NamedDecl *>
  resolveDependentMember(const Type *T, DeclarationName Name,
                         llvm::function_ref<bool(const NamedDecl *ND)> Filter);

  // Try to heuristically resolve the type of a possibly-dependent expression
  // `E`.
  const Type *resolveExprToType(const Expr *E);
  std::vector<const NamedDecl *> resolveExprToDecls(const Expr *E);

  // Helper function for HeuristicResolver::resolveDependentMember()
  // which takes a possibly-dependent type `T` and heuristically
  // resolves it to a CXXRecordDecl in which we can try name lookup.
  CXXRecordDecl *resolveTypeToRecordDecl(const Type *T);

  // This is a reimplementation of CXXRecordDecl::lookupDependentName()
  // so that the implementation can call into other HeuristicResolver helpers.
  // FIXME: Once HeuristicResolver is upstreamed to the clang libraries
  // (https://github.com/clangd/clangd/discussions/1662),
  // CXXRecordDecl::lookupDepenedentName() can be removed, and its call sites
  // can be modified to benefit from the more comprehensive heuristics offered
  // by HeuristicResolver instead.
  std::vector<const NamedDecl *>
  lookupDependentName(CXXRecordDecl *RD, DeclarationName Name,
                      llvm::function_ref<bool(const NamedDecl *ND)> Filter);
  bool findOrdinaryMemberInDependentClasses(const CXXBaseSpecifier *Specifier,
                                            CXXBasePath &Path,
                                            DeclarationName Name);
};

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

const Type *resolveDeclsToType(const std::vector<const NamedDecl *> &Decls,
                               ASTContext &Ctx) {
  if (Decls.size() != 1) // Names an overload set -- just bail.
    return nullptr;
  if (const auto *TD = dyn_cast<TypeDecl>(Decls[0])) {
    return Ctx.getTypeDeclType(TD).getTypePtr();
  }
  if (const auto *VD = dyn_cast<ValueDecl>(Decls[0])) {
    return VD->getType().getTypePtrOrNull();
  }
  return nullptr;
}

// Helper function for HeuristicResolver::resolveDependentMember()
// which takes a possibly-dependent type `T` and heuristically
// resolves it to a CXXRecordDecl in which we can try name lookup.
CXXRecordDecl *HeuristicResolverImpl::resolveTypeToRecordDecl(const Type *T) {
  assert(T);

  // Unwrap type sugar such as type aliases.
  T = T->getCanonicalTypeInternal().getTypePtr();

  if (const auto *DNT = T->getAs<DependentNameType>()) {
    T = resolveDeclsToType(resolveDependentNameType(DNT), Ctx);
    if (!T)
      return nullptr;
    T = T->getCanonicalTypeInternal().getTypePtr();
  }

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

const Type *HeuristicResolverImpl::getPointeeType(const Type *T) {
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
  if (TST->template_arguments().size() == 0)
    return nullptr;
  const TemplateArgument &FirstArg = TST->template_arguments()[0];
  if (FirstArg.getKind() != TemplateArgument::Type)
    return nullptr;
  return FirstArg.getAsType().getTypePtrOrNull();
}

std::vector<const NamedDecl *> HeuristicResolverImpl::resolveMemberExpr(
    const CXXDependentScopeMemberExpr *ME) {
  // If the expression has a qualifier, try resolving the member inside the
  // qualifier's type.
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

    // Do not proceed to try resolving the member in the expression's base type
    // without regard to the qualifier, as that could produce incorrect results.
    // For example, `void foo() { this->Base::foo(); }` shouldn't resolve to
    // foo() itself!
    return {};
  }

  // Try resolving the member inside the expression's base type.
  const Type *BaseType = ME->getBaseType().getTypePtrOrNull();
  if (ME->isArrow()) {
    BaseType = getPointeeType(BaseType);
  }
  if (!BaseType)
    return {};
  if (const auto *BT = BaseType->getAs<BuiltinType>()) {
    // If BaseType is the type of a dependent expression, it's just
    // represented as BuiltinType::Dependent which gives us no information. We
    // can get further by analyzing the dependent expression.
    Expr *Base = ME->isImplicitAccess() ? nullptr : ME->getBase();
    if (Base && BT->getKind() == BuiltinType::Dependent) {
      BaseType = resolveExprToType(Base);
    }
  }
  return resolveDependentMember(BaseType, ME->getMember(), NoFilter);
}

std::vector<const NamedDecl *>
HeuristicResolverImpl::resolveDeclRefExpr(const DependentScopeDeclRefExpr *RE) {
  return resolveDependentMember(RE->getQualifier()->getAsType(),
                                RE->getDeclName(), StaticFilter);
}

std::vector<const NamedDecl *>
HeuristicResolverImpl::resolveTypeOfCallExpr(const CallExpr *CE) {
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
HeuristicResolverImpl::resolveCalleeOfCallExpr(const CallExpr *CE) {
  if (const auto *ND = dyn_cast_or_null<NamedDecl>(CE->getCalleeDecl())) {
    return {ND};
  }

  return resolveExprToDecls(CE->getCallee());
}

std::vector<const NamedDecl *> HeuristicResolverImpl::resolveUsingValueDecl(
    const UnresolvedUsingValueDecl *UUVD) {
  return resolveDependentMember(UUVD->getQualifier()->getAsType(),
                                UUVD->getNameInfo().getName(), ValueFilter);
}

std::vector<const NamedDecl *>
HeuristicResolverImpl::resolveDependentNameType(const DependentNameType *DNT) {
  if (auto [_, inserted] = SeenDependentNameTypes.insert(DNT); !inserted)
    return {};
  return resolveDependentMember(
      resolveNestedNameSpecifierToType(DNT->getQualifier()),
      DNT->getIdentifier(), TypeFilter);
}

std::vector<const NamedDecl *>
HeuristicResolverImpl::resolveTemplateSpecializationType(
    const DependentTemplateSpecializationType *DTST) {
  return resolveDependentMember(
      resolveNestedNameSpecifierToType(DTST->getQualifier()),
      DTST->getIdentifier(), TemplateFilter);
}

std::vector<const NamedDecl *>
HeuristicResolverImpl::resolveExprToDecls(const Expr *E) {
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

const Type *HeuristicResolverImpl::resolveExprToType(const Expr *E) {
  std::vector<const NamedDecl *> Decls = resolveExprToDecls(E);
  if (!Decls.empty())
    return resolveDeclsToType(Decls, Ctx);

  return E->getType().getTypePtr();
}

const Type *HeuristicResolverImpl::resolveNestedNameSpecifierToType(
    const NestedNameSpecifier *NNS) {
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
    return resolveDeclsToType(
        resolveDependentMember(
            resolveNestedNameSpecifierToType(NNS->getPrefix()),
            NNS->getAsIdentifier(), TypeFilter),
        Ctx);
  }
  default:
    break;
  }
  return nullptr;
}

bool isOrdinaryMember(const NamedDecl *ND) {
  return ND->isInIdentifierNamespace(Decl::IDNS_Ordinary | Decl::IDNS_Tag |
                                     Decl::IDNS_Member);
}

bool findOrdinaryMember(const CXXRecordDecl *RD, CXXBasePath &Path,
                        DeclarationName Name) {
  Path.Decls = RD->lookup(Name).begin();
  for (DeclContext::lookup_iterator I = Path.Decls, E = I.end(); I != E; ++I)
    if (isOrdinaryMember(*I))
      return true;

  return false;
}

bool HeuristicResolverImpl::findOrdinaryMemberInDependentClasses(
    const CXXBaseSpecifier *Specifier, CXXBasePath &Path,
    DeclarationName Name) {
  CXXRecordDecl *RD =
      resolveTypeToRecordDecl(Specifier->getType().getTypePtr());
  if (!RD)
    return false;
  return findOrdinaryMember(RD, Path, Name);
}

std::vector<const NamedDecl *> HeuristicResolverImpl::lookupDependentName(
    CXXRecordDecl *RD, DeclarationName Name,
    llvm::function_ref<bool(const NamedDecl *ND)> Filter) {
  std::vector<const NamedDecl *> Results;

  // Lookup in the class.
  bool AnyOrdinaryMembers = false;
  for (const NamedDecl *ND : RD->lookup(Name)) {
    if (isOrdinaryMember(ND))
      AnyOrdinaryMembers = true;
    if (Filter(ND))
      Results.push_back(ND);
  }
  if (AnyOrdinaryMembers)
    return Results;

  // Perform lookup into our base classes.
  CXXBasePaths Paths;
  Paths.setOrigin(RD);
  if (!RD->lookupInBases(
          [&](const CXXBaseSpecifier *Specifier, CXXBasePath &Path) {
            return findOrdinaryMemberInDependentClasses(Specifier, Path, Name);
          },
          Paths, /*LookupInDependent=*/true))
    return Results;
  for (DeclContext::lookup_iterator I = Paths.front().Decls, E = I.end();
       I != E; ++I) {
    if (isOrdinaryMember(*I) && Filter(*I))
      Results.push_back(*I);
  }
  return Results;
}

std::vector<const NamedDecl *> HeuristicResolverImpl::resolveDependentMember(
    const Type *T, DeclarationName Name,
    llvm::function_ref<bool(const NamedDecl *ND)> Filter) {
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
    return lookupDependentName(RD, Name, Filter);
  }
  return {};
}
} // namespace

std::vector<const NamedDecl *> HeuristicResolver::resolveMemberExpr(
    const CXXDependentScopeMemberExpr *ME) const {
  return HeuristicResolverImpl(Ctx).resolveMemberExpr(ME);
}
std::vector<const NamedDecl *> HeuristicResolver::resolveDeclRefExpr(
    const DependentScopeDeclRefExpr *RE) const {
  return HeuristicResolverImpl(Ctx).resolveDeclRefExpr(RE);
}
std::vector<const NamedDecl *>
HeuristicResolver::resolveTypeOfCallExpr(const CallExpr *CE) const {
  return HeuristicResolverImpl(Ctx).resolveTypeOfCallExpr(CE);
}
std::vector<const NamedDecl *>
HeuristicResolver::resolveCalleeOfCallExpr(const CallExpr *CE) const {
  return HeuristicResolverImpl(Ctx).resolveCalleeOfCallExpr(CE);
}
std::vector<const NamedDecl *> HeuristicResolver::resolveUsingValueDecl(
    const UnresolvedUsingValueDecl *UUVD) const {
  return HeuristicResolverImpl(Ctx).resolveUsingValueDecl(UUVD);
}
std::vector<const NamedDecl *> HeuristicResolver::resolveDependentNameType(
    const DependentNameType *DNT) const {
  return HeuristicResolverImpl(Ctx).resolveDependentNameType(DNT);
}
std::vector<const NamedDecl *>
HeuristicResolver::resolveTemplateSpecializationType(
    const DependentTemplateSpecializationType *DTST) const {
  return HeuristicResolverImpl(Ctx).resolveTemplateSpecializationType(DTST);
}
const Type *HeuristicResolver::resolveNestedNameSpecifierToType(
    const NestedNameSpecifier *NNS) const {
  return HeuristicResolverImpl(Ctx).resolveNestedNameSpecifierToType(NNS);
}
const Type *HeuristicResolver::getPointeeType(const Type *T) const {
  return HeuristicResolverImpl(Ctx).getPointeeType(T);
}

} // namespace clangd
} // namespace clang
