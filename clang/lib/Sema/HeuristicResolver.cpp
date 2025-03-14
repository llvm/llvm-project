//===--- HeuristicResolver.cpp ---------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Sema/HeuristicResolver.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/CXXInheritance.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/TemplateBase.h"
#include "clang/AST/Type.h"

namespace clang {

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
  QualType resolveNestedNameSpecifierToType(const NestedNameSpecifier *NNS);
  QualType getPointeeType(QualType T);
  std::vector<const NamedDecl *>
  lookupDependentName(CXXRecordDecl *RD, DeclarationName Name,
                      llvm::function_ref<bool(const NamedDecl *ND)> Filter);
  TagDecl *resolveTypeToTagDecl(QualType T);
  QualType simplifyType(QualType Type, const Expr *E, bool UnwrapPointer);

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
  resolveDependentMember(QualType T, DeclarationName Name,
                         llvm::function_ref<bool(const NamedDecl *ND)> Filter);

  // Try to heuristically resolve the type of a possibly-dependent expression
  // `E`.
  QualType resolveExprToType(const Expr *E);
  std::vector<const NamedDecl *> resolveExprToDecls(const Expr *E);

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

QualType resolveDeclsToType(const std::vector<const NamedDecl *> &Decls,
                            ASTContext &Ctx) {
  if (Decls.size() != 1) // Names an overload set -- just bail.
    return QualType();
  if (const auto *TD = dyn_cast<TypeDecl>(Decls[0])) {
    return Ctx.getTypeDeclType(TD);
  }
  if (const auto *VD = dyn_cast<ValueDecl>(Decls[0])) {
    return VD->getType();
  }
  return QualType();
}

TemplateName getReferencedTemplateName(const Type *T) {
  if (const auto *TST = T->getAs<TemplateSpecializationType>()) {
    return TST->getTemplateName();
  }
  if (const auto *DTST = T->getAs<DeducedTemplateSpecializationType>()) {
    return DTST->getTemplateName();
  }
  return TemplateName();
}

// Helper function for HeuristicResolver::resolveDependentMember()
// which takes a possibly-dependent type `T` and heuristically
// resolves it to a CXXRecordDecl in which we can try name lookup.
TagDecl *HeuristicResolverImpl::resolveTypeToTagDecl(QualType QT) {
  const Type *T = QT.getTypePtrOrNull();
  if (!T)
    return nullptr;

  // If T is the type of a template parameter, we can't get a useful TagDecl
  // out of it. However, if the template parameter has a default argument,
  // as a heuristic we can replace T with the default argument type.
  if (const auto *TTPT = dyn_cast<TemplateTypeParmType>(T)) {
    if (const auto *TTPD = TTPT->getDecl()) {
      if (TTPD->hasDefaultArgument()) {
        const auto &DefaultArg = TTPD->getDefaultArgument().getArgument();
        if (DefaultArg.getKind() == TemplateArgument::Type) {
          T = DefaultArg.getAsType().getTypePtrOrNull();
        }
      }
    }
  }

  // Unwrap type sugar such as type aliases.
  T = T->getCanonicalTypeInternal().getTypePtr();

  if (const auto *DNT = T->getAs<DependentNameType>()) {
    T = resolveDeclsToType(resolveDependentNameType(DNT), Ctx)
            .getTypePtrOrNull();
    if (!T)
      return nullptr;
    T = T->getCanonicalTypeInternal().getTypePtr();
  }

  if (auto *TT = T->getAs<TagType>()) {
    TagDecl *TD = TT->getDecl();
    // Template might not be instantiated yet, fall back to primary template
    // in such cases.
    if (const auto *CTSD = dyn_cast<ClassTemplateSpecializationDecl>(TD)) {
      if (CTSD->getTemplateSpecializationKind() == TSK_Undeclared) {
        return CTSD->getSpecializedTemplate()->getTemplatedDecl();
      }
    }
    return TD;
  }

  if (const auto *ICNT = T->getAs<InjectedClassNameType>())
    T = ICNT->getInjectedSpecializationType().getTypePtrOrNull();
  if (!T)
    return nullptr;

  TemplateName TN = getReferencedTemplateName(T);
  if (TN.isNull())
    return nullptr;

  const ClassTemplateDecl *TD =
      dyn_cast_or_null<ClassTemplateDecl>(TN.getAsTemplateDecl());
  if (!TD)
    return nullptr;

  return TD->getTemplatedDecl();
}

QualType HeuristicResolverImpl::getPointeeType(QualType T) {
  if (T.isNull())
    return QualType();

  if (T->isPointerType())
    return T->castAs<PointerType>()->getPointeeType();

  // Try to handle smart pointer types.

  // Look up operator-> in the primary template. If we find one, it's probably a
  // smart pointer type.
  auto ArrowOps = resolveDependentMember(
      T, Ctx.DeclarationNames.getCXXOperatorName(OO_Arrow), NonStaticFilter);
  if (ArrowOps.empty())
    return QualType();

  // Getting the return type of the found operator-> method decl isn't useful,
  // because we discarded template arguments to perform lookup in the primary
  // template scope, so the return type would just have the form U* where U is a
  // template parameter type.
  // Instead, just handle the common case where the smart pointer type has the
  // form of SmartPtr<X, ...>, and assume X is the pointee type.
  auto *TST = T->getAs<TemplateSpecializationType>();
  if (!TST)
    return QualType();
  if (TST->template_arguments().size() == 0)
    return QualType();
  const TemplateArgument &FirstArg = TST->template_arguments()[0];
  if (FirstArg.getKind() != TemplateArgument::Type)
    return QualType();
  return FirstArg.getAsType();
}

QualType HeuristicResolverImpl::simplifyType(QualType Type, const Expr *E,
                                             bool UnwrapPointer) {
  bool DidUnwrapPointer = false;
  // A type, together with an optional expression whose type it represents
  // which may have additional information about the expression's type
  // not stored in the QualType itself.
  struct TypeExprPair {
    QualType Type;
    const Expr *E = nullptr;
  };
  TypeExprPair Current{Type, E};
  auto SimplifyOneStep = [UnwrapPointer, &DidUnwrapPointer,
                          this](TypeExprPair T) -> TypeExprPair {
    if (UnwrapPointer) {
      if (QualType Pointee = getPointeeType(T.Type); !Pointee.isNull()) {
        DidUnwrapPointer = true;
        return {Pointee};
      }
    }
    if (const auto *RT = T.Type->getAs<ReferenceType>()) {
      // Does not count as "unwrap pointer".
      return {RT->getPointeeType()};
    }
    if (const auto *BT = T.Type->getAs<BuiltinType>()) {
      // If BaseType is the type of a dependent expression, it's just
      // represented as BuiltinType::Dependent which gives us no information. We
      // can get further by analyzing the dependent expression.
      if (T.E && BT->getKind() == BuiltinType::Dependent) {
        return {resolveExprToType(T.E), T.E};
      }
    }
    if (const auto *AT = T.Type->getContainedAutoType()) {
      // If T contains a dependent `auto` type, deduction will not have
      // been performed on it yet. In simple cases (e.g. `auto` variable with
      // initializer), get the approximate type that would result from
      // deduction.
      // FIXME: A more accurate implementation would propagate things like the
      // `const` in `const auto`.
      if (T.E && AT->isUndeducedAutoType()) {
        if (const auto *DRE = dyn_cast<DeclRefExpr>(T.E)) {
          if (const auto *VD = dyn_cast<VarDecl>(DRE->getDecl())) {
            if (auto *Init = VD->getInit())
              return {resolveExprToType(Init), Init};
          }
        }
      }
    }
    return T;
  };
  // As an additional protection against infinite loops, bound the number of
  // simplification steps.
  size_t StepCount = 0;
  const size_t MaxSteps = 64;
  while (!Current.Type.isNull() && StepCount++ < MaxSteps) {
    TypeExprPair New = SimplifyOneStep(Current);
    if (New.Type == Current.Type)
      break;
    Current = New;
  }
  if (UnwrapPointer && !DidUnwrapPointer)
    return QualType();
  return Current.Type;
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
    if (QualType QualifierType = resolveNestedNameSpecifierToType(NNS);
        !QualifierType.isNull()) {
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
  Expr *Base = ME->isImplicitAccess() ? nullptr : ME->getBase();
  QualType BaseType = ME->getBaseType();
  BaseType = simplifyType(BaseType, Base, ME->isArrow());
  return resolveDependentMember(BaseType, ME->getMember(), NoFilter);
}

std::vector<const NamedDecl *>
HeuristicResolverImpl::resolveDeclRefExpr(const DependentScopeDeclRefExpr *RE) {
  return resolveDependentMember(
      resolveNestedNameSpecifierToType(RE->getQualifier()), RE->getDeclName(),
      StaticFilter);
}

std::vector<const NamedDecl *>
HeuristicResolverImpl::resolveTypeOfCallExpr(const CallExpr *CE) {
  QualType CalleeType = resolveExprToType(CE->getCallee());
  if (CalleeType.isNull())
    return {};
  if (const auto *FnTypePtr = CalleeType->getAs<PointerType>())
    CalleeType = FnTypePtr->getPointeeType();
  if (const FunctionType *FnType = CalleeType->getAs<FunctionType>()) {
    if (const auto *D = resolveTypeToTagDecl(FnType->getReturnType())) {
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
  return resolveDependentMember(QualType(UUVD->getQualifier()->getAsType(), 0),
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

QualType HeuristicResolverImpl::resolveExprToType(const Expr *E) {
  std::vector<const NamedDecl *> Decls = resolveExprToDecls(E);
  if (!Decls.empty())
    return resolveDeclsToType(Decls, Ctx);

  return E->getType();
}

QualType HeuristicResolverImpl::resolveNestedNameSpecifierToType(
    const NestedNameSpecifier *NNS) {
  if (!NNS)
    return QualType();

  // The purpose of this function is to handle the dependent (Kind ==
  // Identifier) case, but we need to recurse on the prefix because
  // that may be dependent as well, so for convenience handle
  // the TypeSpec cases too.
  switch (NNS->getKind()) {
  case NestedNameSpecifier::TypeSpec:
  case NestedNameSpecifier::TypeSpecWithTemplate:
    return QualType(NNS->getAsType(), 0);
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
  return QualType();
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
  TagDecl *TD = resolveTypeToTagDecl(Specifier->getType());
  if (const auto *RD = dyn_cast_if_present<CXXRecordDecl>(TD)) {
    return findOrdinaryMember(RD, Path, Name);
  }
  return false;
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
    QualType QT, DeclarationName Name,
    llvm::function_ref<bool(const NamedDecl *ND)> Filter) {
  TagDecl *TD = resolveTypeToTagDecl(QT);
  if (!TD)
    return {};
  if (auto *ED = dyn_cast<EnumDecl>(TD)) {
    auto Result = ED->lookup(Name);
    return {Result.begin(), Result.end()};
  }
  if (auto *RD = dyn_cast<CXXRecordDecl>(TD)) {
    if (!RD->hasDefinition())
      return {};
    RD = RD->getDefinition();
    return lookupDependentName(RD, Name, [&](const NamedDecl *ND) {
      if (!Filter(ND))
        return false;
      if (const auto *MD = dyn_cast<CXXMethodDecl>(ND)) {
        return !MD->isInstance() ||
               MD->getMethodQualifiers().compatiblyIncludes(QT.getQualifiers(),
                                                            Ctx);
      }
      return true;
    });
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
QualType HeuristicResolver::resolveNestedNameSpecifierToType(
    const NestedNameSpecifier *NNS) const {
  return HeuristicResolverImpl(Ctx).resolveNestedNameSpecifierToType(NNS);
}
std::vector<const NamedDecl *> HeuristicResolver::lookupDependentName(
    CXXRecordDecl *RD, DeclarationName Name,
    llvm::function_ref<bool(const NamedDecl *ND)> Filter) {
  return HeuristicResolverImpl(Ctx).lookupDependentName(RD, Name, Filter);
}
const QualType HeuristicResolver::getPointeeType(QualType T) const {
  return HeuristicResolverImpl(Ctx).getPointeeType(T);
}
TagDecl *HeuristicResolver::resolveTypeToTagDecl(QualType T) const {
  return HeuristicResolverImpl(Ctx).resolveTypeToTagDecl(T);
}
QualType HeuristicResolver::simplifyType(QualType Type, const Expr *E,
                                         bool UnwrapPointer) {
  return HeuristicResolverImpl(Ctx).simplifyType(Type, E, UnwrapPointer);
}

} // namespace clang
