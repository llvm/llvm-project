//===--- ForwardingReferenceOverloadCheck.cpp - clang-tidy-----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ForwardingReferenceOverloadCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include <algorithm>

using namespace clang::ast_matchers;

namespace clang {
namespace tidy {
namespace bugprone {

namespace {
// Check if the given type is related to std::enable_if.
AST_MATCHER(QualType, isEnableIf) {
  auto CheckTemplate = [](const TemplateSpecializationType *Spec) {
    if (!Spec || !Spec->getTemplateName().getAsTemplateDecl()) {
      return false;
    }
    const NamedDecl *TypeDecl =
        Spec->getTemplateName().getAsTemplateDecl()->getTemplatedDecl();
    return TypeDecl->isInStdNamespace() &&
           (TypeDecl->getName().equals("enable_if") ||
            TypeDecl->getName().equals("enable_if_t"));
  };
  const Type *BaseType = Node.getTypePtr();
  // Case: pointer or reference to enable_if.
  while (BaseType->isPointerType() || BaseType->isReferenceType()) {
    BaseType = BaseType->getPointeeType().getTypePtr();
  }
  // Case: type parameter dependent (enable_if<is_integral<T>>).
  if (const auto *Dependent = BaseType->getAs<DependentNameType>()) {
    BaseType = Dependent->getQualifier()->getAsType();
  }
  if (!BaseType)
    return false;
  if (CheckTemplate(BaseType->getAs<TemplateSpecializationType>()))
    return true; // Case: enable_if_t< >.
  if (const auto *Elaborated = BaseType->getAs<ElaboratedType>()) {
    if (const auto *Q = Elaborated->getQualifier())
      if (const auto *Qualifier = Q->getAsType()) {
        if (CheckTemplate(Qualifier->getAs<TemplateSpecializationType>())) {
          return true; // Case: enable_if< >::type.
        }
      }
  }
  return false;
}
AST_MATCHER_P(TemplateTypeParmDecl, hasDefaultArgument,
              clang::ast_matchers::internal::Matcher<QualType>, TypeMatcher) {
  return Node.hasDefaultArgument() &&
         TypeMatcher.matches(Node.getDefaultArgument(), Finder, Builder);
}
} // namespace

void ForwardingReferenceOverloadCheck::registerMatchers(MatchFinder *Finder) {
  auto ForwardingRefParm =
      parmVarDecl(
          hasType(qualType(rValueReferenceType(),
                           references(templateTypeParmType(hasDeclaration(
                               templateTypeParmDecl().bind("type-parm-decl")))),
                           unless(references(isConstQualified())))))
          .bind("parm-var");

  DeclarationMatcher FindOverload =
      cxxConstructorDecl(
          hasParameter(0, ForwardingRefParm),
          unless(hasAnyParameter(
              // No warning: enable_if as constructor parameter.
              parmVarDecl(hasType(isEnableIf())))),
          unless(hasParent(functionTemplateDecl(anyOf(
              // No warning: enable_if as type parameter.
              has(templateTypeParmDecl(hasDefaultArgument(isEnableIf()))),
              // No warning: enable_if as non-type template parameter.
              has(nonTypeTemplateParmDecl(
                  hasType(isEnableIf()),
                  anyOf(hasDescendant(cxxBoolLiteral()),
                        hasDescendant(cxxNullPtrLiteralExpr()),
                        hasDescendant(integerLiteral())))))))))
          .bind("ctor");
  Finder->addMatcher(FindOverload, this);
}

void ForwardingReferenceOverloadCheck::check(
    const MatchFinder::MatchResult &Result) {
  const auto *ParmVar = Result.Nodes.getNodeAs<ParmVarDecl>("parm-var");
  const auto *TypeParmDecl =
      Result.Nodes.getNodeAs<TemplateTypeParmDecl>("type-parm-decl");

  // Get the FunctionDecl and FunctionTemplateDecl containing the function
  // parameter.
  const auto *FuncForParam = dyn_cast<FunctionDecl>(ParmVar->getDeclContext());
  if (!FuncForParam)
    return;
  const FunctionTemplateDecl *FuncTemplate =
      FuncForParam->getDescribedFunctionTemplate();
  if (!FuncTemplate)
    return;

  // Check that the template type parameter belongs to the same function
  // template as the function parameter of that type. (This implies that type
  // deduction will happen on the type.)
  const TemplateParameterList *Params = FuncTemplate->getTemplateParameters();
  if (!llvm::is_contained(*Params, TypeParmDecl))
    return;

  // Every parameter after the first must have a default value.
  const auto *Ctor = Result.Nodes.getNodeAs<CXXConstructorDecl>("ctor");
  for (auto *Iter = Ctor->param_begin() + 1; Iter != Ctor->param_end();
       ++Iter) {
    if (!(*Iter)->hasDefaultArg())
      return;
  }
  bool EnabledCopy = false, DisabledCopy = false, EnabledMove = false,
       DisabledMove = false;
  for (const auto *OtherCtor : Ctor->getParent()->ctors()) {
    if (OtherCtor->isCopyOrMoveConstructor()) {
      if (OtherCtor->isDeleted() || OtherCtor->getAccess() == AS_private)
        (OtherCtor->isCopyConstructor() ? DisabledCopy : DisabledMove) = true;
      else
        (OtherCtor->isCopyConstructor() ? EnabledCopy : EnabledMove) = true;
    }
  }
  bool Copy = (!EnabledMove && !DisabledMove && !DisabledCopy) || EnabledCopy;
  bool Move = !DisabledMove || EnabledMove;
  if (!Copy && !Move)
    return;
  diag(Ctor->getLocation(),
       "constructor accepting a forwarding reference can "
       "hide the %select{copy|move|copy and move}0 constructor%s1")
      << (Copy && Move ? 2 : (Copy ? 0 : 1)) << Copy + Move;
  for (const auto *OtherCtor : Ctor->getParent()->ctors()) {
    if (OtherCtor->isCopyOrMoveConstructor() && !OtherCtor->isDeleted() &&
        OtherCtor->getAccess() != AS_private) {
      diag(OtherCtor->getLocation(),
           "%select{copy|move}0 constructor declared here", DiagnosticIDs::Note)
          << OtherCtor->isMoveConstructor();
    }
  }
}

} // namespace bugprone
} // namespace tidy
} // namespace clang
