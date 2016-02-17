//===--- NonConstReferences.cpp - clang-tidy --------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "NonConstReferences.h"
#include "clang/AST/DeclBase.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"

using namespace clang::ast_matchers;

namespace clang {
namespace tidy {
namespace google {
namespace runtime {

void NonConstReferences::registerMatchers(MatchFinder *Finder) {
  Finder->addMatcher(
      parmVarDecl(
          unless(isInstantiated()),
          hasType(references(
              qualType(unless(isConstQualified())).bind("referenced_type"))),
          unless(hasType(rValueReferenceType())))
          .bind("param"),
      this);
}

void NonConstReferences::check(const MatchFinder::MatchResult &Result) {
  const auto *Parameter = Result.Nodes.getNodeAs<ParmVarDecl>("param");
  const auto *Function =
      dyn_cast_or_null<FunctionDecl>(Parameter->getParentFunctionOrMethod());

  if (Function == nullptr || Function->isImplicit())
    return;

  if (!Function->isCanonicalDecl())
    return;

  if (const CXXMethodDecl *Method = dyn_cast<CXXMethodDecl>(Function)) {
    // Don't warn on implementations of an interface using references.
    if (Method->begin_overridden_methods() != Method->end_overridden_methods())
      return;
    // Don't warn on lambdas, as they frequently have to conform to the
    // interface defined elsewhere.
    if (Method->getParent()->isLambda())
      return;
  }

  auto ReferencedType = *Result.Nodes.getNodeAs<QualType>("referenced_type");
  // Don't warn on function references, they shouldn't be constant.
  if (ReferencedType->isFunctionProtoType())
    return;

  // Don't warn on dependent types in templates.
  if (ReferencedType->isDependentType())
    return;

  if (Function->isOverloadedOperator()) {
    switch (Function->getOverloadedOperator()) {
      case clang::OO_LessLess:
      case clang::OO_PlusPlus:
      case clang::OO_MinusMinus:
      case clang::OO_PlusEqual:
      case clang::OO_MinusEqual:
      case clang::OO_StarEqual:
      case clang::OO_SlashEqual:
      case clang::OO_PercentEqual:
      case clang::OO_LessLessEqual:
      case clang::OO_GreaterGreaterEqual:
      case clang::OO_PipeEqual:
      case clang::OO_CaretEqual:
      case clang::OO_AmpEqual:
        // Don't warn on the first parameter of operator<<(Stream&, ...),
        // operator++, operator-- and operation+assignment operators.
        if (Function->getParamDecl(0) == Parameter)
          return;
        break;
      case clang::OO_GreaterGreater: {
        auto isNonConstRef = [](clang::QualType T) {
          return T->isReferenceType() &&
                 !T.getNonReferenceType().isConstQualified();
        };
        // Don't warn on parameters of stream extractors:
        //   Stream& operator>>(Stream&, Value&);
        // Both parameters should be non-const references by convention.
        if (isNonConstRef(Function->getParamDecl(0)->getType()) &&
            (Function->getNumParams() < 2 || // E.g. member operator>>.
             isNonConstRef(Function->getParamDecl(1)->getType())) &&
            isNonConstRef(Function->getReturnType()))
          return;
        break;
      }
      default:
        break;
    }
  }

  // Some functions use references to comply with established standards.
  if (Function->getDeclName().isIdentifier() && Function->getName() == "swap")
    return;

  // iostream parameters are typically passed by non-const reference.
  if (StringRef(ReferencedType.getAsString()).endswith("stream"))
    return;

  if (Parameter->getName().empty()) {
    diag(Parameter->getLocation(),
         "non-const reference parameter at index %0, "
         "make it const or use a pointer")
        << Parameter->getFunctionScopeIndex();
  } else {
    diag(Parameter->getLocation(),
         "non-const reference parameter '%0', make it const or use a pointer")
        << Parameter->getName();
  }
}

} // namespace runtime
} // namespace google
} // namespace tidy
} // namespace clang
