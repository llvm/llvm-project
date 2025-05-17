//===--- FunctionVisibilityChangeCheck.cpp - clang-tidy -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "FunctionVisibilityChangeCheck.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"

using namespace clang::ast_matchers;

namespace clang::tidy::bugprone {

void FunctionVisibilityChangeCheck::registerMatchers(MatchFinder *Finder) {
  Finder->addMatcher(
      cxxMethodDecl(
          ofClass(cxxRecordDecl().bind("class")),
          forEachOverridden(cxxMethodDecl(ofClass(cxxRecordDecl().bind("base")))
                                .bind("base_func")))
          .bind("func"),
      this);
}

void FunctionVisibilityChangeCheck::check(
    const MatchFinder::MatchResult &Result) {
  const auto *MatchedFunction = Result.Nodes.getNodeAs<FunctionDecl>("func");
  const auto *ParentClass = Result.Nodes.getNodeAs<CXXRecordDecl>("class");
  const auto *OverriddenFunction =
      Result.Nodes.getNodeAs<FunctionDecl>("base_func");
  const auto *BaseClass = Result.Nodes.getNodeAs<CXXRecordDecl>("base");

  if (!MatchedFunction->isCanonicalDecl())
    return;

  AccessSpecifier ActualAccess = MatchedFunction->getAccess();
  AccessSpecifier OverriddenAccess = OverriddenFunction->getAccess();

  CXXBasePaths Paths;
  if (!ParentClass->isDerivedFrom(BaseClass, Paths))
    return;
  const CXXBaseSpecifier *InheritanceWithStrictVisibility = nullptr;
  for (const CXXBasePath &Path : Paths) {
    for (auto Elem : Path) {
      if (Elem.Base->getAccessSpecifier() > OverriddenAccess) {
        OverriddenAccess = Elem.Base->getAccessSpecifier();
        InheritanceWithStrictVisibility = Elem.Base;
      }
    }
  }

  if (ActualAccess != OverriddenAccess) {
    if (InheritanceWithStrictVisibility) {
      diag(MatchedFunction->getLocation(),
           "visibility of function %0 is changed from %1 (through %1 "
           "inheritance of class %2) to %3")
          << MatchedFunction << OverriddenAccess
          << InheritanceWithStrictVisibility->getType() << ActualAccess;
      diag(InheritanceWithStrictVisibility->getBeginLoc(),
           "this inheritance would make %0 %1", DiagnosticIDs::Note)
          << MatchedFunction << OverriddenAccess;
    } else {
      diag(MatchedFunction->getLocation(),
           "visibility of function %0 is changed from %1 in class %2 to %3")
          << MatchedFunction << OverriddenAccess << BaseClass << ActualAccess;
    }
    diag(OverriddenFunction->getLocation(), "function declared here as %0",
         DiagnosticIDs::Note)
        << OverriddenFunction->getAccess();
  }
}

} // namespace clang::tidy::bugprone
