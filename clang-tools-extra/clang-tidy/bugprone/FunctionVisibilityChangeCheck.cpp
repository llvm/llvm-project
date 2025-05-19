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

namespace clang::tidy {

template <>
struct OptionEnumMapping<bugprone::FunctionVisibilityChangeCheck::ChangeKind> {
  static llvm::ArrayRef<
      std::pair<bugprone::FunctionVisibilityChangeCheck::ChangeKind, StringRef>>
  getEnumMapping() {
    static constexpr std::pair<
        bugprone::FunctionVisibilityChangeCheck::ChangeKind, StringRef>
        Mapping[] = {
            {bugprone::FunctionVisibilityChangeCheck::ChangeKind::Any, "any"},
            {bugprone::FunctionVisibilityChangeCheck::ChangeKind::Widening,
             "widening"},
            {bugprone::FunctionVisibilityChangeCheck::ChangeKind::Narrowing,
             "narrowing"},
        };
    return {Mapping};
  }
};

namespace bugprone {

FunctionVisibilityChangeCheck::FunctionVisibilityChangeCheck(
    StringRef Name, ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context),
      DetectVisibilityChange(
          Options.get("DisallowedVisibilityChange", ChangeKind::Any)) {}

void FunctionVisibilityChangeCheck::storeOptions(
    ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "DisallowedVisibilityChange", DetectVisibilityChange);
}

void FunctionVisibilityChangeCheck::registerMatchers(MatchFinder *Finder) {
  Finder->addMatcher(
      cxxMethodDecl(
          isVirtual(),
          ofClass(
              cxxRecordDecl(unless(isExpansionInSystemHeader())).bind("class")),
          forEachOverridden(cxxMethodDecl(ofClass(cxxRecordDecl().bind("base")))
                                .bind("base_func")))
          .bind("func"),
      this);
}

void FunctionVisibilityChangeCheck::check(
    const MatchFinder::MatchResult &Result) {
  const auto *MatchedFunction = Result.Nodes.getNodeAs<FunctionDecl>("func");
  if (!MatchedFunction->isCanonicalDecl())
    return;

  const auto *ParentClass = Result.Nodes.getNodeAs<CXXRecordDecl>("class");
  const auto *OverriddenFunction =
      Result.Nodes.getNodeAs<FunctionDecl>("base_func");
  const auto *BaseClass = Result.Nodes.getNodeAs<CXXRecordDecl>("base");

  AccessSpecifier ActualAccess = MatchedFunction->getAccess();
  AccessSpecifier OverriddenAccess = OverriddenFunction->getAccess();

  CXXBasePaths Paths;
  if (!ParentClass->isDerivedFrom(BaseClass, Paths))
    return;
  const CXXBaseSpecifier *InheritanceWithStrictVisibility = nullptr;
  for (const CXXBasePath &Path : Paths) {
    for (const CXXBasePathElement &Elem : Path) {
      if (Elem.Base->getAccessSpecifier() > OverriddenAccess) {
        OverriddenAccess = Elem.Base->getAccessSpecifier();
        InheritanceWithStrictVisibility = Elem.Base;
      }
    }
  }

  if (ActualAccess != OverriddenAccess) {
    if (DetectVisibilityChange == ChangeKind::Widening &&
        ActualAccess > OverriddenAccess)
      return;
    if (DetectVisibilityChange == ChangeKind::Narrowing &&
        ActualAccess < OverriddenAccess)
      return;

    if (InheritanceWithStrictVisibility) {
      diag(MatchedFunction->getLocation(),
           "visibility of function %0 is changed from %1 (through %1 "
           "inheritance of class %2) to %3")
          << MatchedFunction << OverriddenAccess
          << InheritanceWithStrictVisibility->getType() << ActualAccess;
      diag(InheritanceWithStrictVisibility->getBeginLoc(),
           "%0 is inherited as %1 here", DiagnosticIDs::Note)
          << InheritanceWithStrictVisibility->getType() << OverriddenAccess;
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

} // namespace bugprone

} // namespace clang::tidy
