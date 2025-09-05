//===--- OverrideWithDifferentVisibilityCheck.cpp - clang-tidy ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "OverrideWithDifferentVisibilityCheck.h"
#include "../utils/Matchers.h"
#include "../utils/OptionsUtils.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"

using namespace clang::ast_matchers;
using namespace clang;

namespace {

AST_MATCHER(NamedDecl, isOperatorDecl) {
  DeclarationName::NameKind const NK = Node.getDeclName().getNameKind();
  return NK != DeclarationName::Identifier &&
         NK != DeclarationName::CXXConstructorName &&
         NK != DeclarationName::CXXDestructorName;
}

} // namespace

namespace clang::tidy {

template <>
struct OptionEnumMapping<
    misc::OverrideWithDifferentVisibilityCheck::ChangeKind> {
  static llvm::ArrayRef<std::pair<
      misc::OverrideWithDifferentVisibilityCheck::ChangeKind, StringRef>>
  getEnumMapping() {
    static constexpr std::pair<
        misc::OverrideWithDifferentVisibilityCheck::ChangeKind, StringRef>
        Mapping[] = {
            {misc::OverrideWithDifferentVisibilityCheck::ChangeKind::Any,
             "any"},
            {misc::OverrideWithDifferentVisibilityCheck::ChangeKind::Widening,
             "widening"},
            {misc::OverrideWithDifferentVisibilityCheck::ChangeKind::Narrowing,
             "narrowing"},
        };
    return {Mapping};
  }
};

namespace misc {

OverrideWithDifferentVisibilityCheck::OverrideWithDifferentVisibilityCheck(
    StringRef Name, ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context),
      DetectVisibilityChange(
          Options.get("DisallowedVisibilityChange", ChangeKind::Any)),
      CheckDestructors(Options.get("CheckDestructors", false)),
      CheckOperators(Options.get("CheckOperators", false)),
      IgnoredFunctions(utils::options::parseStringList(
          Options.get("IgnoredFunctions", ""))) {}

void OverrideWithDifferentVisibilityCheck::storeOptions(
    ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "DisallowedVisibilityChange", DetectVisibilityChange);
  Options.store(Opts, "CheckDestructors", CheckDestructors);
  Options.store(Opts, "CheckOperators", CheckOperators);
  Options.store(Opts, "IgnoredFunctions",
                utils::options::serializeStringList(IgnoredFunctions));
}

void OverrideWithDifferentVisibilityCheck::registerMatchers(
    MatchFinder *Finder) {
  const auto IgnoredDecl =
      namedDecl(matchers::matchesAnyListedName(IgnoredFunctions));
  const auto FilterDestructors =
      CheckDestructors ? decl() : decl(unless(cxxDestructorDecl()));
  const auto FilterOperators =
      CheckOperators ? namedDecl() : namedDecl(unless(isOperatorDecl()));
  Finder->addMatcher(
      cxxMethodDecl(
          isVirtual(), FilterDestructors, FilterOperators,
          ofClass(
              cxxRecordDecl(unless(isExpansionInSystemHeader())).bind("class")),
          forEachOverridden(cxxMethodDecl(ofClass(cxxRecordDecl().bind("base")),
                                          unless(IgnoredDecl))
                                .bind("base_func")))
          .bind("func"),
      this);
}

void OverrideWithDifferentVisibilityCheck::check(
    const MatchFinder::MatchResult &Result) {
  const auto *const MatchedFunction =
      Result.Nodes.getNodeAs<FunctionDecl>("func");
  if (!MatchedFunction->isCanonicalDecl())
    return;

  const auto *const ParentClass =
      Result.Nodes.getNodeAs<CXXRecordDecl>("class");
  const auto *const BaseClass = Result.Nodes.getNodeAs<CXXRecordDecl>("base");
  CXXBasePaths Paths;
  if (!ParentClass->isDerivedFrom(BaseClass, Paths))
    return;

  const auto *const OverriddenFunction =
      Result.Nodes.getNodeAs<FunctionDecl>("base_func");
  AccessSpecifier const ActualAccess = MatchedFunction->getAccess();
  AccessSpecifier OverriddenAccess = OverriddenFunction->getAccess();

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

} // namespace misc

} // namespace clang::tidy
