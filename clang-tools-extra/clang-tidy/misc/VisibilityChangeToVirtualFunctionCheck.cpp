//===--- VisibilityChangeToVirtualFunctionCheck.cpp - clang-tidy
//-------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "VisibilityChangeToVirtualFunctionCheck.h"
#include "../utils/Matchers.h"
#include "../utils/OptionsUtils.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"

using namespace clang::ast_matchers;
using namespace clang;

namespace {
AST_MATCHER(NamedDecl, isOperatorDecl) {
  DeclarationName::NameKind NK = Node.getDeclName().getNameKind();
  return NK != DeclarationName::Identifier &&
         NK != DeclarationName::CXXConstructorName &&
         NK != DeclarationName::CXXDestructorName;
}
} // namespace

namespace clang::tidy {

template <>
struct OptionEnumMapping<
    misc::VisibilityChangeToVirtualFunctionCheck::ChangeKind> {
  static llvm::ArrayRef<std::pair<
      misc::VisibilityChangeToVirtualFunctionCheck::ChangeKind, StringRef>>
  getEnumMapping() {
    static constexpr std::pair<
        misc::VisibilityChangeToVirtualFunctionCheck::ChangeKind, StringRef>
        Mapping[] = {
            {misc::VisibilityChangeToVirtualFunctionCheck::ChangeKind::Any,
             "any"},
            {misc::VisibilityChangeToVirtualFunctionCheck::ChangeKind::Widening,
             "widening"},
            {misc::VisibilityChangeToVirtualFunctionCheck::ChangeKind::
                 Narrowing,
             "narrowing"},
        };
    return {Mapping};
  }
};

namespace misc {

VisibilityChangeToVirtualFunctionCheck::VisibilityChangeToVirtualFunctionCheck(
    StringRef Name, ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context),
      DetectVisibilityChange(
          Options.get("DisallowedVisibilityChange", ChangeKind::Any)),
      CheckDestructors(Options.get("CheckDestructors", false)),
      CheckOperators(Options.get("CheckOperators", false)),
      IgnoredFunctions(utils::options::parseStringList(
          Options.get("IgnoredFunctions", ""))) {}

void VisibilityChangeToVirtualFunctionCheck::storeOptions(
    ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "DisallowedVisibilityChange", DetectVisibilityChange);
  Options.store(Opts, "CheckDestructors", CheckDestructors);
  Options.store(Opts, "CheckOperators", CheckOperators);
  Options.store(Opts, "IgnoredFunctions",
                utils::options::serializeStringList(IgnoredFunctions));
}

void VisibilityChangeToVirtualFunctionCheck::registerMatchers(
    MatchFinder *Finder) {
  auto IgnoredDecl =
      namedDecl(matchers::matchesAnyListedName(IgnoredFunctions));
  auto FilterDestructors =
      CheckDestructors ? decl() : decl(unless(cxxDestructorDecl()));
  auto FilterOperators =
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

void VisibilityChangeToVirtualFunctionCheck::check(
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

} // namespace misc

} // namespace clang::tidy
