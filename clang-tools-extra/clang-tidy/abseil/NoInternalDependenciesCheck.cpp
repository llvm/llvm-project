//===--- NoInternalDependenciesCheck.cpp - clang-tidy----------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "NoInternalDependenciesCheck.h"
#include "AbseilMatcher.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"

using namespace clang::ast_matchers;

namespace clang {
namespace tidy {
namespace abseil {

void NoInternalDependenciesCheck::registerMatchers(MatchFinder *Finder) {
  if (!getLangOpts().CPlusPlus)
    return;

  // TODO: refactor matcher to be configurable or just match on any internal
  // access from outside the enclosing namespace.

  Finder->addMatcher(
      nestedNameSpecifierLoc(loc(specifiesNamespace(namespaceDecl(
                                 matchesName("internal"),
                                 hasParent(namespaceDecl(hasName("absl")))))),
                             unless(isInAbseilFile()))
          .bind("InternalDep"),
      this);
}

void NoInternalDependenciesCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *InternalDependency =
      Result.Nodes.getNodeAs<NestedNameSpecifierLoc>("InternalDep");

  diag(InternalDependency->getBeginLoc(),
       "do not reference any 'internal' namespaces; those implementation "
       "details are reserved to Abseil");
}

} // namespace abseil
} // namespace tidy
} // namespace clang
