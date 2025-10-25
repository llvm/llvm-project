//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "NoInternalDependenciesCheck.h"
#include "AbseilMatcher.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"

using namespace clang::ast_matchers;

namespace clang::tidy::abseil {

void NoInternalDependenciesCheck::registerMatchers(MatchFinder *Finder) {
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

void NoInternalDependenciesCheck::check(
    const MatchFinder::MatchResult &Result) {
  const auto *InternalDependency =
      Result.Nodes.getNodeAs<NestedNameSpecifierLoc>("InternalDep");

  SourceLocation LocAtFault =
      Result.SourceManager->getSpellingLoc(InternalDependency->getBeginLoc());

  if (!LocAtFault.isValid())
    return;

  diag(LocAtFault,
       "do not reference any 'internal' namespaces; those implementation "
       "details are reserved to Abseil");
}

} // namespace clang::tidy::abseil
