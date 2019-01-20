//===--- DefaultArgumentsCheck.cpp - clang-tidy----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "DefaultArgumentsCheck.h"

using namespace clang::ast_matchers;

namespace clang {
namespace tidy {
namespace fuchsia {

void DefaultArgumentsCheck::registerMatchers(MatchFinder *Finder) {
  // Calling a function which uses default arguments is disallowed.
  Finder->addMatcher(cxxDefaultArgExpr().bind("stmt"), this);
  // Declaring default parameters is disallowed.
  Finder->addMatcher(parmVarDecl(hasDefaultArgument()).bind("decl"), this);
}

void DefaultArgumentsCheck::check(const MatchFinder::MatchResult &Result) {
  if (const auto *S =
          Result.Nodes.getNodeAs<CXXDefaultArgExpr>("stmt")) {
    diag(S->getUsedLocation(),
         "calling a function that uses a default argument is disallowed");
    diag(S->getParam()->getBeginLoc(), "default parameter was declared here",
         DiagnosticIDs::Note);
  } else if (const ParmVarDecl *D =
          Result.Nodes.getNodeAs<ParmVarDecl>("decl")) {
    SourceRange DefaultArgRange = D->getDefaultArgRange();

    if (DefaultArgRange.getEnd() != D->getEndLoc()) {
      return;
    } else if (DefaultArgRange.getBegin().isMacroID()) {
      diag(D->getBeginLoc(),
           "declaring a parameter with a default argument is disallowed");
    } else {
      SourceLocation StartLocation =
          D->getName().empty() ? D->getBeginLoc() : D->getLocation();

      SourceRange RemovalRange(Lexer::getLocForEndOfToken(
             StartLocation, 0,
             *Result.SourceManager,
             Result.Context->getLangOpts()
           ),
           DefaultArgRange.getEnd()
         );

      diag(D->getBeginLoc(),
           "declaring a parameter with a default argument is disallowed")
          << D << FixItHint::CreateRemoval(RemovalRange);
    }
  }
}

} // namespace fuchsia
} // namespace tidy
} // namespace clang
