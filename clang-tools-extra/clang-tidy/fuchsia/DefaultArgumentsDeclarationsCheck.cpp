//===--- DefaultArgumentsDeclarationsCheck.cpp - clang-tidy ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "DefaultArgumentsDeclarationsCheck.h"
#include "clang/Lex/Lexer.h"

using namespace clang::ast_matchers;

namespace clang {
namespace tidy {
namespace fuchsia {

void DefaultArgumentsDeclarationsCheck::registerMatchers(MatchFinder *Finder) {
  // Declaring default parameters is disallowed.
  Finder->addMatcher(parmVarDecl(hasDefaultArgument()).bind("decl"), this);
}

void DefaultArgumentsDeclarationsCheck::check(
    const MatchFinder::MatchResult &Result) {
  const auto *D = Result.Nodes.getNodeAs<ParmVarDecl>("decl");
  if (!D)
    return;

  SourceRange DefaultArgRange = D->getDefaultArgRange();

  if (DefaultArgRange.getEnd() != D->getEndLoc())
    return;

  if (DefaultArgRange.getBegin().isMacroID()) {
    diag(D->getBeginLoc(),
         "declaring a parameter with a default argument is disallowed");
    return;
  }

  SourceLocation StartLocation =
      D->getName().empty() ? D->getBeginLoc() : D->getLocation();

  SourceRange RemovalRange(
      Lexer::getLocForEndOfToken(StartLocation, 0, *Result.SourceManager,
                                 Result.Context->getLangOpts()),
      DefaultArgRange.getEnd());

  diag(D->getBeginLoc(),
       "declaring a parameter with a default argument is disallowed")
      << D << FixItHint::CreateRemoval(RemovalRange);
}

} // namespace fuchsia
} // namespace tidy
} // namespace clang
