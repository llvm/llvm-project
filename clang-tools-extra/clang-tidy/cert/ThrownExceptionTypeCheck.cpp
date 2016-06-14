//===--- ThrownExceptionTypeCheck.cpp - clang-tidy-------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "ThrownExceptionTypeCheck.h"
#include "../utils/Matchers.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"

using namespace clang::ast_matchers;

namespace clang {
namespace tidy {
namespace cert {

void ThrownExceptionTypeCheck::registerMatchers(MatchFinder *Finder) {
  if (!getLangOpts().CPlusPlus)
    return;

  Finder->addMatcher(
      cxxThrowExpr(has(ignoringParenImpCasts(
          cxxConstructExpr(hasDeclaration(cxxConstructorDecl(
                               isCopyConstructor(), unless(isNoThrow()))))
              .bind("expr")))),
      this);
}

void ThrownExceptionTypeCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *E = Result.Nodes.getNodeAs<Expr>("expr");
  diag(E->getExprLoc(),
       "thrown exception type is not nothrow copy constructible");
}

} // namespace cert
} // namespace tidy
} // namespace clang
