//===--- ComparisonInTempFailureRetryCheck.cpp - clang-tidy----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "../utils/Matchers.h"
#include "ComparisonInTempFailureRetryCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Lex/Lexer.h"

using namespace clang::ast_matchers;

namespace clang {
namespace tidy {
namespace android {

namespace {
AST_MATCHER(BinaryOperator, isRHSATempFailureRetryArg) {
  if (!Node.getLocStart().isMacroID())
    return false;

  const SourceManager &SM = Finder->getASTContext().getSourceManager();
  if (!SM.isMacroArgExpansion(Node.getRHS()->IgnoreParenCasts()->getLocStart()))
    return false;

  const LangOptions &Opts = Finder->getASTContext().getLangOpts();
  SourceLocation LocStart = Node.getLocStart();
  while (LocStart.isMacroID()) {
    SourceLocation Invocation = SM.getImmediateMacroCallerLoc(LocStart);
    Token Tok;
    if (!Lexer::getRawToken(SM.getSpellingLoc(Invocation), Tok, SM, Opts,
                            /*IgnoreWhiteSpace=*/true)) {
      if (Tok.getKind() == tok::raw_identifier &&
          Tok.getRawIdentifier() == "TEMP_FAILURE_RETRY")
        return true;
    }

    LocStart = Invocation;
  }
  return false;
}
} // namespace

void ComparisonInTempFailureRetryCheck::registerMatchers(MatchFinder *Finder) {
  // Both glibc's and Bionic's TEMP_FAILURE_RETRY macros structurally look like:
  //
  // #define TEMP_FAILURE_RETRY(x) ({ \
  //    typeof(x) y; \
  //    do y = (x); \
  //    while (y == -1 && errno == EINTR); \
  //    y; \
  // })
  //
  // (glibc uses `long int` instead of `typeof(x)` for the type of y).
  //
  // It's unclear how to walk up the AST from inside the expansion of `x`, and
  // we need to not complain about things like TEMP_FAILURE_RETRY(foo(x == 1)),
  // so we just match the assignment of `y = (x)` and inspect `x` from there.
  Finder->addMatcher(
      binaryOperator(
          hasOperatorName("="),
          hasRHS(ignoringParenCasts(
              binaryOperator(matchers::isComparisonOperator()).bind("binop"))),
          isRHSATempFailureRetryArg()),
      this);
}

void ComparisonInTempFailureRetryCheck::check(
    const MatchFinder::MatchResult &Result) {
  const auto &BinOp = *Result.Nodes.getNodeAs<BinaryOperator>("binop");
  diag(BinOp.getOperatorLoc(), "top-level comparison in TEMP_FAILURE_RETRY");

  // FIXME: FixIts would be nice, but potentially nontrivial when nested macros
  // happen, e.g. `TEMP_FAILURE_RETRY(IS_ZERO(foo()))`
}

} // namespace android
} // namespace tidy
} // namespace clang
