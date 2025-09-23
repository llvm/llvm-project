//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ComparisonInTempFailureRetryCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Lex/Lexer.h"

using namespace clang::ast_matchers;

namespace clang::tidy::android {

ComparisonInTempFailureRetryCheck::ComparisonInTempFailureRetryCheck(
    StringRef Name, ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context),
      RawRetryList(Options.get("RetryMacros", "TEMP_FAILURE_RETRY")) {
  StringRef(RawRetryList).split(RetryMacros, ",", -1, false);
}

void ComparisonInTempFailureRetryCheck::storeOptions(
    ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "RetryMacros", RawRetryList);
}

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
      binaryOperator(hasOperatorName("="),
                     hasRHS(ignoringParenCasts(
                         binaryOperator(isComparisonOperator()).bind("inner"))))
          .bind("outer"),
      this);
}

void ComparisonInTempFailureRetryCheck::check(
    const MatchFinder::MatchResult &Result) {
  StringRef RetryMacroName;
  const auto &Node = *Result.Nodes.getNodeAs<BinaryOperator>("outer");
  if (!Node.getBeginLoc().isMacroID())
    return;

  const SourceManager &SM = *Result.SourceManager;
  if (!SM.isMacroArgExpansion(Node.getRHS()->IgnoreParenCasts()->getBeginLoc()))
    return;

  const LangOptions &Opts = Result.Context->getLangOpts();
  SourceLocation LocStart = Node.getBeginLoc();
  while (LocStart.isMacroID()) {
    SourceLocation Invocation = SM.getImmediateMacroCallerLoc(LocStart);
    Token Tok;
    if (!Lexer::getRawToken(SM.getSpellingLoc(Invocation), Tok, SM, Opts,
                            /*IgnoreWhiteSpace=*/true)) {
      if (Tok.getKind() == tok::raw_identifier &&
          llvm::is_contained(RetryMacros, Tok.getRawIdentifier())) {
        RetryMacroName = Tok.getRawIdentifier();
        break;
      }
    }

    LocStart = Invocation;
  }
  if (RetryMacroName.empty())
    return;

  const auto &Inner = *Result.Nodes.getNodeAs<BinaryOperator>("inner");
  diag(Inner.getOperatorLoc(), "top-level comparison in %0") << RetryMacroName;

  // FIXME: FixIts would be nice, but potentially nontrivial when nested macros
  // happen, e.g. `TEMP_FAILURE_RETRY(IS_ZERO(foo()))`
}

} // namespace clang::tidy::android
