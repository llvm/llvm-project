//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "RedundantControlFlowCheck.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchersMacros.h"
#include "clang/Lex/Lexer.h"

using namespace clang::ast_matchers;

namespace clang::tidy::readability {

namespace {

AST_MATCHER_P(CompoundStmt, hasFinalStmt, StatementMatcher, InnerMatcher) {
  return !Node.body_empty() &&
         InnerMatcher.matches(*Node.body_back(), Finder, Builder);
}

} // namespace

static const char *const RedundantReturnDiag =
    "redundant return statement at the end "
    "of a function with a void return type";
static const char *const RedundantContinueDiag =
    "redundant continue statement at the "
    "end of loop statement";

static bool isLocationInMacroExpansion(const SourceManager &SM,
                                       SourceLocation Loc) {
  return SM.isMacroBodyExpansion(Loc) || SM.isMacroArgExpansion(Loc);
}

void RedundantControlFlowCheck::registerMatchers(MatchFinder *Finder) {
  Finder->addMatcher(
      functionDecl(isDefinition(), returns(voidType()),
                   hasBody(compoundStmt(hasFinalStmt(
                       returnStmt(unless(has(expr()))).bind("stmt"))))),
      this);
  Finder->addMatcher(mapAnyOf(forStmt, cxxForRangeStmt, whileStmt, doStmt)
                         .with(hasBody(compoundStmt(
                             hasFinalStmt(continueStmt().bind("stmt"))))),
                     this);
}

void RedundantControlFlowCheck::check(const MatchFinder::MatchResult &Result) {
  const auto &RedundantStmt = *Result.Nodes.getNodeAs<Stmt>("stmt");
  const SourceRange StmtRange = RedundantStmt.getSourceRange();
  const SourceManager &SM = *Result.SourceManager;

  if (isLocationInMacroExpansion(SM, StmtRange.getBegin()))
    return;

  const auto RemovedRange = CharSourceRange::getCharRange(
      StmtRange.getBegin(),
      Lexer::findLocationAfterToken(StmtRange.getEnd(), tok::semi, SM,
                                    getLangOpts(),
                                    /*SkipTrailingWhitespaceAndNewLine=*/true));

  diag(StmtRange.getBegin(), isa<ReturnStmt>(RedundantStmt)
                                 ? RedundantReturnDiag
                                 : RedundantContinueDiag)
      << FixItHint::CreateRemoval(RemovedRange);
}

} // namespace clang::tidy::readability
