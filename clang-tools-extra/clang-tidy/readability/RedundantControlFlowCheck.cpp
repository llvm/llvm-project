//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "RedundantControlFlowCheck.h"
#include "clang/AST/Stmt.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Lex/Lexer.h"

using namespace clang::ast_matchers;

namespace clang::tidy::readability {

static constexpr StringRef RedundantReturnDiag =
    "redundant return statement at the end "
    "of a function with a void return type";
static constexpr StringRef RedundantContinueDiag =
    "redundant continue statement at the "
    "end of loop statement";

void RedundantControlFlowCheck::registerMatchers(MatchFinder *Finder) {
  Finder->addMatcher(functionDecl(returns(voidType()),
                                  hasBody(compoundStmt().bind("return-body"))),
                     this);
  Finder->addMatcher(mapAnyOf(forStmt, cxxForRangeStmt, whileStmt, doStmt)
                         .with(hasBody(stmt().bind("continue-body"))),
                     this);
}

void RedundantControlFlowCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *Body = Result.Nodes.getNodeAs<Stmt>("return-body");
  const bool IsReturn = Body != nullptr;
  if (!Body)
    Body = Result.Nodes.getNodeAs<Stmt>("continue-body");
  checkTrailingStmt(Result, Body, IsReturn, /*IsCompoundChild=*/false);
}

void RedundantControlFlowCheck::checkTrailingStmt(
    const MatchFinder::MatchResult &Result, const Stmt *S, bool IsReturn,
    bool IsCompoundChild) {
  if (const auto *Block = dyn_cast<CompoundStmt>(S)) {
    if (!Block->body_empty())
      checkTrailingStmt(Result, Block->body_back(), IsReturn,
                        /*IsCompoundChild=*/true);
    return;
  }
  if (const auto *If = dyn_cast<IfStmt>(S)) {
    checkTrailingStmt(Result, If->getThen(), IsReturn,
                      /*IsCompoundChild=*/false);
    if (const Stmt *Else = If->getElse())
      checkTrailingStmt(Result, Else, IsReturn, /*IsCompoundChild=*/false);
    return;
  }

  if (IsReturn) {
    const auto *Return = dyn_cast<ReturnStmt>(S);
    if (!Return || Return->getRetValue())
      return;
  } else if (!isa<ContinueStmt>(S)) {
    return;
  }

  const SourceRange StmtRange = S->getSourceRange();

  if (StmtRange.getBegin().isMacroID())
    return;

  const auto Diag =
      diag(StmtRange.getBegin(),
           IsReturn ? RedundantReturnDiag : RedundantContinueDiag);
  if (!IsCompoundChild) {
    // Keep a null statement as the body of an unbraced branch or loop.
    Diag << FixItHint::CreateRemoval(CharSourceRange::getTokenRange(StmtRange));
    return;
  }

  const auto RemovedRange = CharSourceRange::getCharRange(
      StmtRange.getBegin(),
      Lexer::findLocationAfterToken(StmtRange.getEnd(), tok::semi,
                                    *Result.SourceManager, getLangOpts(),
                                    /*SkipTrailingWhitespaceAndNewLine=*/true));

  Diag << FixItHint::CreateRemoval(RemovedRange);
}

} // namespace clang::tidy::readability
