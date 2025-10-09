//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "BracesAroundStatementsCheck.h"
#include "../utils/BracesAroundStatement.h"
#include "../utils/LexerUtils.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Lex/Lexer.h"

using namespace clang::ast_matchers;

namespace clang::tidy::readability {

static tok::TokenKind getTokenKind(SourceLocation Loc, const SourceManager &SM,
                                   const LangOptions &LangOpts) {
  Token Tok;
  SourceLocation Beginning = Lexer::GetBeginningOfToken(Loc, SM, LangOpts);
  const bool Invalid = Lexer::getRawToken(Beginning, Tok, SM, LangOpts);
  assert(!Invalid && "Expected a valid token.");

  if (Invalid)
    return tok::NUM_TOKENS;

  return Tok.getKind();
}

static SourceLocation
forwardSkipWhitespaceAndComments(SourceLocation Loc, const SourceManager &SM,
                                 const LangOptions &LangOpts) {
  assert(Loc.isValid());
  for (;;) {
    while (isWhitespace(*SM.getCharacterData(Loc)))
      Loc = Loc.getLocWithOffset(1);

    tok::TokenKind TokKind = getTokenKind(Loc, SM, LangOpts);
    if (TokKind != tok::comment)
      return Loc;

    // Fast-forward current token.
    Loc = Lexer::getLocForEndOfToken(Loc, 0, SM, LangOpts);
  }
}

BracesAroundStatementsCheck::BracesAroundStatementsCheck(
    StringRef Name, ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context),
      // Always add braces by default.
      ShortStatementLines(Options.get("ShortStatementLines", 0U)) {}

void BracesAroundStatementsCheck::storeOptions(
    ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "ShortStatementLines", ShortStatementLines);
}

void BracesAroundStatementsCheck::registerMatchers(MatchFinder *Finder) {
  Finder->addMatcher(ifStmt().bind("if"), this);
  Finder->addMatcher(whileStmt().bind("while"), this);
  Finder->addMatcher(doStmt().bind("do"), this);
  Finder->addMatcher(forStmt().bind("for"), this);
  Finder->addMatcher(cxxForRangeStmt().bind("for-range"), this);
}

void BracesAroundStatementsCheck::check(
    const MatchFinder::MatchResult &Result) {
  const SourceManager &SM = *Result.SourceManager;
  const ASTContext *Context = Result.Context;

  // Get location of closing parenthesis or 'do' to insert opening brace.
  if (const auto *S = Result.Nodes.getNodeAs<ForStmt>("for")) {
    checkStmt(Result, S->getBody(), S->getRParenLoc());
  } else if (const auto *S =
                 Result.Nodes.getNodeAs<CXXForRangeStmt>("for-range")) {
    checkStmt(Result, S->getBody(), S->getRParenLoc());
  } else if (const auto *S = Result.Nodes.getNodeAs<DoStmt>("do")) {
    checkStmt(Result, S->getBody(), S->getDoLoc(), S->getWhileLoc());
  } else if (const auto *S = Result.Nodes.getNodeAs<WhileStmt>("while")) {
    SourceLocation StartLoc = findRParenLoc(S, SM, Context->getLangOpts());
    if (StartLoc.isInvalid())
      return;
    checkStmt(Result, S->getBody(), StartLoc);
  } else if (const auto *S = Result.Nodes.getNodeAs<IfStmt>("if")) {
    // "if consteval" always has braces.
    if (S->isConsteval())
      return;

    SourceLocation StartLoc = findRParenLoc(S, SM, Context->getLangOpts());
    if (StartLoc.isInvalid())
      return;
    if (ForceBracesStmts.erase(S))
      ForceBracesStmts.insert(S->getThen());
    bool BracedIf = checkStmt(Result, S->getThen(), StartLoc, S->getElseLoc());
    const Stmt *Else = S->getElse();
    if (Else && BracedIf)
      ForceBracesStmts.insert(Else);
    if (Else && !isa<IfStmt>(Else)) {
      // Omit 'else if' statements here, they will be handled directly.
      checkStmt(Result, Else, S->getElseLoc());
    }
  } else {
    llvm_unreachable("Invalid match");
  }
}

/// Find location of right parenthesis closing condition.
template <typename IfOrWhileStmt>
SourceLocation
BracesAroundStatementsCheck::findRParenLoc(const IfOrWhileStmt *S,
                                           const SourceManager &SM,
                                           const LangOptions &LangOpts) {
  // Skip macros.
  if (S->getBeginLoc().isMacroID())
    return {};

  SourceLocation CondEndLoc = S->getCond()->getEndLoc();
  if (const DeclStmt *CondVar = S->getConditionVariableDeclStmt())
    CondEndLoc = CondVar->getEndLoc();

  if (!CondEndLoc.isValid()) {
    return {};
  }

  SourceLocation PastCondEndLoc =
      Lexer::getLocForEndOfToken(CondEndLoc, 0, SM, LangOpts);
  if (PastCondEndLoc.isInvalid())
    return {};
  SourceLocation RParenLoc =
      forwardSkipWhitespaceAndComments(PastCondEndLoc, SM, LangOpts);
  if (RParenLoc.isInvalid())
    return {};
  tok::TokenKind TokKind = getTokenKind(RParenLoc, SM, LangOpts);
  if (TokKind != tok::r_paren)
    return {};
  return RParenLoc;
}

/// Determine if the statement needs braces around it, and add them if it does.
/// Returns true if braces where added.
bool BracesAroundStatementsCheck::checkStmt(
    const MatchFinder::MatchResult &Result, const Stmt *S,
    SourceLocation StartLoc, SourceLocation EndLocHint) {
  while (const auto *AS = dyn_cast<AttributedStmt>(S))
    S = AS->getSubStmt();

  const auto BraceInsertionHints = utils::getBraceInsertionsHints(
      S, Result.Context->getLangOpts(), *Result.SourceManager, StartLoc,
      EndLocHint);
  if (BraceInsertionHints) {
    if (ShortStatementLines && !ForceBracesStmts.erase(S) &&
        BraceInsertionHints.resultingCompoundLineExtent(*Result.SourceManager) <
            ShortStatementLines)
      return false;
    auto Diag = diag(BraceInsertionHints.DiagnosticPos,
                     "statement should be inside braces");
    if (BraceInsertionHints.offersFixIts())
      Diag << BraceInsertionHints.openingBraceFixIt()
           << BraceInsertionHints.closingBraceFixIt();
  }
  return true;
}

void BracesAroundStatementsCheck::onEndOfTranslationUnit() {
  ForceBracesStmts.clear();
}

} // namespace clang::tidy::readability
