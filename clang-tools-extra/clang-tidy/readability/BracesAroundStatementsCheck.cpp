//===--- BracesAroundStatementsCheck.cpp - clang-tidy ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "BracesAroundStatementsCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Lex/Lexer.h"

using namespace clang::ast_matchers;

namespace clang {
namespace tidy {
namespace readability {
namespace {

tok::TokenKind getTokenKind(SourceLocation Loc, const SourceManager &SM,
                            const ASTContext *Context) {
  Token Tok;
  SourceLocation Beginning =
      Lexer::GetBeginningOfToken(Loc, SM, Context->getLangOpts());
  const bool Invalid =
      Lexer::getRawToken(Beginning, Tok, SM, Context->getLangOpts());
  assert(!Invalid && "Expected a valid token.");

  if (Invalid)
    return tok::NUM_TOKENS;

  return Tok.getKind();
}

SourceLocation forwardSkipWhitespaceAndComments(SourceLocation Loc,
                                                const SourceManager &SM,
                                                const ASTContext *Context) {
  assert(Loc.isValid());
  for (;;) {
    while (isWhitespace(*SM.getCharacterData(Loc)))
      Loc = Loc.getLocWithOffset(1);

    tok::TokenKind TokKind = getTokenKind(Loc, SM, Context);
    if (TokKind == tok::NUM_TOKENS || TokKind != tok::comment)
      return Loc;

    // Fast-forward current token.
    Loc = Lexer::getLocForEndOfToken(Loc, 0, SM, Context->getLangOpts());
  }
}

SourceLocation findEndLocation(SourceLocation LastTokenLoc,
                               const SourceManager &SM,
                               const ASTContext *Context) {
  SourceLocation Loc =
      Lexer::GetBeginningOfToken(LastTokenLoc, SM, Context->getLangOpts());
  // Loc points to the beginning of the last (non-comment non-ws) token
  // before end or ';'.
  assert(Loc.isValid());
  bool SkipEndWhitespaceAndComments = true;
  tok::TokenKind TokKind = getTokenKind(Loc, SM, Context);
  if (TokKind == tok::NUM_TOKENS || TokKind == tok::semi ||
      TokKind == tok::r_brace) {
    // If we are at ";" or "}", we found the last token. We could use as well
    // `if (isa<NullStmt>(S))`, but it wouldn't work for nested statements.
    SkipEndWhitespaceAndComments = false;
  }

  Loc = Lexer::getLocForEndOfToken(Loc, 0, SM, Context->getLangOpts());
  // Loc points past the last token before end or after ';'.
  if (SkipEndWhitespaceAndComments) {
    Loc = forwardSkipWhitespaceAndComments(Loc, SM, Context);
    tok::TokenKind TokKind = getTokenKind(Loc, SM, Context);
    if (TokKind == tok::semi)
      Loc = Lexer::getLocForEndOfToken(Loc, 0, SM, Context->getLangOpts());
  }

  for (;;) {
    assert(Loc.isValid());
    while (isHorizontalWhitespace(*SM.getCharacterData(Loc))) {
      Loc = Loc.getLocWithOffset(1);
    }

    if (isVerticalWhitespace(*SM.getCharacterData(Loc))) {
      // EOL, insert brace before.
      break;
    }
    tok::TokenKind TokKind = getTokenKind(Loc, SM, Context);
    if (TokKind != tok::comment) {
      // Non-comment token, insert brace before.
      break;
    }

    SourceLocation TokEndLoc =
        Lexer::getLocForEndOfToken(Loc, 0, SM, Context->getLangOpts());
    SourceRange TokRange(Loc, TokEndLoc);
    StringRef Comment = Lexer::getSourceText(
        CharSourceRange::getTokenRange(TokRange), SM, Context->getLangOpts());
    if (Comment.startswith("/*") && Comment.find('\n') != StringRef::npos) {
      // Multi-line block comment, insert brace before.
      break;
    }
    // else: Trailing comment, insert brace after the newline.

    // Fast-forward current token.
    Loc = TokEndLoc;
  }
  return Loc;
}

} // namespace

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
  if (auto S = Result.Nodes.getNodeAs<ForStmt>("for")) {
    checkStmt(Result, S->getBody(), S->getRParenLoc());
  } else if (auto S = Result.Nodes.getNodeAs<CXXForRangeStmt>("for-range")) {
    checkStmt(Result, S->getBody(), S->getRParenLoc());
  } else if (auto S = Result.Nodes.getNodeAs<DoStmt>("do")) {
    checkStmt(Result, S->getBody(), S->getDoLoc(), S->getWhileLoc());
  } else if (auto S = Result.Nodes.getNodeAs<WhileStmt>("while")) {
    SourceLocation StartLoc = findRParenLoc(S, SM, Context);
    if (StartLoc.isInvalid())
      return;
    checkStmt(Result, S->getBody(), StartLoc);
  } else if (auto S = Result.Nodes.getNodeAs<IfStmt>("if")) {
    SourceLocation StartLoc = findRParenLoc(S, SM, Context);
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
                                           const ASTContext *Context) {
  // Skip macros.
  if (S->getBeginLoc().isMacroID())
    return SourceLocation();

  SourceLocation CondEndLoc = S->getCond()->getEndLoc();
  if (const DeclStmt *CondVar = S->getConditionVariableDeclStmt())
    CondEndLoc = CondVar->getEndLoc();

  if (!CondEndLoc.isValid()) {
    return SourceLocation();
  }

  SourceLocation PastCondEndLoc =
      Lexer::getLocForEndOfToken(CondEndLoc, 0, SM, Context->getLangOpts());
  if (PastCondEndLoc.isInvalid())
    return SourceLocation();
  SourceLocation RParenLoc =
      forwardSkipWhitespaceAndComments(PastCondEndLoc, SM, Context);
  if (RParenLoc.isInvalid())
    return SourceLocation();
  tok::TokenKind TokKind = getTokenKind(RParenLoc, SM, Context);
  if (TokKind != tok::r_paren)
    return SourceLocation();
  return RParenLoc;
}

/// Determine if the statement needs braces around it, and add them if it does.
/// Returns true if braces where added.
bool BracesAroundStatementsCheck::checkStmt(
    const MatchFinder::MatchResult &Result, const Stmt *S,
    SourceLocation InitialLoc, SourceLocation EndLocHint) {
  // 1) If there's a corresponding "else" or "while", the check inserts "} "
  // right before that token.
  // 2) If there's a multi-line block comment starting on the same line after
  // the location we're inserting the closing brace at, or there's a non-comment
  // token, the check inserts "\n}" right before that token.
  // 3) Otherwise the check finds the end of line (possibly after some block or
  // line comments) and inserts "\n}" right before that EOL.
  if (!S || isa<CompoundStmt>(S)) {
    // Already inside braces.
    return false;
  }

  if (!InitialLoc.isValid())
    return false;
  const SourceManager &SM = *Result.SourceManager;
  const ASTContext *Context = Result.Context;

  // Treat macros.
  CharSourceRange FileRange = Lexer::makeFileCharRange(
      CharSourceRange::getTokenRange(S->getSourceRange()), SM,
      Context->getLangOpts());
  if (FileRange.isInvalid())
    return false;

  // Convert InitialLoc to file location, if it's on the same macro expansion
  // level as the start of the statement. We also need file locations for
  // Lexer::getLocForEndOfToken working properly.
  InitialLoc = Lexer::makeFileCharRange(
                   CharSourceRange::getCharRange(InitialLoc, S->getBeginLoc()),
                   SM, Context->getLangOpts())
                   .getBegin();
  if (InitialLoc.isInvalid())
    return false;
  SourceLocation StartLoc =
      Lexer::getLocForEndOfToken(InitialLoc, 0, SM, Context->getLangOpts());

  // StartLoc points at the location of the opening brace to be inserted.
  SourceLocation EndLoc;
  std::string ClosingInsertion;
  if (EndLocHint.isValid()) {
    EndLoc = EndLocHint;
    ClosingInsertion = "} ";
  } else {
    const auto FREnd = FileRange.getEnd().getLocWithOffset(-1);
    EndLoc = findEndLocation(FREnd, SM, Context);
    ClosingInsertion = "\n}";
  }

  assert(StartLoc.isValid());
  assert(EndLoc.isValid());
  // Don't require braces for statements spanning less than certain number of
  // lines.
  if (ShortStatementLines && !ForceBracesStmts.erase(S)) {
    unsigned StartLine = SM.getSpellingLineNumber(StartLoc);
    unsigned EndLine = SM.getSpellingLineNumber(EndLoc);
    if (EndLine - StartLine < ShortStatementLines)
      return false;
  }

  auto Diag = diag(StartLoc, "statement should be inside braces");
  Diag << FixItHint::CreateInsertion(StartLoc, " {")
       << FixItHint::CreateInsertion(EndLoc, ClosingInsertion);
  return true;
}

void BracesAroundStatementsCheck::onEndOfTranslationUnit() {
  ForceBracesStmts.clear();
}

} // namespace readability
} // namespace tidy
} // namespace clang
