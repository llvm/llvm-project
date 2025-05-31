//===--- BracesAroundStatement.cpp - clang-tidy -------- ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file provides utilities to put braces around a statement.
///
//===----------------------------------------------------------------------===//

#include "BracesAroundStatement.h"
#include "../utils/LexerUtils.h"
#include "LexerUtils.h"
#include "clang/AST/ASTContext.h"
#include "clang/Basic/CharInfo.h"
#include "clang/Basic/LangOptions.h"
#include "clang/Lex/Lexer.h"

namespace clang::tidy::utils {

BraceInsertionHints::operator bool() const { return DiagnosticPos.isValid(); }

bool BraceInsertionHints::offersFixIts() const {
  return OpeningBracePos.isValid() && ClosingBracePos.isValid();
}

unsigned BraceInsertionHints::resultingCompoundLineExtent(
    const SourceManager &SourceMgr) const {
  return SourceMgr.getSpellingLineNumber(ClosingBracePos) -
         SourceMgr.getSpellingLineNumber(OpeningBracePos);
}

FixItHint BraceInsertionHints::openingBraceFixIt() const {
  return OpeningBracePos.isValid()
             ? FixItHint::CreateInsertion(OpeningBracePos, " {")
             : FixItHint();
}

FixItHint BraceInsertionHints::closingBraceFixIt() const {
  return ClosingBracePos.isValid()
             ? FixItHint::CreateInsertion(ClosingBracePos, ClosingBrace)
             : FixItHint();
}

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

static SourceLocation findEndLocation(const Stmt &S, const SourceManager &SM,
                                      const LangOptions &LangOpts) {
  SourceLocation Loc = lexer::getUnifiedEndLoc(S, SM, LangOpts);
  if (!Loc.isValid())
    return Loc;

  // Start searching right after S.
  Loc = Loc.getLocWithOffset(1);

  for (;;) {
    assert(Loc.isValid());
    while (isHorizontalWhitespace(*SM.getCharacterData(Loc))) {
      Loc = Loc.getLocWithOffset(1);
    }

    if (isVerticalWhitespace(*SM.getCharacterData(Loc))) {
      // EOL, insert brace before.
      break;
    }
    tok::TokenKind TokKind = getTokenKind(Loc, SM, LangOpts);
    if (TokKind != tok::comment) {
      // Non-comment token, insert brace before.
      break;
    }

    SourceLocation TokEndLoc = Lexer::getLocForEndOfToken(Loc, 0, SM, LangOpts);
    SourceRange TokRange(Loc, TokEndLoc);
    StringRef Comment = Lexer::getSourceText(
        CharSourceRange::getTokenRange(TokRange), SM, LangOpts);
    if (Comment.starts_with("/*") && Comment.contains('\n')) {
      // Multi-line block comment, insert brace before.
      break;
    }
    // else: Trailing comment, insert brace after the newline.

    // Fast-forward current token.
    Loc = TokEndLoc;
  }
  return Loc;
}

BraceInsertionHints getBraceInsertionsHints(const Stmt *const S,
                                            const LangOptions &LangOpts,
                                            const SourceManager &SM,
                                            SourceLocation StartLoc,
                                            SourceLocation EndLocHint) {
  // 1) If there's a corresponding "else" or "while", the check inserts "} "
  // right before that token.
  // 2) If there's a multi-line block comment starting on the same line after
  // the location we're inserting the closing brace at, or there's a non-comment
  // token, the check inserts "\n}" right before that token.
  // 3) Otherwise the check finds the end of line (possibly after some block or
  // line comments) and inserts "\n}" right before that EOL.
  if (!S || isa<CompoundStmt>(S)) {
    // Already inside braces.
    return {};
  }

  // When TreeTransform, Stmt in constexpr IfStmt will be transform to NullStmt.
  // This NullStmt can be detected according to beginning token.
  const SourceLocation StmtBeginLoc = S->getBeginLoc();
  if (isa<NullStmt>(S) && StmtBeginLoc.isValid() &&
      getTokenKind(StmtBeginLoc, SM, LangOpts) == tok::l_brace)
    return {};

  if (StartLoc.isInvalid())
    return {};

  // Convert StartLoc to file location, if it's on the same macro expansion
  // level as the start of the statement. We also need file locations for
  // Lexer::getLocForEndOfToken working properly.
  StartLoc = Lexer::makeFileCharRange(
                 CharSourceRange::getCharRange(StartLoc, S->getBeginLoc()), SM,
                 LangOpts)
                 .getBegin();
  if (StartLoc.isInvalid())
    return {};
  StartLoc = Lexer::getLocForEndOfToken(StartLoc, 0, SM, LangOpts);

  // StartLoc points at the location of the opening brace to be inserted.
  SourceLocation EndLoc;
  std::string ClosingInsertion;
  if (EndLocHint.isValid()) {
    EndLoc = EndLocHint;
    ClosingInsertion = "} ";
  } else {
    EndLoc = findEndLocation(*S, SM, LangOpts);
    ClosingInsertion = "\n}";
  }

  assert(StartLoc.isValid());

  // Change only if StartLoc and EndLoc are on the same macro expansion level.
  // This will also catch invalid EndLoc.
  // Example: LLVM_DEBUG( for(...) do_something() );
  // In this case fix-it cannot be provided as the semicolon which is not
  // visible here is part of the macro. Adding braces here would require adding
  // another semicolon.
  if (Lexer::makeFileCharRange(
          CharSourceRange::getTokenRange(SourceRange(
              SM.getSpellingLoc(StartLoc), SM.getSpellingLoc(EndLoc))),
          SM, LangOpts)
          .isInvalid())
    return {StartLoc};
  return {StartLoc, EndLoc, ClosingInsertion};
}

} // namespace clang::tidy::utils
