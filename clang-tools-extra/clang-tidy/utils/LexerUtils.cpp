//===--- LexerUtils.cpp - clang-tidy---------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "LexerUtils.h"
#include "clang/AST/AST.h"
#include "clang/Basic/SourceManager.h"
#include <optional>

namespace clang::tidy::utils::lexer {

Token getPreviousToken(SourceLocation Location, const SourceManager &SM,
                       const LangOptions &LangOpts, bool SkipComments) {
  Token Token;
  Token.setKind(tok::unknown);

  Location = Location.getLocWithOffset(-1);
  if (Location.isInvalid())
      return Token;

  auto StartOfFile = SM.getLocForStartOfFile(SM.getFileID(Location));
  while (Location != StartOfFile) {
    Location = Lexer::GetBeginningOfToken(Location, SM, LangOpts);
    if (!Lexer::getRawToken(Location, Token, SM, LangOpts) &&
        (!SkipComments || !Token.is(tok::comment))) {
      break;
    }
    Location = Location.getLocWithOffset(-1);
  }
  return Token;
}

SourceLocation findPreviousTokenStart(SourceLocation Start,
                                      const SourceManager &SM,
                                      const LangOptions &LangOpts) {
  if (Start.isInvalid() || Start.isMacroID())
    return SourceLocation();

  SourceLocation BeforeStart = Start.getLocWithOffset(-1);
  if (BeforeStart.isInvalid() || BeforeStart.isMacroID())
    return SourceLocation();

  return Lexer::GetBeginningOfToken(BeforeStart, SM, LangOpts);
}

SourceLocation findPreviousTokenKind(SourceLocation Start,
                                     const SourceManager &SM,
                                     const LangOptions &LangOpts,
                                     tok::TokenKind TK) {
  if (Start.isInvalid() || Start.isMacroID())
    return SourceLocation();

  while (true) {
    SourceLocation L = findPreviousTokenStart(Start, SM, LangOpts);
    if (L.isInvalid() || L.isMacroID())
      return SourceLocation();

    Token T;
    if (Lexer::getRawToken(L, T, SM, LangOpts, /*IgnoreWhiteSpace=*/true))
      return SourceLocation();

    if (T.is(TK))
      return T.getLocation();

    Start = L;
  }
}

SourceLocation findNextTerminator(SourceLocation Start, const SourceManager &SM,
                                  const LangOptions &LangOpts) {
  return findNextAnyTokenKind(Start, SM, LangOpts, tok::comma, tok::semi);
}

std::optional<Token>
findNextTokenSkippingComments(SourceLocation Start, const SourceManager &SM,
                              const LangOptions &LangOpts) {
  while (Start.isValid()) {
    std::optional<Token> CurrentToken =
        Lexer::findNextToken(Start, SM, LangOpts);
    if (!CurrentToken || !CurrentToken->is(tok::comment))
      return CurrentToken;

    Start = CurrentToken->getLocation();
  }

  return std::nullopt;
}

bool rangeContainsExpansionsOrDirectives(SourceRange Range,
                                         const SourceManager &SM,
                                         const LangOptions &LangOpts) {
  assert(Range.isValid() && "Invalid Range for relexing provided");
  SourceLocation Loc = Range.getBegin();

  while (Loc <= Range.getEnd()) {
    if (Loc.isMacroID())
      return true;

    std::optional<Token> Tok = Lexer::findNextToken(Loc, SM, LangOpts);

    if (!Tok)
      return true;

    if (Tok->is(tok::hash))
      return true;

    Loc = Tok->getLocation();
  }

  return false;
}

std::optional<Token> getQualifyingToken(tok::TokenKind TK,
                                        CharSourceRange Range,
                                        const ASTContext &Context,
                                        const SourceManager &SM) {
  assert((TK == tok::kw_const || TK == tok::kw_volatile ||
          TK == tok::kw_restrict) &&
         "TK is not a qualifier keyword");
  std::pair<FileID, unsigned> LocInfo = SM.getDecomposedLoc(Range.getBegin());
  StringRef File = SM.getBufferData(LocInfo.first);
  Lexer RawLexer(SM.getLocForStartOfFile(LocInfo.first), Context.getLangOpts(),
                 File.begin(), File.data() + LocInfo.second, File.end());
  std::optional<Token> LastMatchBeforeTemplate;
  std::optional<Token> LastMatchAfterTemplate;
  bool SawTemplate = false;
  Token Tok;
  while (!RawLexer.LexFromRawLexer(Tok) &&
         Range.getEnd() != Tok.getLocation() &&
         !SM.isBeforeInTranslationUnit(Range.getEnd(), Tok.getLocation())) {
    if (Tok.is(tok::raw_identifier)) {
      IdentifierInfo &Info = Context.Idents.get(
          StringRef(SM.getCharacterData(Tok.getLocation()), Tok.getLength()));
      Tok.setIdentifierInfo(&Info);
      Tok.setKind(Info.getTokenID());
    }
    if (Tok.is(tok::less))
      SawTemplate = true;
    else if (Tok.isOneOf(tok::greater, tok::greatergreater))
      LastMatchAfterTemplate = std::nullopt;
    else if (Tok.is(TK)) {
      if (SawTemplate)
        LastMatchAfterTemplate = Tok;
      else
        LastMatchBeforeTemplate = Tok;
    }
  }
  return LastMatchAfterTemplate != std::nullopt ? LastMatchAfterTemplate
                                                : LastMatchBeforeTemplate;
}

static bool breakAndReturnEnd(const Stmt &S) {
  return isa<CompoundStmt, DeclStmt, NullStmt>(S);
}

static bool breakAndReturnEndPlus1Token(const Stmt &S) {
  return isa<Expr, DoStmt, ReturnStmt, BreakStmt, ContinueStmt, GotoStmt, SEHLeaveStmt>(S);
}

// Given a Stmt which does not include it's semicolon this method returns the
// SourceLocation of the semicolon.
static SourceLocation getSemicolonAfterStmtEndLoc(const SourceLocation &EndLoc,
                                                  const SourceManager &SM,
                                                  const LangOptions &LangOpts) {

  if (EndLoc.isMacroID()) {
    // Assuming EndLoc points to a function call foo within macro F.
    // This method is supposed to return location of the semicolon within
    // those macro arguments:
    //  F     (      foo()               ;   )
    //  ^ EndLoc         ^ SpellingLoc   ^ next token of SpellingLoc
    const SourceLocation SpellingLoc = SM.getSpellingLoc(EndLoc);
    std::optional<Token> NextTok =
        findNextTokenSkippingComments(SpellingLoc, SM, LangOpts);

    // Was the next token found successfully?
    // All macro issues are simply resolved by ensuring it's a semicolon.
    if (NextTok && NextTok->is(tok::TokenKind::semi)) {
      // Ideally this would return `F` with spelling location `;` (NextTok)
      // following the example above. For now simply return NextTok location.
      return NextTok->getLocation();
    }

    // Fallthrough to 'normal handling'.
    //  F     (      foo()              ) ;
    //  ^ EndLoc         ^ SpellingLoc  ) ^ next token of EndLoc
  }

  std::optional<Token> NextTok =
      findNextTokenSkippingComments(EndLoc, SM, LangOpts);

  // Testing for semicolon again avoids some issues with macros.
  if (NextTok && NextTok->is(tok::TokenKind::semi))
    return NextTok->getLocation();

  return SourceLocation();
}

SourceLocation getUnifiedEndLoc(const Stmt &S, const SourceManager &SM,
                                const LangOptions &LangOpts) {

  const Stmt *LastChild = &S;
  while (!LastChild->children().empty() && !breakAndReturnEnd(*LastChild) &&
         !breakAndReturnEndPlus1Token(*LastChild)) {
    for (const Stmt *Child : LastChild->children())
      LastChild = Child;
  }

  if (!breakAndReturnEnd(*LastChild) &&
      breakAndReturnEndPlus1Token(*LastChild))
    return getSemicolonAfterStmtEndLoc(S.getEndLoc(), SM, LangOpts);

  return S.getEndLoc();
}

} // namespace clang::tidy::utils::lexer
