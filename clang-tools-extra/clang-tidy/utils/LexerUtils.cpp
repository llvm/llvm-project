//===--- LexerUtils.cpp - clang-tidy---------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "LexerUtils.h"

namespace clang {
namespace tidy {
namespace utils {
namespace lexer {

Token getPreviousToken(SourceLocation Location, const SourceManager &SM,
                       const LangOptions &LangOpts, bool SkipComments) {
  Token Token;
  Token.setKind(tok::unknown);
  Location = Location.getLocWithOffset(-1);
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

bool rangeContainsExpansionsOrDirectives(SourceRange Range,
                                         const SourceManager &SM,
                                         const LangOptions &LangOpts) {
  assert(Range.isValid() && "Invalid Range for relexing provided");
  SourceLocation Loc = Range.getBegin();

  while (Loc < Range.getEnd()) {
    if (Loc.isMacroID())
      return true;

    llvm::Optional<Token> Tok = Lexer::findNextToken(Loc, SM, LangOpts);

    if (!Tok)
      return true;

    if (Tok->is(tok::hash))
      return true;

    Loc = Lexer::getLocForEndOfToken(Loc, 0, SM, LangOpts).getLocWithOffset(1);
  }

  return false;
}

llvm::Optional<Token> getConstQualifyingToken(CharSourceRange Range,
                                              const ASTContext &Context,
                                              const SourceManager &SM) {
  std::pair<FileID, unsigned> LocInfo = SM.getDecomposedLoc(Range.getBegin());
  StringRef File = SM.getBufferData(LocInfo.first);
  Lexer RawLexer(SM.getLocForStartOfFile(LocInfo.first), Context.getLangOpts(),
                 File.begin(), File.data() + LocInfo.second, File.end());
  llvm::Optional<Token> FirstConstTok;
  Token LastTokInRange;
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
    if (Tok.is(tok::kw_const) && !FirstConstTok)
      FirstConstTok = Tok;
    LastTokInRange = Tok;
  }
  // If the last token in the range is a `const`, then it const qualifies the
  // type.  Otherwise, the first `const` token, if any, is the qualifier.
  return LastTokInRange.is(tok::kw_const) ? LastTokInRange : FirstConstTok;
}
} // namespace lexer
} // namespace utils
} // namespace tidy
} // namespace clang
