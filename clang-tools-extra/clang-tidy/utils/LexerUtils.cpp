//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "LexerUtils.h"
#include "clang/Basic/SourceManager.h"
#include <optional>
#include <utility>
#include <vector>

namespace clang::tidy::utils::lexer {

std::pair<std::optional<Token>, SourceLocation>
getPreviousTokenAndStart(SourceLocation Location, const SourceManager &SM,
                         const LangOptions &LangOpts, bool SkipComments) {
  const std::optional<Token> Tok =
      Lexer::findPreviousToken(Location, SM, LangOpts, !SkipComments);

  if (Tok.has_value())
    return {*Tok, Lexer::GetBeginningOfToken(Tok->getLocation(), SM, LangOpts)};

  return {std::nullopt, SourceLocation()};
}

std::optional<Token> getPreviousToken(SourceLocation Location,
                                      const SourceManager &SM,
                                      const LangOptions &LangOpts,
                                      bool SkipComments) {
  auto [Token, Start] =
      getPreviousTokenAndStart(Location, SM, LangOpts, SkipComments);
  return Token;
}

SourceLocation findPreviousTokenStart(SourceLocation Start,
                                      const SourceManager &SM,
                                      const LangOptions &LangOpts) {
  if (Start.isInvalid() || Start.isMacroID())
    return {};

  const SourceLocation BeforeStart = Start.getLocWithOffset(-1);
  if (BeforeStart.isInvalid() || BeforeStart.isMacroID())
    return {};

  return Lexer::GetBeginningOfToken(BeforeStart, SM, LangOpts);
}

SourceLocation findPreviousTokenKind(SourceLocation Start,
                                     const SourceManager &SM,
                                     const LangOptions &LangOpts,
                                     tok::TokenKind TK) {
  if (Start.isInvalid() || Start.isMacroID())
    return {};

  while (true) {
    const SourceLocation L = findPreviousTokenStart(Start, SM, LangOpts);
    if (L.isInvalid() || L.isMacroID())
      return {};

    Token T;
    if (Lexer::getRawToken(L, T, SM, LangOpts, /*IgnoreWhiteSpace=*/true))
      return {};

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

  while (Loc <= Range.getEnd()) {
    if (Loc.isMacroID())
      return true;

    std::optional<Token> Tok = findNextTokenSkippingComments(Loc, SM, LangOpts);

    if (!Tok)
      return true;

    if (Tok->is(tok::hash))
      return true;

    Loc = Tok->getLocation();
  }

  return false;
}

std::vector<CommentToken>
getTrailingCommentsInRange(CharSourceRange Range, const SourceManager &SM,
                           const LangOptions &LangOpts) {
  std::vector<CommentToken> Comments;
  if (Range.isInvalid())
    return Comments;

  const CharSourceRange FileRange =
      Lexer::makeFileCharRange(Range, SM, LangOpts);
  if (FileRange.isInvalid())
    return Comments;

  const std::pair<FileID, unsigned> BeginLoc =
      SM.getDecomposedLoc(FileRange.getBegin());
  const std::pair<FileID, unsigned> EndLoc =
      SM.getDecomposedLoc(FileRange.getEnd());

  if (BeginLoc.first != EndLoc.first)
    return Comments;

  bool Invalid = false;
  const StringRef Buffer = SM.getBufferData(BeginLoc.first, &Invalid);
  if (Invalid)
    return Comments;

  const char *StrData = Buffer.data() + BeginLoc.second;

  Lexer TheLexer(SM.getLocForStartOfFile(BeginLoc.first), LangOpts,
                 Buffer.begin(), StrData, Buffer.end());
  // Use raw lexing with comment retention so we can see comment tokens without
  // preprocessing or macro expansion effects.
  TheLexer.SetCommentRetentionState(true);

  while (true) {
    Token Tok;
    if (TheLexer.LexFromRawLexer(Tok))
      break;
    if (Tok.is(tok::eof) || Tok.getLocation() == FileRange.getEnd() ||
        SM.isBeforeInTranslationUnit(FileRange.getEnd(), Tok.getLocation()))
      break;

    if (Tok.is(tok::comment)) {
      const std::pair<FileID, unsigned> CommentLoc =
          SM.getDecomposedLoc(Tok.getLocation());
      assert(CommentLoc.first == BeginLoc.first);
      Comments.emplace_back(CommentToken{
          Tok.getLocation(),
          StringRef(Buffer.begin() + CommentLoc.second, Tok.getLength()),
      });
    } else {
      // Clear comments found before the different token, e.g. comma. Callers
      // use this to retrieve only the contiguous comment block that directly
      // precedes a token of interest.
      Comments.clear();
    }
  }

  return Comments;
}

std::optional<Token> getQualifyingToken(tok::TokenKind TK,
                                        CharSourceRange Range,
                                        const ASTContext &Context,
                                        const SourceManager &SM) {
  assert((TK == tok::kw_const || TK == tok::kw_volatile ||
          TK == tok::kw_restrict) &&
         "TK is not a qualifier keyword");
  const std::pair<FileID, unsigned> LocInfo =
      SM.getDecomposedLoc(Range.getBegin());
  const StringRef File = SM.getBufferData(LocInfo.first);
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
  return isa<Expr, DoStmt, ReturnStmt, BreakStmt, ContinueStmt, GotoStmt,
             SEHLeaveStmt>(S);
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

  return {};
}

SourceLocation getUnifiedEndLoc(const Stmt &S, const SourceManager &SM,
                                const LangOptions &LangOpts) {
  const Stmt *LastChild = &S;
  while (!LastChild->children().empty() && !breakAndReturnEnd(*LastChild) &&
         !breakAndReturnEndPlus1Token(*LastChild)) {
    for (const Stmt *Child : LastChild->children())
      LastChild = Child;
  }

  if (!breakAndReturnEnd(*LastChild) && breakAndReturnEndPlus1Token(*LastChild))
    return getSemicolonAfterStmtEndLoc(S.getEndLoc(), SM, LangOpts);

  return S.getEndLoc();
}

SourceLocation getLocationForNoexceptSpecifier(const FunctionDecl *FuncDecl,
                                               const SourceManager &SM) {
  if (!FuncDecl)
    return {};

  const LangOptions &LangOpts = FuncDecl->getLangOpts();

  if (FuncDecl->getNumParams() == 0) {
    // Start at the beginning of the function declaration, and find the closing
    // parenthesis after which we would place the noexcept specifier.
    Token CurrentToken;
    SourceLocation CurrentLocation = FuncDecl->getBeginLoc();
    while (!Lexer::getRawToken(CurrentLocation, CurrentToken, SM, LangOpts,
                               true)) {
      if (CurrentToken.is(tok::r_paren))
        return CurrentLocation.getLocWithOffset(1);

      CurrentLocation = CurrentToken.getEndLoc();
    }

    // Failed to find the closing parenthesis, so just return an invalid
    // SourceLocation.
    return {};
  }

  // FunctionDecl with parameters
  const SourceLocation NoexceptLoc =
      FuncDecl->getParamDecl(FuncDecl->getNumParams() - 1)->getEndLoc();
  if (NoexceptLoc.isValid())
    return Lexer::findLocationAfterToken(
        NoexceptLoc, tok::r_paren, SM, LangOpts,
        /*SkipTrailingWhitespaceAndNewLine=*/true);

  return {};
}

} // namespace clang::tidy::utils::lexer
