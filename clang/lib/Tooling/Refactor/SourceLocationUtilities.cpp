//===--- SourceLocationUtilities.cpp - Source location helper functions ---===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "SourceLocationUtilities.h"
#include "clang/AST/Stmt.h"
#include "clang/Lex/Lexer.h"
#include <limits>

namespace clang {
namespace tooling {

SourceLocation findLastLocationOfSourceConstruct(SourceLocation HeaderEnd,
                                                 const Stmt *Body,
                                                 const SourceManager &SM) {
  SourceLocation BodyStart = SM.getSpellingLoc(Body->getBeginLoc());
  unsigned BodyLine = SM.getSpellingLineNumber(BodyStart);
  unsigned HeaderLine = SM.getSpellingLineNumber(HeaderEnd);

  if (BodyLine > HeaderLine) {
    // The Last location on the previous line if the body is not on the same
    // line as the last known location.
    SourceLocation LineLocThatPrecedesBody =
        SM.translateLineCol(SM.getFileID(BodyStart), BodyLine - 1,
                            std::numeric_limits<unsigned>::max());
    if (LineLocThatPrecedesBody.isValid())
      return LineLocThatPrecedesBody;
  }
  // We want to include the location of the '{'.
  return isa<CompoundStmt>(Body) ? BodyStart : BodyStart.getLocWithOffset(-1);
}

SourceLocation findFirstLocationOfSourceConstruct(SourceLocation HeaderStart,
                                                  const Stmt *PreviousBody,
                                                  const SourceManager &SM) {
  if (!isa<CompoundStmt>(PreviousBody))
    return HeaderStart;
  SourceLocation BodyEnd = SM.getSpellingLoc(PreviousBody->getEndLoc());
  unsigned BodyLine = SM.getSpellingLineNumber(BodyEnd);
  unsigned HeaderLine = SM.getSpellingLineNumber(HeaderStart);
  if (BodyLine >= HeaderLine)
    return BodyEnd;
  return HeaderStart;
}

bool isLocationInAnyRange(SourceLocation Location, ArrayRef<SourceRange> Ranges,
                          const SourceManager &SM) {
  for (const SourceRange &Range : Ranges) {
    if (!isPointWithin(Location, Range.getBegin(), Range.getEnd(), SM))
      continue;
    return true;
  }
  return false;
}

SourceLocation getPreciseTokenLocEnd(SourceLocation Loc,
                                     const SourceManager &SM,
                                     const LangOptions &LangOpts) {
  return Lexer::getLocForEndOfToken(Loc, 0, SM, LangOpts);
}

SourceLocation findClosingParenLocEnd(SourceLocation LastKnownLoc,
                                      const SourceManager &SM,
                                      const LangOptions &LangOpts) {
  return Lexer::findLocationAfterToken(
      LastKnownLoc, tok::r_paren, SM, LangOpts,
      /*SkipTrailingWhitespaceAndNewLine=*/false);
}

SourceRange getRangeOfNextToken(SourceLocation Loc, tok::TokenKind Kind,
                                const SourceManager &SM,
                                const LangOptions &LangOpts) {
  SourceLocation NextLoc =
      Lexer::findLocationAfterToken(Loc, Kind, SM, LangOpts,
                                    /*SkipTrailingWhitespaceAndNewLine=*/false);
  if (NextLoc.isInvalid())
    return SourceRange();
  return SourceRange(
      Lexer::GetBeginningOfToken(NextLoc.getLocWithOffset(-1), SM, LangOpts),
      NextLoc);
}

SourceLocation findLastNonCompoundLocation(const Stmt *S) {
  const auto *CS = dyn_cast<CompoundStmt>(S);
  if (!CS)
    return S->getEndLoc();
  return CS->body_back() ? CS->body_back()->getEndLoc() : SourceLocation();
}

bool areOnSameLine(SourceLocation Loc1, SourceLocation Loc2,
                   const SourceManager &SM) {
  return !Loc1.isMacroID() && !Loc2.isMacroID() &&
         SM.getSpellingLineNumber(Loc1) == SM.getSpellingLineNumber(Loc2);
}

SourceLocation
getLastLineLocationUnlessItHasOtherTokens(SourceLocation SpellingLoc,
                                          const SourceManager &SM,
                                          const LangOptions &LangOpts) {
  assert(!SpellingLoc.isMacroID() && "Expecting a spelling location");
  SourceLocation NextTokenLoc =
      Lexer::findNextTokenLocationAfterTokenAt(SpellingLoc, SM, LangOpts);
  if (NextTokenLoc.isValid()) {
    bool IsSameLine = areOnSameLine(SpellingLoc, NextTokenLoc, SM);
    if (IsSameLine) {
      // Could be a ';' on the same line, so try looking after the ';'
      if (isSemicolonAtLocation(NextTokenLoc, SM, LangOpts))
        return getLastLineLocationUnlessItHasOtherTokens(NextTokenLoc, SM,
                                                         LangOpts);
    } else {
      SourceLocation LastLoc = SM.translateLineCol(
          SM.getFileID(SpellingLoc), SM.getSpellingLineNumber(SpellingLoc),
          std::numeric_limits<unsigned>::max());
      if (LastLoc.isValid())
        return LastLoc;
    }
  }
  return getPreciseTokenLocEnd(SpellingLoc, SM, LangOpts);
}

bool isSemicolonAtLocation(SourceLocation TokenLoc, const SourceManager &SM,
                           const LangOptions &LangOpts) {
  return Lexer::getSourceText(
             CharSourceRange::getTokenRange(TokenLoc, TokenLoc), SM,
             LangOpts) == ";";
}

SourceRange trimSelectionRange(SourceRange Range, const SourceManager &SM,
                               const LangOptions &LangOpts) {
  bool IsInvalid = false;
  StringRef Text = Lexer::getSourceText(CharSourceRange::getCharRange(Range),
                                        SM, LangOpts, &IsInvalid);
  if (IsInvalid || Text.empty())
    return Range;
  assert(Range.getBegin().isFileID() && "Not a file range!");

  std::string Source = Text.str();
  Lexer Lex(Range.getBegin(), LangOpts, Source.c_str(), Source.c_str(),
            Source.c_str() + Source.size());
  // Get comment tokens as well.
  Lex.SetCommentRetentionState(true);
  SourceLocation StartLoc, EndLoc;
  while (true) {
    Token Tok;
    Lex.LexFromRawLexer(Tok);
    if (Tok.getKind() == tok::eof)
      break;
    if (StartLoc.isInvalid())
      StartLoc = Tok.getLocation();
    if (Tok.getKind() != tok::semi)
      EndLoc = Tok.getEndLoc();
  }
  return StartLoc.isValid() && EndLoc.isValid() ? SourceRange(StartLoc, EndLoc)
                                                : SourceRange();
}

/// Tokenize the given file and check if it contains a comment that ends at the
/// given location.
static SourceLocation findCommentThatEndsAt(FileID FID,
                                            SourceLocation StartOfFile,
                                            const SourceManager &SM,
                                            const LangOptions &LangOpts,
                                            SourceLocation ExpectedEndLoc) {
  // Try to load the file buffer.
  bool InvalidTemp = false;
  StringRef File = SM.getBufferData(FID, &InvalidTemp);
  if (InvalidTemp)
    return SourceLocation();

  // Search for the comment that ends at the given location.
  Lexer Lex(StartOfFile, LangOpts, File.begin(), File.begin(), File.end());
  Lex.SetCommentRetentionState(true);
  Token Tok;
  while (!Lex.LexFromRawLexer(Tok)) {
    if (Tok.is(tok::comment) && Tok.getEndLoc() == ExpectedEndLoc)
      return Tok.getLocation();
  }
  // Find the token.
  return SourceLocation();
}

SourceLocation getLocationOfPrecedingComment(SourceLocation Location,
                                             const SourceManager &SM,
                                             const LangOptions &LangOpts) {
  SourceLocation PrevResult = Location;
  SourceLocation Result = Location;
  if (Result.isMacroID())
    Result = SM.getExpansionLoc(Result);
  FileID FID = SM.getFileID(Result);
  SourceLocation StartOfFile = SM.getLocForStartOfFile(FID);
  Token Tok;
  Tok.setKind(tok::unknown);
  SourceLocation TokenLoc = Result;
  auto GetPreviousToken = [&]() -> bool {
    TokenLoc =
        Lexer::GetBeginningOfToken(TokenLoc.getLocWithOffset(-1), SM, LangOpts);
    return !Lexer::getRawToken(TokenLoc, Tok, SM, LangOpts);
  };
  // Look for a comment token.
  while (TokenLoc != StartOfFile) {
    bool LocHasToken = GetPreviousToken();
    if (LocHasToken && Tok.is(tok::slash)) {
      // Check if this  the end of a multiline '/*' comment before returning.
      SourceLocation CommentLoc = findCommentThatEndsAt(
          FID, StartOfFile, SM, LangOpts, Tok.getEndLoc());
      return CommentLoc.isInvalid() ? Result : CommentLoc;
    }
    if (LocHasToken && Tok.isNot(tok::comment))
      break;
    if (!LocHasToken)
      continue;
    // We found a preceding comment. Check if there are other preceding
    // comments.
    PrevResult = Result;
    Result = Tok.getLocation();
    while (TokenLoc != StartOfFile) {
      bool LocHasToken = GetPreviousToken();
      if (LocHasToken && Tok.isNot(tok::comment)) {
        // Reset the result to the previous location if this comment trails
        // another token on the same line.
        if (SM.getSpellingLineNumber(Tok.getEndLoc()) ==
            SM.getSpellingLineNumber(Result))
          Result = PrevResult;
        break;
      }
      if (!LocHasToken)
        continue;
      // The location of this comment is accepted only when the next comment
      // is located immediately after this comment.
      if (SM.getSpellingLineNumber(Tok.getEndLoc()) !=
          SM.getSpellingLineNumber(Result) - 1)
        break;
      PrevResult = Result;
      Result = Tok.getLocation();
    }
    break;
  }
  return Result;
}

SourceLocation getLocationOfPrecedingToken(SourceLocation Loc,
                                           const SourceManager &SM,
                                           const LangOptions &LangOpts) {
  SourceLocation Result = Loc;
  if (Result.isMacroID())
    Result = SM.getExpansionLoc(Result);
  FileID FID = SM.getFileID(Result);
  SourceLocation StartOfFile = SM.getLocForStartOfFile(FID);
  if (Loc == StartOfFile)
    return SourceLocation();
  return Lexer::GetBeginningOfToken(Result.getLocWithOffset(-1), SM, LangOpts);
}

} // end namespace tooling
} // end namespace clang
