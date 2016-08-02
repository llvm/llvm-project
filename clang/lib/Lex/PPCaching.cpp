//===--- PPCaching.cpp - Handle caching lexed tokens ----------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements pieces of the Preprocessor interface that manage the
// caching of lexed tokens.
//
//===----------------------------------------------------------------------===//

#include "clang/Lex/Preprocessor.h"
using namespace clang;

// EnableBacktrackAtThisPos - From the point that this method is called, and
// until CommitBacktrackedTokens() or Backtrack() is called, the Preprocessor
// keeps track of the lexed tokens so that a subsequent Backtrack() call will
// make the Preprocessor re-lex the same tokens.
//
// Nested backtracks are allowed, meaning that EnableBacktrackAtThisPos can
// be called multiple times and CommitBacktrackedTokens/Backtrack calls will
// be combined with the EnableBacktrackAtThisPos calls in reverse order.
void Preprocessor::EnableBacktrackAtThisPos() {
  BacktrackPositions.push_back(CachedLexPos);
  EnterCachingLexMode();
}

// Disable the last EnableBacktrackAtThisPos call.
void Preprocessor::CommitBacktrackedTokens() {
  assert(!BacktrackPositions.empty()
         && "EnableBacktrackAtThisPos was not called!");
  BacktrackPositions.pop_back();
}

// Make Preprocessor re-lex the tokens that were lexed since
// EnableBacktrackAtThisPos() was previously called.
void Preprocessor::Backtrack() {
  assert(!BacktrackPositions.empty()
         && "EnableBacktrackAtThisPos was not called!");
  CachedLexPos = BacktrackPositions.back();
  BacktrackPositions.pop_back();
  recomputeCurLexerKind();
}

void Preprocessor::CachingLex(Token &Result) {
  if (!InCachingLexMode())
    return;

  if (CachedLexPos < CachedTokens.size()) {
    Result = CachedTokens[CachedLexPos++];
    return;
  }

  ExitCachingLexMode();
  Lex(Result);

  if (isBacktrackEnabled()) {
    // Cache the lexed token.
    EnterCachingLexMode();
    CachedTokens.push_back(Result);
    ++CachedLexPos;
    return;
  }

  if (CachedLexPos < CachedTokens.size()) {
    EnterCachingLexMode();
  } else {
    // All cached tokens were consumed.
    CachedTokens.clear();
    CachedLexPos = 0;
  }
}

void Preprocessor::EnterCachingLexMode() {
  if (InCachingLexMode())
    return;

  PushIncludeMacroStack();
  CurLexerKind = CLK_CachingLexer;
}


const Token &Preprocessor::PeekAhead(unsigned N) {
  assert(CachedLexPos + N > CachedTokens.size() && "Confused caching.");
  ExitCachingLexMode();
  for (unsigned C = CachedLexPos + N - CachedTokens.size(); C > 0; --C) {
    CachedTokens.push_back(Token());
    Lex(CachedTokens.back());
  }
  EnterCachingLexMode();
  return CachedTokens.back();
}

void Preprocessor::AnnotatePreviousCachedTokens(const Token &Tok) {
  assert(Tok.isAnnotation() && "Expected annotation token");
  assert(CachedLexPos != 0 && "Expected to have some cached tokens");
  assert(CachedTokens[CachedLexPos-1].getLastLoc() == Tok.getAnnotationEndLoc()
         && "The annotation should be until the most recent cached token");

  // Start from the end of the cached tokens list and look for the token
  // that is the beginning of the annotation token.
  for (CachedTokensTy::size_type i = CachedLexPos; i != 0; --i) {
    CachedTokensTy::iterator AnnotBegin = CachedTokens.begin() + i-1;
    if (AnnotBegin->getLocation() == Tok.getLocation()) {
      assert((BacktrackPositions.empty() || BacktrackPositions.back() < i) &&
             "The backtrack pos points inside the annotated tokens!");
      // Replace the cached tokens with the single annotation token.
      if (i < CachedLexPos)
        CachedTokens.erase(AnnotBegin + 1, CachedTokens.begin() + CachedLexPos);
      *AnnotBegin = Tok;
      CachedLexPos = i;
      return;
    }
  }
}
