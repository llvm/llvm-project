
//===--- FormatTokenSource.h - Format C++ code ------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file defines the \c TokenSource interface, which provides a token
/// stream as well as the ability to manipulate the token stream.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LIB_FORMAT_FORMATTOKENSOURCE_H
#define LLVM_CLANG_LIB_FORMAT_FORMATTOKENSOURCE_H

#include "FormatToken.h"
#include "UnwrappedLineParser.h"

#define DEBUG_TYPE "format-token-source"

namespace clang {
namespace format {

class FormatTokenSource {
public:
  virtual ~FormatTokenSource() {}

  // Returns the next token in the token stream.
  virtual FormatToken *getNextToken() = 0;

  // Returns the token preceding the token returned by the last call to
  // getNextToken() in the token stream, or nullptr if no such token exists.
  virtual FormatToken *getPreviousToken() = 0;

  // Returns the token that would be returned by the next call to
  // getNextToken().
  virtual FormatToken *peekNextToken(bool SkipComment = false) = 0;

  // Returns whether we are at the end of the file.
  // This can be different from whether getNextToken() returned an eof token
  // when the FormatTokenSource is a view on a part of the token stream.
  virtual bool isEOF() = 0;

  // Gets the current position in the token stream, to be used by setPosition().
  virtual unsigned getPosition() = 0;

  // Resets the token stream to the state it was in when getPosition() returned
  // Position, and return the token at that position in the stream.
  virtual FormatToken *setPosition(unsigned Position) = 0;
};

class IndexedTokenSource : public FormatTokenSource {
public:
  IndexedTokenSource(ArrayRef<FormatToken *> Tokens)
      : Tokens(Tokens), Position(-1) {}

  FormatToken *getNextToken() override {
    if (Position >= 0 && isEOF()) {
      LLVM_DEBUG({
        llvm::dbgs() << "Next ";
        dbgToken(Position);
      });
      return Tokens[Position];
    }
    ++Position;
    LLVM_DEBUG({
      llvm::dbgs() << "Next ";
      dbgToken(Position);
    });
    return Tokens[Position];
  }

  FormatToken *getPreviousToken() override {
    return Position > 0 ? Tokens[Position - 1] : nullptr;
  }

  FormatToken *peekNextToken(bool SkipComment) override {
    int Next = Position + 1;
    if (SkipComment)
      while (Tokens[Next]->is(tok::comment))
        ++Next;
    LLVM_DEBUG({
      llvm::dbgs() << "Peeking ";
      dbgToken(Next);
    });
    return Tokens[Next];
  }

  bool isEOF() override { return Tokens[Position]->is(tok::eof); }

  unsigned getPosition() override {
    LLVM_DEBUG(llvm::dbgs() << "Getting Position: " << Position << "\n");
    assert(Position >= 0);
    return Position;
  }

  FormatToken *setPosition(unsigned P) override {
    LLVM_DEBUG(llvm::dbgs() << "Setting Position: " << P << "\n");
    Position = P;
    return Tokens[Position];
  }

  void reset() { Position = -1; }

private:
  void dbgToken(int Position, llvm::StringRef Indent = "") {
    FormatToken *Tok = Tokens[Position];
    llvm::dbgs() << Indent << "[" << Position
                 << "] Token: " << Tok->Tok.getName() << " / " << Tok->TokenText
                 << ", Macro: " << !!Tok->MacroCtx << "\n";
  }

  ArrayRef<FormatToken *> Tokens;
  int Position;
};

class ScopedMacroState : public FormatTokenSource {
public:
  ScopedMacroState(UnwrappedLine &Line, FormatTokenSource *&TokenSource,
                   FormatToken *&ResetToken)
      : Line(Line), TokenSource(TokenSource), ResetToken(ResetToken),
        PreviousLineLevel(Line.Level), PreviousTokenSource(TokenSource),
        Token(nullptr), PreviousToken(nullptr) {
    FakeEOF.Tok.startToken();
    FakeEOF.Tok.setKind(tok::eof);
    TokenSource = this;
    Line.Level = 0;
    Line.InPPDirective = true;
    // InMacroBody gets set after the `#define x` part.
  }

  ~ScopedMacroState() override {
    TokenSource = PreviousTokenSource;
    ResetToken = Token;
    Line.InPPDirective = false;
    Line.InMacroBody = false;
    Line.Level = PreviousLineLevel;
  }

  FormatToken *getNextToken() override {
    // The \c UnwrappedLineParser guards against this by never calling
    // \c getNextToken() after it has encountered the first eof token.
    assert(!eof());
    PreviousToken = Token;
    Token = PreviousTokenSource->getNextToken();
    if (eof())
      return &FakeEOF;
    return Token;
  }

  FormatToken *getPreviousToken() override {
    return PreviousTokenSource->getPreviousToken();
  }

  FormatToken *peekNextToken(bool SkipComment) override {
    if (eof())
      return &FakeEOF;
    return PreviousTokenSource->peekNextToken(SkipComment);
  }

  bool isEOF() override { return PreviousTokenSource->isEOF(); }

  unsigned getPosition() override { return PreviousTokenSource->getPosition(); }

  FormatToken *setPosition(unsigned Position) override {
    PreviousToken = nullptr;
    Token = PreviousTokenSource->setPosition(Position);
    return Token;
  }

private:
  bool eof() {
    return Token && Token->HasUnescapedNewline &&
           !continuesLineComment(*Token, PreviousToken,
                                 /*MinColumnToken=*/PreviousToken);
  }

  FormatToken FakeEOF;
  UnwrappedLine &Line;
  FormatTokenSource *&TokenSource;
  FormatToken *&ResetToken;
  unsigned PreviousLineLevel;
  FormatTokenSource *PreviousTokenSource;

  FormatToken *Token;
  FormatToken *PreviousToken;
};

} // namespace format
} // namespace clang

#undef DEBUG_TYPE

#endif