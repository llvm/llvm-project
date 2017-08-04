//===--- PTHLexer.h - Lexer based on Pre-tokenized input --------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the PTHLexer interface.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LEX_PTHLEXER_H
#define LLVM_CLANG_LEX_PTHLEXER_H

#include "clang/Lex/PreprocessorLexer.h"

namespace clang {

class PTHManager;
class PTHSpellingSearch;

class PTHLexer : public PreprocessorLexer {
  SourceLocation FileStartLoc;

  /// TokBuf - Buffer from PTH file containing raw token data.
  const unsigned char* TokBuf;

  /// CurPtr - Pointer into current offset of the token buffer where
  ///  the next token will be read.
  const unsigned char* CurPtr;

  /// LastHashTokPtr - Pointer into TokBuf of the last processed '#'
  ///  token that appears at the start of a line.
  const unsigned char* LastHashTokPtr;

  /// PPCond - Pointer to a side table in the PTH file that provides a
  ///  a concise summary of the preprocessor conditional block structure.
  ///  This is used to perform quick skipping of conditional blocks.
  const unsigned char* PPCond;

  /// CurPPCondPtr - Pointer inside PPCond that refers to the next entry
  ///  to process when doing quick skipping of preprocessor blocks.
  const unsigned char* CurPPCondPtr;

  PTHLexer(const PTHLexer &) = delete;
  void operator=(const PTHLexer &) = delete;

  /// ReadToken - Used by PTHLexer to read tokens TokBuf.
  void ReadToken(Token& T);
  
  bool LexEndOfFile(Token &Result);

  /// PTHMgr - The PTHManager object that created this PTHLexer.
  PTHManager& PTHMgr;

  Token EofToken;

protected:
  friend class PTHManager;

  /// Create a PTHLexer for the specified token stream.
  PTHLexer(Preprocessor& pp, FileID FID, const unsigned char *D,
           const unsigned char* ppcond, PTHManager &PM);
public:
  ~PTHLexer() override {}

  /// Lex - Return the next token.
  bool Lex(Token &Tok);

  void getEOF(Token &Tok);

  /// DiscardToEndOfLine - Read the rest of the current preprocessor line as an
  /// uninterpreted string.  This switches the lexer out of directive mode.
  void DiscardToEndOfLine();

  /// isNextPPTokenLParen - Return 1 if the next unexpanded token will return a
  /// tok::l_paren token, 0 if it is something else and 2 if there are no more
  /// tokens controlled by this lexer.
  unsigned isNextPPTokenLParen() {
    // isNextPPTokenLParen is not on the hot path, and all we care about is
    // whether or not we are at a token with kind tok::eof or tok::l_paren.
    // Just read the first byte from the current token pointer to determine
    // its kind.
    tok::TokenKind x = (tok::TokenKind)*CurPtr;
    return x == tok::eof ? 2 : x == tok::l_paren;
  }

  /// IndirectLex - An indirect call to 'Lex' that can be invoked via
  ///  the PreprocessorLexer interface.
  void IndirectLex(Token &Result) override { Lex(Result); }

  /// getSourceLocation - Return a source location for the token in
  /// the current file.
  SourceLocation getSourceLocation() override;

  /// SkipBlock - Used by Preprocessor to skip the current conditional block.
  bool SkipBlock();
};

}  // end namespace clang

#endif
