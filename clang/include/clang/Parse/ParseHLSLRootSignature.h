//===--- ParseHLSLRootSignature.h -------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  This file defines the ParseHLSLRootSignature interface.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_PARSE_PARSEHLSLROOTSIGNATURE_H
#define LLVM_CLANG_PARSE_PARSEHLSLROOTSIGNATURE_H

#include "clang/AST/APValue.h"
#include "clang/Basic/DiagnosticLex.h"
#include "clang/Lex/LiteralSupport.h"
#include "clang/Lex/Preprocessor.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSwitch.h"

#include "llvm/Frontend/HLSL/HLSLRootSignature.h"

namespace clang {
namespace hlsl {

struct RootSignatureToken {
  enum Kind {
#define TOK(X) X,
#include "clang/Parse/HLSLRootSignatureTokenKinds.def"
  };

  Kind Kind = Kind::error;

  // Retain the SouceLocation of the token for diagnostics
  clang::SourceLocation TokLoc;

  APValue NumLiteral = APValue();

  // Constructors
  RootSignatureToken() : TokLoc(SourceLocation()) {}
  RootSignatureToken(clang::SourceLocation TokLoc) : TokLoc(TokLoc) {}
  RootSignatureToken(enum Kind Kind, clang::SourceLocation TokLoc)
      : Kind(Kind), TokLoc(TokLoc) {}
};
using TokenKind = enum RootSignatureToken::Kind;

class RootSignatureLexer {
public:
  RootSignatureLexer(StringRef Signature, clang::SourceLocation SourceLoc,
                     clang::Preprocessor &PP)
      : Buffer(Signature), SourceLoc(SourceLoc), PP(PP) {}

  /// Updates CurToken to the next token. Either it will take the previously
  /// lexed NextToken, or it will lex a token.
  ///
  /// The return value denotes if there was a failure.
  bool ConsumeToken();

  /// Returns the token that comes after CurToken or std::nullopt if an
  /// error is encountered during lexing of the next token.
  std::optional<RootSignatureToken> PeekNextToken();

  RootSignatureToken GetCurToken() { return CurToken; }

  /// Check if we have reached the end of input
  bool EndOfBuffer() {
    AdvanceBuffer(Buffer.take_while(isspace).size());
    return Buffer.empty();
  }

private:
  // Internal buffer to iterate over
  StringRef Buffer;

  // Current Token state
  RootSignatureToken CurToken;
  std::optional<RootSignatureToken> NextToken = std::nullopt;

  // Passed down parameters from Sema
  clang::SourceLocation SourceLoc;
  clang::Preprocessor &PP;

  bool LexNumber(RootSignatureToken &Result);
  bool LexToken(RootSignatureToken &Result);

  // Advance the buffer by the specified number of characters. Updates the
  // SourceLocation appropriately.
  void AdvanceBuffer(unsigned NumCharacters = 1) {
    Buffer = Buffer.drop_front(NumCharacters);
    SourceLoc = SourceLoc.getLocWithOffset(NumCharacters);
  }
};

class RootSignatureParser {
public:
  RootSignatureParser(SmallVector<llvm::hlsl::rootsig::RootElement> &Elements,
                      RootSignatureLexer &Lexer, DiagnosticsEngine &Diags);

  /// Iterates over the provided tokens and constructs the in-memory
  /// representations of the RootElements.
  ///
  /// The return value denotes if there was a failure and the method will
  /// return on the first encountered failure, or, return false if it
  /// can sucessfully reach the end of the tokens.
  bool Parse();

private:
  SmallVector<llvm::hlsl::rootsig::RootElement> &Elements;
  RootSignatureLexer &Lexer;
  DiagnosticsEngine &Diags;

  RootSignatureToken CurToken;
};

} // namespace hlsl
} // namespace clang

#endif // LLVM_CLANG_PARSE_PARSEHLSLROOTSIGNATURE_H
