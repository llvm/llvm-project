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

namespace clang {
namespace hlsl {

struct RootSignatureToken {
  enum Kind {
#define TOK(X) X,
#include "clang/Parse/HLSLRootSignatureTokenKinds.def"
  };

  Kind Kind = Kind::invalid;

  // Retain the SouceLocation of the token for diagnostics
  clang::SourceLocation TokLoc;

  APValue NumLiteral = APValue();

  // Constructors
  RootSignatureToken(clang::SourceLocation TokLoc) : TokLoc(TokLoc) {}
};
using TokenKind = enum RootSignatureToken::Kind;

class RootSignatureLexer {
public:
  RootSignatureLexer(StringRef Signature, clang::SourceLocation SourceLoc,
                     clang::Preprocessor &PP)
      : Buffer(Signature), SourceLoc(SourceLoc), PP(PP) {}

  // Consumes the internal buffer as a list of tokens and will emplace them
  // onto the given tokens.
  //
  // It will consume until it successfully reaches the end of the buffer,
  // or, until the first error is encountered. The return value denotes if
  // there was a failure.
  bool Lex(SmallVector<RootSignatureToken> &Tokens);

private:
  // Internal buffer to iterate over
  StringRef Buffer;

  // Passed down parameters from Sema
  clang::SourceLocation SourceLoc;
  clang::Preprocessor &PP;

  bool LexNumber(RootSignatureToken &Result);

  // Consumes the internal buffer for a single token.
  //
  // The return value denotes if there was a failure.
  bool LexToken(RootSignatureToken &Token);

  // Advance the buffer by the specified number of characters. Updates the
  // SourceLocation appropriately.
  void AdvanceBuffer(unsigned NumCharacters = 1) {
    Buffer = Buffer.drop_front(NumCharacters);
    SourceLoc = SourceLoc.getLocWithOffset(NumCharacters);
  }
};

} // namespace hlsl
} // namespace clang

#endif // LLVM_CLANG_PARSE_PARSEHLSLROOTSIGNATURE_H
