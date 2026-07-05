//===--- LexHLSLRootSignature.h ---------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  This file defines the LexHLSLRootSignature interface.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LEX_LEXHLSLROOTSIGNATURE_H
#define LLVM_CLANG_LEX_LEXHLSLROOTSIGNATURE_H

#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/SourceLocation.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSwitch.h"

namespace clang {
namespace hlsl {

struct RootSignatureToken {
  enum Kind {
#define TOK(X, SPELLING) X,
#include "clang/Lex/HLSLRootSignatureTokenKinds.def"
  };

  Kind TokKind = Kind::invalid;

  // Retain the location offset of the token in the Signature
  // string
  uint32_t LocOffset;

  // Retain spelling of an numeric constant to be parsed later
  StringRef NumSpelling;

  // Constructors
  RootSignatureToken(uint32_t LocOffset) : LocOffset(LocOffset) {}
  RootSignatureToken(Kind TokKind, uint32_t LocOffset)
      : TokKind(TokKind), LocOffset(LocOffset) {}
};

inline const DiagnosticBuilder &
operator<<(const DiagnosticBuilder &DB, const RootSignatureToken::Kind Kind) {
  switch (Kind) {
#define TOK(X, SPELLING)                                                       \
  case RootSignatureToken::Kind::X:                                            \
    DB << SPELLING;                                                            \
    break;
#define PUNCTUATOR(X, SPELLING)                                                \
  case RootSignatureToken::Kind::pu_##X:                                       \
    DB << #SPELLING;                                                           \
    break;
#include "clang/Lex/HLSLRootSignatureTokenKinds.def"
  }
  return DB;
}

class RootSignatureLexer {
public:
  RootSignatureLexer(StringRef Signature) : Buffer(Signature) {}

  /// Consumes and returns the next token.
  RootSignatureToken consumeToken();

  /// Returns the token that proceeds CurToken
  RootSignatureToken peekNextToken();

  bool isEndOfBuffer() {
    advanceBuffer(Buffer.take_while(isspace).size());
    return Buffer.empty();
  }

private:
  // Internal buffer state
  StringRef Buffer;
  uint32_t LocOffset = 0;

  // Current peek state
  std::optional<RootSignatureToken> NextToken = std::nullopt;

  /// Consumes the buffer and returns the lexed token.
  RootSignatureToken lexToken();

  /// Advance the buffer by the specified number of characters.
  /// Updates the SourceLocation appropriately.
  void advanceBuffer(unsigned NumCharacters = 1) {
    Buffer = Buffer.drop_front(NumCharacters);
    LocOffset += NumCharacters;
  }
};

} // namespace hlsl
} // namespace clang

#endif // LLVM_CLANG_LEX_PARSEHLSLROOTSIGNATURE_H
