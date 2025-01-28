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
#include "clang/Basic/DiagnosticParse.h"
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
  // Root Element helpers
  bool ParseRootElement();
  bool ParseDescriptorTable();
  bool ParseDescriptorTableClause();

  // Helper dispatch method
  //
  // These will switch on the Variant kind to dispatch to the respective Parse
  // method and store the parsed value back into Ref.
  //
  // It is helpful to have a generalized dispatch method so that when we need
  // to parse multiple optional parameters in any order, we can invoke this
  // method
  bool ParseParam(llvm::hlsl::rootsig::ParamType Ref);

  // Parse as many optional parameters as possible in any order
  bool ParseOptionalParams(
      llvm::SmallDenseMap<TokenKind, llvm::hlsl::rootsig::ParamType> &RefMap);

  // Common parsing helpers
  bool ParseRegister(llvm::hlsl::rootsig::Register *Reg);
  bool ParseUInt(uint32_t *X);
  bool
  ParseDescriptorRangeOffset(llvm::hlsl::rootsig::DescriptorRangeOffset *X);

  // Various flags/enum parsing helpers
  template <bool AllowZero = false, typename EnumType>
  bool ParseEnum(llvm::SmallDenseMap<TokenKind, EnumType> &EnumMap,
                 EnumType *Enum);
  bool ParseShaderVisibility(llvm::hlsl::rootsig::ShaderVisibility *Enum);

  /// Invoke the lexer to consume a token and update CurToken with the result
  ///
  /// Return value denotes if we were already at the last token.
  ///
  /// This is used to avoid having to constantly access the Lexer's CurToken
  bool ConsumeNextToken() {
    if (Lexer.ConsumeToken())
      return true; // Report lexer err
    CurToken = Lexer.GetCurToken();
    return false;
  }

  // Attempt to retrieve the next token, if TokenKind is invalid then there was
  // no next token.
  RootSignatureToken PeekNextToken();

  // Is the current token one of the expected kinds
  bool EnsureExpectedToken(TokenKind AnyExpected);
  bool EnsureExpectedToken(ArrayRef<TokenKind> AnyExpected);

  // Peek if the next token is of the expected kind.
  //
  // Return value denotes if it failed to match the expected kind, either it is
  // the end of the stream or it didn't match any of the expected kinds.
  bool PeekExpectedToken(TokenKind Expected);
  bool PeekExpectedToken(ArrayRef<TokenKind> AnyExpected);

  /// Consume the next token and report an error if it is not of the expected
  /// kind.
  ///
  /// Return value denotes if it failed to match the expected kind, either it is
  /// the end of the stream or it didn't match any of the expected kinds.
  bool ConsumeExpectedToken(TokenKind Expected);
  bool ConsumeExpectedToken(ArrayRef<TokenKind> AnyExpected);

  // Peek if the next token is of the expected kind and if it is then consume
  // it.
  //
  // Return value denotes if it failed to match the expected kind, either it is
  // the end of the stream or it didn't match any of the expected kinds. It will
  // not report an error if there isn't a match.
  bool TryConsumeExpectedToken(TokenKind Expected);
  bool TryConsumeExpectedToken(ArrayRef<TokenKind> Expected);

private:
  SmallVector<llvm::hlsl::rootsig::RootElement> &Elements;
  RootSignatureLexer &Lexer;
  DiagnosticsEngine &Diags;

  RootSignatureToken CurToken;
};

} // namespace hlsl
} // namespace clang

#endif // LLVM_CLANG_PARSE_PARSEHLSLROOTSIGNATURE_H
