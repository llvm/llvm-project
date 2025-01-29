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

namespace rs = llvm::hlsl::root_signature;

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

class RootSignatureParser {
public:
  RootSignatureParser(SmallVector<rs::RootElement> &Elements,
                      const SmallVector<RootSignatureToken> &Tokens,
                      DiagnosticsEngine &Diags);

  // Iterates over the provided tokens and constructs the in-memory
  // representations of the RootElements.
  //
  // The return value denotes if there was a failure and the method will
  // return on the first encountered failure, or, return false if it
  // can sucessfully reach the end of the tokens.
  bool Parse();

private:
  // Root Element helpers
  bool ParseRootElement(bool First);
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
  bool ParseParam(rs::ParamType Ref);

  // Parse as many optional parameters as possible in any order
  bool
  ParseOptionalParams(llvm::SmallDenseMap<TokenKind, rs::ParamType> RefMap);

  // Common parsing helpers
  bool ParseRegister(rs::Register *Reg);
  bool ParseUInt(uint32_t *X);
  bool ParseDescriptorRangeOffset(rs::DescriptorRangeOffset *X);

  // Various flags/enum parsing helpers
  template <bool AllowZero = false, typename EnumType>
  bool ParseEnum(llvm::SmallDenseMap<TokenKind, EnumType> EnumMap,
                 EnumType *Enum);
  template <typename FlagType>
  bool ParseFlags(llvm::SmallDenseMap<TokenKind, FlagType> EnumMap,
                  FlagType *Enum);
  bool ParseDescriptorRangeFlags(rs::DescriptorRangeFlags *Enum);
  bool ParseShaderVisibility(rs::ShaderVisibility *Enum);

  // Increment the token iterator if we have not reached the end.
  // Return value denotes if we were already at the last token.
  bool ConsumeNextToken();

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

  // Consume the next token and report an error if it is not of the expected
  // kind.
  //
  // Return value denotes if it failed to match the expected kind, either it is
  // the end of the stream or it didn't match any of the expected kinds.
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
  SmallVector<rs::RootElement> &Elements;
  SmallVector<RootSignatureToken>::const_iterator CurTok;
  SmallVector<RootSignatureToken>::const_iterator LastTok;

  DiagnosticsEngine &Diags;
};

} // namespace hlsl
} // namespace clang

#endif // LLVM_CLANG_PARSE_PARSEHLSLROOTSIGNATURE_H
