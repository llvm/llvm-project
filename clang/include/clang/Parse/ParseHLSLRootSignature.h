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

#include "clang/Basic/DiagnosticParse.h"
#include "clang/Lex/LexHLSLRootSignature.h"
#include "clang/Lex/Preprocessor.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include "llvm/Frontend/HLSL/HLSLRootSignature.h"

namespace clang {
namespace hlsl {

class RootSignatureParser {
public:
  RootSignatureParser(SmallVector<llvm::hlsl::rootsig::RootElement> &Elements,
                      RootSignatureLexer &Lexer, clang::Preprocessor &PP);

  /// Consumes tokens from the Lexer and constructs the in-memory
  /// representations of the RootElements. Tokens are consumed until an
  /// error is encountered or the end of the buffer.
  ///
  /// Returns true if a parsing error is encountered.
  bool Parse();

  DiagnosticsEngine &Diags() { return PP.getDiagnostics(); }

private:
  /// All private Parse.* methods follow a similar pattern:
  ///   - Each method will expect that the the next token is of a certain kind
  /// and will invoke `ConsumeExpectedToken`
  ///   - As such, an error will be raised if the proceeding tokens are not
  /// what is expected
  ///   - Therefore, it is the callers responsibility to ensure that you are
  /// expecting the next element type. Or equivalently, the methods should not
  /// be called as a way to try and parse an element
  ///   - All methods return true if a parsing error is encountered. It is the
  /// callers responsibility to propogate this error up, or deal with it
  /// otherwise
  bool ParseRootElement();
  bool ParseDescriptorTable();
  bool ParseDescriptorTableClause();

  /// It is helpful to have a generalized dispatch method so that when we need
  /// to parse multiple optional parameters in any order, we can invoke this
  /// method.
  ///
  /// Each unique ParamType is expected to define a custom Parse method. This
  /// function will switch on the ParamType using std::visit and dispatch onto
  /// the corresponding Parse method
  bool ParseParam(llvm::hlsl::rootsig::ParamType Ref);

  /// Parses as many optional parameters as possible in any order
  bool ParseOptionalParams(
      llvm::SmallDenseMap<TokenKind, llvm::hlsl::rootsig::ParamType> &RefMap);

  /// Use NumericLiteralParser to convert CurToken.NumSpelling into a unsigned
  /// 32-bit integer
  bool HandleUIntLiteral(uint32_t &X);
  bool ParseRegister(llvm::hlsl::rootsig::Register *Reg);
  bool ParseUInt(uint32_t *X);
  bool
  ParseDescriptorRangeOffset(llvm::hlsl::rootsig::DescriptorRangeOffset *X);

  /// Method for parsing any type of the ENUM defined token kinds (from
  /// HLSLRootSignatureTokenKinds.def)
  ///
  /// EnumMap provides a mapping from the unique TokenKind to the in-memory
  /// enum value
  ///
  /// If AllowZero is true, then the Enum is used as a flag and can also have
  /// the value of '0' to denote no flag
  template <bool AllowZero = false, typename EnumType>
  bool ParseEnum(llvm::SmallDenseMap<TokenKind, EnumType> &EnumMap,
                 EnumType *Enum);

  /// Helper methods that define the mappings and invoke ParseEnum for
  /// different enum types
  bool ParseShaderVisibility(llvm::hlsl::rootsig::ShaderVisibility *Enum);

  /// A wrapper method around ParseEnum that will parse an 'or' chain of
  /// enums, with AllowZero = true
  template <typename FlagType>
  bool ParseFlags(llvm::SmallDenseMap<TokenKind, FlagType> &EnumMap,
                  FlagType *Enum);

  /// Helper methods that define the mappings and invoke ParseFlags for
  /// different enum types
  bool
  ParseDescriptorRangeFlags(llvm::hlsl::rootsig::DescriptorRangeFlags *Enum);

  /// Invoke the Lexer to consume a token and update CurToken with the result
  void ConsumeNextToken() { CurToken = Lexer.ConsumeToken(); }

  /// Return true if the next token one of the expected kinds
  bool PeekExpectedToken(TokenKind Expected);
  bool PeekExpectedToken(ArrayRef<TokenKind> AnyExpected);

  /// Consumes the next token and report an error if it is not of the expected
  /// kind.
  ///
  /// Returns true if there was an error reported.
  bool ConsumeExpectedToken(TokenKind Expected);
  bool ConsumeExpectedToken(ArrayRef<TokenKind> AnyExpected);

  /// Peek if the next token is of the expected kind and if it is then consume
  /// it.
  ///
  /// Returns true if it successfully matches the expected kind and the token
  /// was consumed.
  bool TryConsumeExpectedToken(TokenKind Expected);
  bool TryConsumeExpectedToken(ArrayRef<TokenKind> Expected);

private:
  SmallVector<llvm::hlsl::rootsig::RootElement> &Elements;
  RootSignatureLexer &Lexer;

  clang::Preprocessor &PP;

  RootSignatureToken CurToken;
};

} // namespace hlsl
} // namespace clang

#endif // LLVM_CLANG_PARSE_PARSEHLSLROOTSIGNATURE_H
