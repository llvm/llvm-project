//===--- ParseHLSLRootSignature.h -------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  This file defines the RootSignatureParser interface.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_PARSE_PARSEHLSLROOTSIGNATURE_H
#define LLVM_CLANG_PARSE_PARSEHLSLROOTSIGNATURE_H

#include "clang/AST/Expr.h"
#include "clang/Basic/DiagnosticParse.h"
#include "clang/Lex/LexHLSLRootSignature.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Sema/SemaHLSL.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include "llvm/Frontend/HLSL/HLSLRootSignature.h"

namespace clang {
namespace hlsl {

class RootSignatureParser {
public:
  RootSignatureParser(llvm::dxbc::RootSignatureVersion Version,
                      StringLiteral *Signature, Preprocessor &PP);

  /// Consumes tokens from the Lexer and constructs the in-memory
  /// representations of the RootElements. Tokens are consumed until an
  /// error is encountered or the end of the buffer.
  ///
  /// Returns true if a parsing error is encountered.
  bool parse();

  /// Return all elements that have been parsed.
  ArrayRef<RootSignatureElement> getElements() { return Elements; }

private:
  DiagnosticsEngine &getDiags() { return PP.getDiagnostics(); }

  // All private parse.* methods follow a similar pattern:
  //   - Each method will start with an assert to denote what the CurToken is
  // expected to be and will parse from that token forward
  //
  //   - Therefore, it is the callers responsibility to ensure that you are
  // at the correct CurToken. This should be done with the pattern of:
  //
  //  if (tryConsumeExpectedToken(RootSignatureToken::Kind)) {
  //    auto ParsedObject = parse.*();
  //    if (!ParsedObject.has_value())
  //      return std::nullopt;
  //    ...
  // }
  //
  // or,
  //
  //  if (consumeExpectedToken(RootSignatureToken::Kind, ...))
  //    return std::nullopt;
  //  auto ParsedObject = parse.*();
  //  if (!ParsedObject.has_value())
  //    return std::nullopt;
  //  ...
  //
  //   - All methods return std::nullopt if a parsing error is encountered. It
  // is the callers responsibility to propogate this error up, or deal with it
  // otherwise
  //
  //   - An error will be raised if the proceeding tokens are not what is
  // expected, or, there is a lexing error

  /// Root Element parse methods:
  std::optional<llvm::dxbc::RootFlags> parseRootFlags();
  std::optional<llvm::hlsl::rootsig::RootConstants> parseRootConstants();
  std::optional<llvm::hlsl::rootsig::RootDescriptor> parseRootDescriptor();
  std::optional<llvm::hlsl::rootsig::DescriptorTable> parseDescriptorTable();
  std::optional<llvm::hlsl::rootsig::DescriptorTableClause>
  parseDescriptorTableClause();
  std::optional<llvm::hlsl::rootsig::StaticSampler> parseStaticSampler();

  /// Parameter arguments (eg. `bReg`, `space`, ...) can be specified in any
  /// order and only exactly once. The following methods define a
  /// `Parsed.*Params` struct to denote the current state of parsed params
  struct ParsedConstantParams {
    std::optional<llvm::hlsl::rootsig::Register> Reg;
    std::optional<uint32_t> Num32BitConstants;
    std::optional<uint32_t> Space;
    std::optional<llvm::dxbc::ShaderVisibility> Visibility;
  };
  std::optional<ParsedConstantParams> parseRootConstantParams();

  struct ParsedRootDescriptorParams {
    std::optional<llvm::hlsl::rootsig::Register> Reg;
    std::optional<uint32_t> Space;
    std::optional<llvm::dxbc::ShaderVisibility> Visibility;
    std::optional<llvm::dxbc::RootDescriptorFlags> Flags;
  };
  std::optional<ParsedRootDescriptorParams>
  parseRootDescriptorParams(RootSignatureToken::Kind DescKind,
                            RootSignatureToken::Kind RegType);

  struct ParsedClauseParams {
    std::optional<llvm::hlsl::rootsig::Register> Reg;
    std::optional<uint32_t> NumDescriptors;
    std::optional<uint32_t> Space;
    std::optional<uint32_t> Offset;
    std::optional<llvm::dxbc::DescriptorRangeFlags> Flags;
  };
  std::optional<ParsedClauseParams>
  parseDescriptorTableClauseParams(RootSignatureToken::Kind ClauseKind,
                                   RootSignatureToken::Kind RegType);

  struct ParsedStaticSamplerParams {
    std::optional<llvm::hlsl::rootsig::Register> Reg;
    std::optional<llvm::dxbc::SamplerFilter> Filter;
    std::optional<llvm::dxbc::TextureAddressMode> AddressU;
    std::optional<llvm::dxbc::TextureAddressMode> AddressV;
    std::optional<llvm::dxbc::TextureAddressMode> AddressW;
    std::optional<float> MipLODBias;
    std::optional<uint32_t> MaxAnisotropy;
    std::optional<llvm::dxbc::ComparisonFunc> CompFunc;
    std::optional<llvm::dxbc::StaticBorderColor> BorderColor;
    std::optional<float> MinLOD;
    std::optional<float> MaxLOD;
    std::optional<uint32_t> Space;
    std::optional<llvm::dxbc::ShaderVisibility> Visibility;
  };
  std::optional<ParsedStaticSamplerParams> parseStaticSamplerParams();

  // Common parsing methods
  std::optional<uint32_t> parseUIntParam();
  std::optional<llvm::hlsl::rootsig::Register> parseRegister();
  std::optional<float> parseFloatParam();

  /// Parsing methods of various enums
  std::optional<llvm::dxbc::ShaderVisibility>
  parseShaderVisibility(RootSignatureToken::Kind Context);
  std::optional<llvm::dxbc::SamplerFilter>
  parseSamplerFilter(RootSignatureToken::Kind Context);
  std::optional<llvm::dxbc::TextureAddressMode>
  parseTextureAddressMode(RootSignatureToken::Kind Context);
  std::optional<llvm::dxbc::ComparisonFunc>
  parseComparisonFunc(RootSignatureToken::Kind Context);
  std::optional<llvm::dxbc::StaticBorderColor>
  parseStaticBorderColor(RootSignatureToken::Kind Context);
  std::optional<llvm::dxbc::RootDescriptorFlags>
  parseRootDescriptorFlags(RootSignatureToken::Kind Context);
  std::optional<llvm::dxbc::DescriptorRangeFlags>
  parseDescriptorRangeFlags(RootSignatureToken::Kind Context);

  /// Use NumericLiteralParser to convert CurToken.NumSpelling into a unsigned
  /// 32-bit integer
  std::optional<uint32_t> handleUIntLiteral();
  /// Use NumericLiteralParser to convert CurToken.NumSpelling into a signed
  /// 32-bit integer
  std::optional<int32_t> handleIntLiteral(bool Negated);
  /// Use NumericLiteralParser to convert CurToken.NumSpelling into a float
  ///
  /// This matches the behaviour of DXC, which is as follows:
  ///  - convert the spelling with `strtod`
  ///  - check for a float overflow
  ///  - cast the double to a float
  /// The behaviour of `strtod` is replicated using:
  ///  Semantics: llvm::APFloat::Semantics::S_IEEEdouble
  ///  RoundingMode: llvm::RoundingMode::NearestTiesToEven
  std::optional<float> handleFloatLiteral(bool Negated);

  /// Flags may specify the value of '0' to denote that there should be no
  /// flags set.
  ///
  /// Return true if the current int_literal token is '0', otherwise false
  bool verifyZeroFlag();

  /// Invoke the Lexer to consume a token and update CurToken with the result
  void consumeNextToken() { CurToken = Lexer.consumeToken(); }

  /// Return true if the next token one of the expected kinds
  bool peekExpectedToken(RootSignatureToken::Kind Expected);
  bool peekExpectedToken(ArrayRef<RootSignatureToken::Kind> AnyExpected);

  /// Consumes the next token and report an error if it is not of the expected
  /// kind.
  ///
  /// Returns true if there was an error reported.
  bool consumeExpectedToken(
      RootSignatureToken::Kind Expected, unsigned DiagID = diag::err_expected,
      RootSignatureToken::Kind Context = RootSignatureToken::Kind::invalid);

  /// Peek if the next token is of the expected kind and if it is then consume
  /// it.
  ///
  /// Returns true if it successfully matches the expected kind and the token
  /// was consumed.
  bool tryConsumeExpectedToken(RootSignatureToken::Kind Expected);
  bool tryConsumeExpectedToken(ArrayRef<RootSignatureToken::Kind> Expected);

  /// Consume tokens until the expected token has been peeked to be next
  /// or we have reached the end of the stream. Note that this means the
  /// expected token will be the next token not CurToken.
  ///
  /// Returns true if it found a token of the given type.
  bool skipUntilExpectedToken(RootSignatureToken::Kind Expected);
  bool skipUntilExpectedToken(ArrayRef<RootSignatureToken::Kind> Expected);

  /// Consume tokens until we reach a closing right paren, ')', or, until we
  /// have reached the end of the stream. This will place the current token
  /// to be the end of stream or the right paren.
  ///
  /// Returns true if it is closed before the end of stream.
  bool skipUntilClosedParens(uint32_t NumParens = 1);

  /// Convert the token's offset in the signature string to its SourceLocation
  ///
  /// This allows to currently retrieve the location for multi-token
  /// StringLiterals
  SourceLocation getTokenLocation(RootSignatureToken Tok);

  /// Construct a diagnostics at the location of the current token
  DiagnosticBuilder reportDiag(unsigned DiagID) {
    return getDiags().Report(getTokenLocation(CurToken), DiagID);
  }

private:
  llvm::dxbc::RootSignatureVersion Version;
  SmallVector<RootSignatureElement> Elements;
  StringLiteral *Signature;
  RootSignatureLexer Lexer;
  Preprocessor &PP;

  RootSignatureToken CurToken;
};

IdentifierInfo *ParseHLSLRootSignature(Sema &Actions,
                                       llvm::dxbc::RootSignatureVersion Version,
                                       StringLiteral *Signature);

void HandleRootSignatureTarget(Sema &S, StringRef EntryRootSig);

} // namespace hlsl
} // namespace clang

#endif // LLVM_CLANG_PARSE_PARSEHLSLROOTSIGNATURE_H
