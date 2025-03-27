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
#include "clang/Lex/Preprocessor.h"
#include "clang/Lex/LexHLSLRootSignature.h"

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

private:
  DiagnosticsEngine &Diags() { return PP.getDiagnostics(); }

  /// Root Element parse methods:
  bool ParseDescriptorTable();
  bool ParseDescriptorTableClause();

  /// Invoke the Lexer to consume a token and update CurToken with the result
  void ConsumeNextToken() { CurToken = Lexer.ConsumeToken(); }

  /// Return true if the next token one of the expected kinds
  bool PeekExpectedToken(TokenKind Expected);
  bool PeekExpectedToken(ArrayRef<TokenKind> AnyExpected);

  /// Consumes the next token and report an error if it is not of the expected
  /// kind.
  ///
  /// Returns true if there was an error reported.
  bool ConsumeExpectedToken(TokenKind Expected,
                            unsigned DiagID = diag::err_expected,
                            TokenKind Context = TokenKind::invalid);
  bool ConsumeExpectedToken(ArrayRef<TokenKind> AnyExpected,
                            unsigned DiagID = diag::err_expected,
                            TokenKind Context = TokenKind::invalid);

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
