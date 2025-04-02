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
  bool parse();

private:
  DiagnosticsEngine &getDiags() { return PP.getDiagnostics(); }

  // All private Parse.* methods follow a similar pattern:
  //   - Each method will start with an assert to denote what the CurToken is
  // expected to be and will parse from that token forward
  //
  //   - Therefore, it is the callers responsibility to ensure that you are
  // at the correct CurToken. This should be done with the pattern of:
  //
  //  if (TryConsumeExpectedToken(RootSignatureToken::Kind))
  //    if (Parse.*())
  //      return true;
  //
  // or,
  //
  //  if (ConsumeExpectedToken(RootSignatureToken::Kind, ...))
  //    return true;
  //  if (Parse.*())
  //    return true;
  //
  //   - All methods return true if a parsing error is encountered. It is the
  // callers responsibility to propogate this error up, or deal with it
  // otherwise
  //
  //   - An error will be raised if the proceeding tokens are not what is
  // expected, or, there is a lexing error

  /// Root Element parse methods:
  bool parseDescriptorTable();
  bool parseDescriptorTableClause();

  /// Each unique ParamType will have a custom parse method defined that can be
  /// invoked to set a value to the referenced paramtype.
  ///
  /// This function will switch on the ParamType using std::visit and dispatch
  /// onto the corresponding parse method
  bool parseParam(llvm::hlsl::rootsig::ParamType Ref);

  /// Parameter arguments (eg. `bReg`, `space`, ...) can be specified in any
  /// order, exactly once, and only a subset are mandatory. This function acts
  /// as the infastructure to do so in a declarative way.
  ///
  /// For the example:
  ///  SmallDenseMap<RootSignatureToken::Kind, ParamType> Params = {
  ///    RootSignatureToken::Kind::bReg, &Clause.Register,
  ///    RootSignatureToken::Kind::kw_space, &Clause.Space
  ///  };
  ///  SmallDenseSet<RootSignatureToken::Kind> Mandatory = {
  ///    RootSignatureToken::Kind::bReg
  ///  };
  ///
  /// We can read it is as:
  ///
  /// when 'b0' is encountered, invoke the parse method for the type
  ///   of &Clause.Register (Register *) and update the parameter
  /// when 'space' is encountered, invoke a parse method for the type
  ///   of &Clause.Space (uint32_t *) and update the parameter
  ///
  /// and 'bReg' must be specified
  bool parseParams(llvm::SmallDenseMap<RootSignatureToken::Kind,
                                       llvm::hlsl::rootsig::ParamType> &Params,
                   llvm::SmallDenseSet<RootSignatureToken::Kind> &Mandatory);

  /// Parameter parse methods corresponding to a ParamType
  bool parseUIntParam(uint32_t *X);
  bool parseRegister(llvm::hlsl::rootsig::Register *Reg);

  /// Use NumericLiteralParser to convert CurToken.NumSpelling into a unsigned
  /// 32-bit integer
  bool handleUIntLiteral(uint32_t *X);

  /// Invoke the Lexer to consume a token and update CurToken with the result
  void consumeNextToken() { CurToken = Lexer.ConsumeToken(); }

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

private:
  SmallVector<llvm::hlsl::rootsig::RootElement> &Elements;
  RootSignatureLexer &Lexer;

  clang::Preprocessor &PP;

  RootSignatureToken CurToken;
};

} // namespace hlsl
} // namespace clang

#endif // LLVM_CLANG_PARSE_PARSEHLSLROOTSIGNATURE_H
