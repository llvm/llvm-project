//=== ParseHLSLRootSignature.cpp - Parse Root Signature -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Parse/ParseHLSLRootSignature.h"

#include "llvm/Support/raw_ostream.h"

using namespace llvm::hlsl::rootsig;

namespace clang {
namespace hlsl {

using TokenKind = RootSignatureToken::Kind;

RootSignatureParser::RootSignatureParser(SmallVector<RootElement> &Elements,
                                         RootSignatureLexer &Lexer,
                                         Preprocessor &PP)
    : Elements(Elements), Lexer(Lexer), PP(PP), CurToken(SourceLocation()) {}

bool RootSignatureParser::parse() {
  // Iterate as many RootElements as possible
  while (tryConsumeExpectedToken(TokenKind::kw_DescriptorTable)) {
    // Dispatch onto parser method.
    // We guard against the unreachable here as we just ensured that CurToken
    // will be one of the kinds in the while condition
    switch (CurToken.TokKind) {
    case TokenKind::kw_DescriptorTable:
      if (parseDescriptorTable())
        return true;
      break;
    default:
      llvm_unreachable("Switch for consumed token was not provided");
    }

    if (!tryConsumeExpectedToken(TokenKind::pu_comma))
      break;
  }

  if (!tryConsumeExpectedToken(TokenKind::end_of_stream)) {
    getDiags().Report(CurToken.TokLoc, diag::err_hlsl_unexpected_end_of_params)
        << /*expected=*/TokenKind::end_of_stream
        << /*param of=*/TokenKind::kw_RootSignature;
    return true;
  }
  return false;
}

bool RootSignatureParser::parseDescriptorTable() {
  assert(CurToken.TokKind == TokenKind::kw_DescriptorTable &&
         "Expects to only be invoked starting at given keyword");

  DescriptorTable Table;

  if (consumeExpectedToken(TokenKind::pu_l_paren, diag::err_expected_after,
                           CurToken.TokKind))
    return true;

  // Iterate as many Clauses as possible
  while (tryConsumeExpectedToken({TokenKind::kw_CBV, TokenKind::kw_SRV,
                                  TokenKind::kw_UAV, TokenKind::kw_Sampler})) {
    if (parseDescriptorTableClause())
      return true;

    Table.NumClauses++;

    if (!tryConsumeExpectedToken(TokenKind::pu_comma))
      break;
  }

  if (!tryConsumeExpectedToken(TokenKind::pu_r_paren)) {
    getDiags().Report(CurToken.TokLoc, diag::err_hlsl_unexpected_end_of_params)
        << /*expected=*/TokenKind::pu_r_paren
        << /*param of=*/TokenKind::kw_DescriptorTable;
    return true;
  }

  Elements.push_back(Table);
  return false;
}

bool RootSignatureParser::parseDescriptorTableClause() {
  assert((CurToken.TokKind == TokenKind::kw_CBV ||
          CurToken.TokKind == TokenKind::kw_SRV ||
          CurToken.TokKind == TokenKind::kw_UAV ||
          CurToken.TokKind == TokenKind::kw_Sampler) &&
         "Expects to only be invoked starting at given keyword");

  DescriptorTableClause Clause;
  switch (CurToken.TokKind) {
  default:
    llvm_unreachable("Switch for consumed token was not provided");
  case TokenKind::kw_CBV:
    Clause.Type = ClauseType::CBuffer;
    break;
  case TokenKind::kw_SRV:
    Clause.Type = ClauseType::SRV;
    break;
  case TokenKind::kw_UAV:
    Clause.Type = ClauseType::UAV;
    break;
  case TokenKind::kw_Sampler:
    Clause.Type = ClauseType::Sampler;
    break;
  }

  if (consumeExpectedToken(TokenKind::pu_l_paren, diag::err_expected_after,
                           CurToken.TokKind))
    return true;

  if (consumeExpectedToken(TokenKind::pu_r_paren, diag::err_expected_after,
                           CurToken.TokKind))
    return true;

  Elements.push_back(Clause);
  return false;
}

bool RootSignatureParser::peekExpectedToken(TokenKind Expected) {
  return peekExpectedToken(ArrayRef{Expected});
}

bool RootSignatureParser::peekExpectedToken(ArrayRef<TokenKind> AnyExpected) {
  RootSignatureToken Result = Lexer.peekNextToken();
  return llvm::is_contained(AnyExpected, Result.TokKind);
}

bool RootSignatureParser::consumeExpectedToken(TokenKind Expected,
                                               unsigned DiagID,
                                               TokenKind Context) {
  if (tryConsumeExpectedToken(Expected))
    return false;

  // Report unexpected token kind error
  DiagnosticBuilder DB = getDiags().Report(CurToken.TokLoc, DiagID);
  switch (DiagID) {
  case diag::err_expected:
    DB << Expected;
    break;
  case diag::err_expected_either:
  case diag::err_expected_after:
    DB << Expected << Context;
    break;
  default:
    break;
  }
  return true;
}

bool RootSignatureParser::tryConsumeExpectedToken(TokenKind Expected) {
  return tryConsumeExpectedToken(ArrayRef{Expected});
}

bool RootSignatureParser::tryConsumeExpectedToken(
    ArrayRef<TokenKind> AnyExpected) {
  // If not the expected token just return
  if (!peekExpectedToken(AnyExpected))
    return false;
  consumeNextToken();
  return true;
}

} // namespace hlsl
} // namespace clang
