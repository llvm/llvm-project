//=== ParseHLSLRootSignature.cpp - Parse Root Signature -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Parse/ParseHLSLRootSignature.h"

#include "clang/Lex/LiteralSupport.h"

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

  if (consumeExpectedToken(TokenKind::end_of_stream,
                           diag::err_hlsl_unexpected_end_of_params,
                           /*param of=*/TokenKind::kw_RootSignature))
    return true;

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

  if (consumeExpectedToken(TokenKind::pu_r_paren,
                           diag::err_hlsl_unexpected_end_of_params,
                           /*param of=*/TokenKind::kw_DescriptorTable))
    return true;

  Elements.push_back(Table);
  return false;
}

bool RootSignatureParser::parseDescriptorTableClause() {
  assert((CurToken.TokKind == TokenKind::kw_CBV ||
          CurToken.TokKind == TokenKind::kw_SRV ||
          CurToken.TokKind == TokenKind::kw_UAV ||
          CurToken.TokKind == TokenKind::kw_Sampler) &&
         "Expects to only be invoked starting at given keyword");
  TokenKind ParamKind = CurToken.TokKind; // retain for diagnostics

  DescriptorTableClause Clause;
  TokenKind ExpectedRegister;
  switch (ParamKind) {
  default:
    llvm_unreachable("Switch for consumed token was not provided");
  case TokenKind::kw_CBV:
    Clause.Type = ClauseType::CBuffer;
    ExpectedRegister = TokenKind::bReg;
    break;
  case TokenKind::kw_SRV:
    Clause.Type = ClauseType::SRV;
    ExpectedRegister = TokenKind::tReg;
    break;
  case TokenKind::kw_UAV:
    Clause.Type = ClauseType::UAV;
    ExpectedRegister = TokenKind::uReg;
    break;
  case TokenKind::kw_Sampler:
    Clause.Type = ClauseType::Sampler;
    ExpectedRegister = TokenKind::sReg;
    break;
  }

  if (consumeExpectedToken(TokenKind::pu_l_paren, diag::err_expected_after,
                           ParamKind))
    return true;

  TokenKind Keywords[2] = {
      ExpectedRegister,
      TokenKind::kw_space,
  };
  ParsedParamState Params(Keywords, ParamKind, CurToken.TokLoc);
  if (parseParams(Params))
    return true;

  // Mandatory parameters:
  if (!Params.checkAndClearSeen(ExpectedRegister)) {
    getDiags().Report(CurToken.TokLoc, diag::err_hlsl_rootsig_missing_param)
        << ExpectedRegister;
    return true;
  }

  Clause.Register = Params.Register;

  // Optional parameters:
  if (Params.checkAndClearSeen(TokenKind::kw_space))
    Clause.Space = Params.Space;

  if (consumeExpectedToken(TokenKind::pu_r_paren,
                           diag::err_hlsl_unexpected_end_of_params,
                           /*param of=*/ParamKind))
    return true;

  Elements.push_back(Clause);
  return false;
}

size_t RootSignatureParser::ParsedParamState::getKeywordIdx(
    RootSignatureToken::Kind Keyword) {
  ArrayRef KeywordRef = Keywords;
  auto It = llvm::find(KeywordRef, Keyword);
  assert(It != KeywordRef.end() && "Did not provide a valid param keyword");
  return std::distance(KeywordRef.begin(), It);
}

bool RootSignatureParser::ParsedParamState::checkAndSetSeen(
    RootSignatureToken::Kind Keyword) {
  size_t Idx = getKeywordIdx(Keyword);
  bool WasSeen = Seen & (1 << Idx);
  Seen |= 1u << Idx;
  return WasSeen;
}

bool RootSignatureParser::ParsedParamState::checkAndClearSeen(
    RootSignatureToken::Kind Keyword) {
  size_t Idx = getKeywordIdx(Keyword);
  bool WasSeen = Seen & (1 << Idx);
  Seen &= ~(1u << Idx);
  return WasSeen;
}

bool RootSignatureParser::parseParam(ParsedParamState &Params) {
  TokenKind Keyword = CurToken.TokKind;
  if (Keyword == TokenKind::bReg || Keyword == TokenKind::tReg ||
      Keyword == TokenKind::uReg || Keyword == TokenKind::sReg) {
    return parseRegister(Params.Register);
  }

  if (consumeExpectedToken(TokenKind::pu_equal, diag::err_expected_after,
                           Keyword))
    return true;

  switch (Keyword) {
  case RootSignatureToken::Kind::kw_space:
    return parseUIntParam(Params.Space);
  default:
    llvm_unreachable("Switch for consumed keyword was not provided");
  }
}

bool RootSignatureParser::parseParams(ParsedParamState &Params) {
  assert(CurToken.TokKind == TokenKind::pu_l_paren &&
         "Expects to only be invoked starting at given token");

  while (tryConsumeExpectedToken(Params.Keywords)) {
    if (Params.checkAndSetSeen(CurToken.TokKind)) {
      getDiags().Report(CurToken.TokLoc, diag::err_hlsl_rootsig_repeat_param)
          << CurToken.TokKind;
      return true;
    }

    if (parseParam(Params))
      return true;

    if (!tryConsumeExpectedToken(TokenKind::pu_comma))
      break;
  }

  return false;
}

bool RootSignatureParser::parseUIntParam(uint32_t &X) {
  assert(CurToken.TokKind == TokenKind::pu_equal &&
         "Expects to only be invoked starting at given keyword");
  tryConsumeExpectedToken(TokenKind::pu_plus);
  return consumeExpectedToken(TokenKind::int_literal, diag::err_expected_after,
                              CurToken.TokKind) ||
         handleUIntLiteral(X);
}

bool RootSignatureParser::parseRegister(Register &Register) {
  assert((CurToken.TokKind == TokenKind::bReg ||
          CurToken.TokKind == TokenKind::tReg ||
          CurToken.TokKind == TokenKind::uReg ||
          CurToken.TokKind == TokenKind::sReg) &&
         "Expects to only be invoked starting at given keyword");

  switch (CurToken.TokKind) {
  default:
    llvm_unreachable("Switch for consumed token was not provided");
  case TokenKind::bReg:
    Register.ViewType = RegisterType::BReg;
    break;
  case TokenKind::tReg:
    Register.ViewType = RegisterType::TReg;
    break;
  case TokenKind::uReg:
    Register.ViewType = RegisterType::UReg;
    break;
  case TokenKind::sReg:
    Register.ViewType = RegisterType::SReg;
    break;
  }

  if (handleUIntLiteral(Register.Number))
    return true; // propogate NumericLiteralParser error

  return false;
}

bool RootSignatureParser::handleUIntLiteral(uint32_t &X) {
  // Parse the numeric value and do semantic checks on its specification
  clang::NumericLiteralParser Literal(CurToken.NumSpelling, CurToken.TokLoc,
                                      PP.getSourceManager(), PP.getLangOpts(),
                                      PP.getTargetInfo(), PP.getDiagnostics());
  if (Literal.hadError)
    return true; // Error has already been reported so just return

  assert(Literal.isIntegerLiteral() && "IsNumberChar will only support digits");

  llvm::APSInt Val = llvm::APSInt(32, false);
  if (Literal.GetIntegerValue(Val)) {
    // Report that the value has overflowed
    PP.getDiagnostics().Report(CurToken.TokLoc,
                               diag::err_hlsl_number_literal_overflow)
        << 0 << CurToken.NumSpelling;
    return true;
  }

  X = Val.getExtValue();
  return false;
}

bool RootSignatureParser::peekExpectedToken(TokenKind Expected) {
  return peekExpectedToken(ArrayRef{Expected});
}

bool RootSignatureParser::peekExpectedToken(ArrayRef<TokenKind> AnyExpected) {
  RootSignatureToken Result = Lexer.PeekNextToken();
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
  case diag::err_hlsl_unexpected_end_of_params:
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
