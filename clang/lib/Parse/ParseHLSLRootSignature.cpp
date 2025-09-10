//=== ParseHLSLRootSignature.cpp - Parse Root Signature -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Parse/ParseHLSLRootSignature.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/Lex/LiteralSupport.h"
#include "clang/Parse/Parser.h"
#include "clang/Sema/Sema.h"

using namespace llvm::hlsl::rootsig;

namespace clang {
namespace hlsl {

using TokenKind = RootSignatureToken::Kind;

static const TokenKind RootElementKeywords[] = {
    TokenKind::kw_RootFlags,
    TokenKind::kw_CBV,
    TokenKind::kw_UAV,
    TokenKind::kw_SRV,
    TokenKind::kw_DescriptorTable,
    TokenKind::kw_StaticSampler,
};

RootSignatureParser::RootSignatureParser(
    llvm::dxbc::RootSignatureVersion Version, StringLiteral *Signature,
    Preprocessor &PP)
    : Version(Version), Signature(Signature), Lexer(Signature->getString()),
      PP(PP), CurToken(0) {}

bool RootSignatureParser::parse() {
  // Iterate as many RootSignatureElements as possible, until we hit the
  // end of the stream
  bool HadError = false;
  bool HasRootFlags = false;
  while (!peekExpectedToken(TokenKind::end_of_stream)) {
    if (tryConsumeExpectedToken(TokenKind::kw_RootFlags)) {
      if (HasRootFlags) {
        reportDiag(diag::err_hlsl_rootsig_repeat_param)
            << TokenKind::kw_RootFlags;
        HadError = true;
        skipUntilExpectedToken(RootElementKeywords);
        continue;
      }
      HasRootFlags = true;

      SourceLocation ElementLoc = getTokenLocation(CurToken);
      auto Flags = parseRootFlags();
      if (!Flags.has_value()) {
        HadError = true;
        skipUntilExpectedToken(RootElementKeywords);
        continue;
      }

      Elements.emplace_back(ElementLoc, *Flags);
    } else if (tryConsumeExpectedToken(TokenKind::kw_RootConstants)) {
      SourceLocation ElementLoc = getTokenLocation(CurToken);
      auto Constants = parseRootConstants();
      if (!Constants.has_value()) {
        HadError = true;
        skipUntilExpectedToken(RootElementKeywords);
        continue;
      }
      Elements.emplace_back(ElementLoc, *Constants);
    } else if (tryConsumeExpectedToken(TokenKind::kw_DescriptorTable)) {
      SourceLocation ElementLoc = getTokenLocation(CurToken);
      auto Table = parseDescriptorTable();
      if (!Table.has_value()) {
        HadError = true;
        // We are within a DescriptorTable, we will do our best to recover
        // by skipping until we encounter the expected closing ')'.
        skipUntilClosedParens();
        consumeNextToken();
        skipUntilExpectedToken(RootElementKeywords);
        continue;
      }
      Elements.emplace_back(ElementLoc, *Table);
    } else if (tryConsumeExpectedToken(
                   {TokenKind::kw_CBV, TokenKind::kw_SRV, TokenKind::kw_UAV})) {
      SourceLocation ElementLoc = getTokenLocation(CurToken);
      auto Descriptor = parseRootDescriptor();
      if (!Descriptor.has_value()) {
        HadError = true;
        skipUntilExpectedToken(RootElementKeywords);
        continue;
      }
      Elements.emplace_back(ElementLoc, *Descriptor);
    } else if (tryConsumeExpectedToken(TokenKind::kw_StaticSampler)) {
      SourceLocation ElementLoc = getTokenLocation(CurToken);
      auto Sampler = parseStaticSampler();
      if (!Sampler.has_value()) {
        HadError = true;
        skipUntilExpectedToken(RootElementKeywords);
        continue;
      }
      Elements.emplace_back(ElementLoc, *Sampler);
    } else {
      HadError = true;
      consumeNextToken(); // let diagnostic be at the start of invalid token
      reportDiag(diag::err_hlsl_invalid_token)
          << /*parameter=*/0 << /*param of*/ TokenKind::kw_RootSignature;
      skipUntilExpectedToken(RootElementKeywords);
      continue;
    }

    if (!tryConsumeExpectedToken(TokenKind::pu_comma)) {
      // ',' denotes another element, otherwise, expected to be at end of stream
      break;
    }
  }

  return HadError ||
         consumeExpectedToken(TokenKind::end_of_stream,
                              diag::err_expected_either, TokenKind::pu_comma);
}

template <typename FlagType>
static FlagType maybeOrFlag(std::optional<FlagType> Flags, FlagType Flag) {
  if (!Flags.has_value())
    return Flag;

  return static_cast<FlagType>(llvm::to_underlying(Flags.value()) |
                               llvm::to_underlying(Flag));
}

std::optional<llvm::dxbc::RootFlags> RootSignatureParser::parseRootFlags() {
  assert(CurToken.TokKind == TokenKind::kw_RootFlags &&
         "Expects to only be invoked starting at given keyword");

  if (consumeExpectedToken(TokenKind::pu_l_paren, diag::err_expected_after,
                           CurToken.TokKind))
    return std::nullopt;

  std::optional<llvm::dxbc::RootFlags> Flags = llvm::dxbc::RootFlags::None;

  // Handle valid empty case
  if (tryConsumeExpectedToken(TokenKind::pu_r_paren))
    return Flags;

  // Handle the edge-case of '0' to specify no flags set
  if (tryConsumeExpectedToken(TokenKind::int_literal)) {
    if (!verifyZeroFlag()) {
      reportDiag(diag::err_hlsl_rootsig_non_zero_flag);
      return std::nullopt;
    }
  } else {
    // Otherwise, parse as many flags as possible
    TokenKind Expected[] = {
#define ROOT_FLAG_ENUM(NAME, LIT) TokenKind::en_##NAME,
#include "clang/Lex/HLSLRootSignatureTokenKinds.def"
    };

    do {
      if (tryConsumeExpectedToken(Expected)) {
        switch (CurToken.TokKind) {
#define ROOT_FLAG_ENUM(NAME, LIT)                                              \
  case TokenKind::en_##NAME:                                                   \
    Flags = maybeOrFlag<llvm::dxbc::RootFlags>(Flags,                          \
                                               llvm::dxbc::RootFlags::NAME);   \
    break;
#include "clang/Lex/HLSLRootSignatureTokenKinds.def"
        default:
          llvm_unreachable("Switch for consumed enum token was not provided");
        }
      } else {
        consumeNextToken(); // consume token to point at invalid token
        reportDiag(diag::err_hlsl_invalid_token)
            << /*value=*/1 << /*value of*/ TokenKind::kw_RootFlags;
        return std::nullopt;
      }
    } while (tryConsumeExpectedToken(TokenKind::pu_or));
  }

  if (consumeExpectedToken(TokenKind::pu_r_paren, diag::err_expected_either,
                           TokenKind::pu_comma))
    return std::nullopt;

  return Flags;
}

std::optional<RootConstants> RootSignatureParser::parseRootConstants() {
  assert(CurToken.TokKind == TokenKind::kw_RootConstants &&
         "Expects to only be invoked starting at given keyword");

  if (consumeExpectedToken(TokenKind::pu_l_paren, diag::err_expected_after,
                           CurToken.TokKind))
    return std::nullopt;

  RootConstants Constants;

  auto Params = parseRootConstantParams();
  if (!Params.has_value())
    return std::nullopt;

  if (consumeExpectedToken(TokenKind::pu_r_paren, diag::err_expected_either,
                           TokenKind::pu_comma))
    return std::nullopt;

  // Check mandatory parameters where provided
  if (!Params->Num32BitConstants.has_value()) {
    reportDiag(diag::err_hlsl_rootsig_missing_param)
        << TokenKind::kw_num32BitConstants;
    return std::nullopt;
  }

  Constants.Num32BitConstants = Params->Num32BitConstants.value();

  if (!Params->Reg.has_value()) {
    reportDiag(diag::err_hlsl_rootsig_missing_param) << TokenKind::bReg;
    return std::nullopt;
  }

  Constants.Reg = Params->Reg.value();

  // Fill in optional parameters
  if (Params->Visibility.has_value())
    Constants.Visibility = Params->Visibility.value();

  if (Params->Space.has_value())
    Constants.Space = Params->Space.value();

  return Constants;
}

std::optional<RootDescriptor> RootSignatureParser::parseRootDescriptor() {
  assert((CurToken.TokKind == TokenKind::kw_CBV ||
          CurToken.TokKind == TokenKind::kw_SRV ||
          CurToken.TokKind == TokenKind::kw_UAV) &&
         "Expects to only be invoked starting at given keyword");

  TokenKind DescriptorKind = CurToken.TokKind;

  if (consumeExpectedToken(TokenKind::pu_l_paren, diag::err_expected_after,
                           CurToken.TokKind))
    return std::nullopt;

  RootDescriptor Descriptor;
  TokenKind ExpectedReg;
  switch (DescriptorKind) {
  default:
    llvm_unreachable("Switch for consumed token was not provided");
  case TokenKind::kw_CBV:
    Descriptor.Type = ResourceClass::CBuffer;
    ExpectedReg = TokenKind::bReg;
    break;
  case TokenKind::kw_SRV:
    Descriptor.Type = ResourceClass::SRV;
    ExpectedReg = TokenKind::tReg;
    break;
  case TokenKind::kw_UAV:
    Descriptor.Type = ResourceClass::UAV;
    ExpectedReg = TokenKind::uReg;
    break;
  }
  Descriptor.setDefaultFlags(Version);

  auto Params = parseRootDescriptorParams(DescriptorKind, ExpectedReg);
  if (!Params.has_value())
    return std::nullopt;

  if (consumeExpectedToken(TokenKind::pu_r_paren, diag::err_expected_either,
                           TokenKind::pu_comma))
    return std::nullopt;

  // Check mandatory parameters were provided
  if (!Params->Reg.has_value()) {
    reportDiag(diag::err_hlsl_rootsig_missing_param) << ExpectedReg;
    return std::nullopt;
  }

  Descriptor.Reg = Params->Reg.value();

  // Fill in optional values
  if (Params->Space.has_value())
    Descriptor.Space = Params->Space.value();

  if (Params->Visibility.has_value())
    Descriptor.Visibility = Params->Visibility.value();

  if (Params->Flags.has_value())
    Descriptor.Flags = Params->Flags.value();

  return Descriptor;
}

std::optional<DescriptorTable> RootSignatureParser::parseDescriptorTable() {
  assert(CurToken.TokKind == TokenKind::kw_DescriptorTable &&
         "Expects to only be invoked starting at given keyword");

  if (consumeExpectedToken(TokenKind::pu_l_paren, diag::err_expected_after,
                           CurToken.TokKind))
    return std::nullopt;

  DescriptorTable Table;
  std::optional<llvm::dxbc::ShaderVisibility> Visibility;

  // Iterate as many Clauses as possible, until we hit ')'
  while (!peekExpectedToken(TokenKind::pu_r_paren)) {
    if (tryConsumeExpectedToken({TokenKind::kw_CBV, TokenKind::kw_SRV,
                                 TokenKind::kw_UAV, TokenKind::kw_Sampler})) {
      // DescriptorTableClause - CBV, SRV, UAV, or Sampler
      SourceLocation ElementLoc = getTokenLocation(CurToken);
      auto Clause = parseDescriptorTableClause();
      if (!Clause.has_value()) {
        // We are within a DescriptorTableClause, we will do our best to recover
        // by skipping until we encounter the expected closing ')'
        skipUntilExpectedToken(TokenKind::pu_r_paren);
        consumeNextToken();
        return std::nullopt;
      }
      Elements.emplace_back(ElementLoc, *Clause);
      Table.NumClauses++;
    } else if (tryConsumeExpectedToken(TokenKind::kw_visibility)) {
      // visibility = SHADER_VISIBILITY
      if (Visibility.has_value()) {
        reportDiag(diag::err_hlsl_rootsig_repeat_param) << CurToken.TokKind;
        return std::nullopt;
      }

      if (consumeExpectedToken(TokenKind::pu_equal))
        return std::nullopt;

      Visibility = parseShaderVisibility(TokenKind::kw_visibility);
      if (!Visibility.has_value())
        return std::nullopt;
    } else {
      consumeNextToken(); // let diagnostic be at the start of invalid token
      reportDiag(diag::err_hlsl_invalid_token)
          << /*parameter=*/0 << /*param of*/ TokenKind::kw_DescriptorTable;
      return std::nullopt;
    }

    // ',' denotes another element, otherwise, expected to be at ')'
    if (!tryConsumeExpectedToken(TokenKind::pu_comma))
      break;
  }

  if (consumeExpectedToken(TokenKind::pu_r_paren, diag::err_expected_either,
                           TokenKind::pu_comma))
    return std::nullopt;

  // Fill in optional visibility
  if (Visibility.has_value())
    Table.Visibility = Visibility.value();

  return Table;
}

std::optional<DescriptorTableClause>
RootSignatureParser::parseDescriptorTableClause() {
  assert((CurToken.TokKind == TokenKind::kw_CBV ||
          CurToken.TokKind == TokenKind::kw_SRV ||
          CurToken.TokKind == TokenKind::kw_UAV ||
          CurToken.TokKind == TokenKind::kw_Sampler) &&
         "Expects to only be invoked starting at given keyword");

  TokenKind ParamKind = CurToken.TokKind;

  if (consumeExpectedToken(TokenKind::pu_l_paren, diag::err_expected_after,
                           CurToken.TokKind))
    return std::nullopt;

  DescriptorTableClause Clause;
  TokenKind ExpectedReg;
  switch (ParamKind) {
  default:
    llvm_unreachable("Switch for consumed token was not provided");
  case TokenKind::kw_CBV:
    Clause.Type = ResourceClass::CBuffer;
    ExpectedReg = TokenKind::bReg;
    break;
  case TokenKind::kw_SRV:
    Clause.Type = ResourceClass::SRV;
    ExpectedReg = TokenKind::tReg;
    break;
  case TokenKind::kw_UAV:
    Clause.Type = ResourceClass::UAV;
    ExpectedReg = TokenKind::uReg;
    break;
  case TokenKind::kw_Sampler:
    Clause.Type = ResourceClass::Sampler;
    ExpectedReg = TokenKind::sReg;
    break;
  }
  Clause.setDefaultFlags(Version);

  auto Params = parseDescriptorTableClauseParams(ParamKind, ExpectedReg);
  if (!Params.has_value())
    return std::nullopt;

  if (consumeExpectedToken(TokenKind::pu_r_paren, diag::err_expected_either,
                           TokenKind::pu_comma))
    return std::nullopt;

  // Check mandatory parameters were provided
  if (!Params->Reg.has_value()) {
    reportDiag(diag::err_hlsl_rootsig_missing_param) << ExpectedReg;
    return std::nullopt;
  }

  Clause.Reg = Params->Reg.value();

  // Fill in optional values
  if (Params->NumDescriptors.has_value())
    Clause.NumDescriptors = Params->NumDescriptors.value();

  if (Params->Space.has_value())
    Clause.Space = Params->Space.value();

  if (Params->Offset.has_value())
    Clause.Offset = Params->Offset.value();

  if (Params->Flags.has_value())
    Clause.Flags = Params->Flags.value();

  return Clause;
}

std::optional<StaticSampler> RootSignatureParser::parseStaticSampler() {
  assert(CurToken.TokKind == TokenKind::kw_StaticSampler &&
         "Expects to only be invoked starting at given keyword");

  if (consumeExpectedToken(TokenKind::pu_l_paren, diag::err_expected_after,
                           CurToken.TokKind))
    return std::nullopt;

  StaticSampler Sampler;

  auto Params = parseStaticSamplerParams();
  if (!Params.has_value())
    return std::nullopt;

  if (consumeExpectedToken(TokenKind::pu_r_paren, diag::err_expected_either,
                           TokenKind::pu_comma))
    return std::nullopt;

  // Check mandatory parameters were provided
  if (!Params->Reg.has_value()) {
    reportDiag(diag::err_hlsl_rootsig_missing_param) << TokenKind::sReg;
    return std::nullopt;
  }

  Sampler.Reg = Params->Reg.value();

  // Fill in optional values
  if (Params->Filter.has_value())
    Sampler.Filter = Params->Filter.value();

  if (Params->AddressU.has_value())
    Sampler.AddressU = Params->AddressU.value();

  if (Params->AddressV.has_value())
    Sampler.AddressV = Params->AddressV.value();

  if (Params->AddressW.has_value())
    Sampler.AddressW = Params->AddressW.value();

  if (Params->MipLODBias.has_value())
    Sampler.MipLODBias = Params->MipLODBias.value();

  if (Params->MaxAnisotropy.has_value())
    Sampler.MaxAnisotropy = Params->MaxAnisotropy.value();

  if (Params->CompFunc.has_value())
    Sampler.CompFunc = Params->CompFunc.value();

  if (Params->BorderColor.has_value())
    Sampler.BorderColor = Params->BorderColor.value();

  if (Params->MinLOD.has_value())
    Sampler.MinLOD = Params->MinLOD.value();

  if (Params->MaxLOD.has_value())
    Sampler.MaxLOD = Params->MaxLOD.value();

  if (Params->Space.has_value())
    Sampler.Space = Params->Space.value();

  if (Params->Visibility.has_value())
    Sampler.Visibility = Params->Visibility.value();

  return Sampler;
}

// Parameter arguments (eg. `bReg`, `space`, ...) can be specified in any
// order and only exactly once. The following methods will parse through as
// many arguments as possible reporting an error if a duplicate is seen.
std::optional<RootSignatureParser::ParsedConstantParams>
RootSignatureParser::parseRootConstantParams() {
  assert(CurToken.TokKind == TokenKind::pu_l_paren &&
         "Expects to only be invoked starting at given token");

  ParsedConstantParams Params;
  while (!peekExpectedToken(TokenKind::pu_r_paren)) {
    if (tryConsumeExpectedToken(TokenKind::kw_num32BitConstants)) {
      // `num32BitConstants` `=` POS_INT
      if (Params.Num32BitConstants.has_value()) {
        reportDiag(diag::err_hlsl_rootsig_repeat_param) << CurToken.TokKind;
        return std::nullopt;
      }

      if (consumeExpectedToken(TokenKind::pu_equal))
        return std::nullopt;

      auto Num32BitConstants = parseUIntParam();
      if (!Num32BitConstants.has_value())
        return std::nullopt;
      Params.Num32BitConstants = Num32BitConstants;
    } else if (tryConsumeExpectedToken(TokenKind::bReg)) {
      // `b` POS_INT
      if (Params.Reg.has_value()) {
        reportDiag(diag::err_hlsl_rootsig_repeat_param) << CurToken.TokKind;
        return std::nullopt;
      }
      auto Reg = parseRegister();
      if (!Reg.has_value())
        return std::nullopt;
      Params.Reg = Reg;
    } else if (tryConsumeExpectedToken(TokenKind::kw_space)) {
      // `space` `=` POS_INT
      if (Params.Space.has_value()) {
        reportDiag(diag::err_hlsl_rootsig_repeat_param) << CurToken.TokKind;
        return std::nullopt;
      }

      if (consumeExpectedToken(TokenKind::pu_equal))
        return std::nullopt;

      auto Space = parseUIntParam();
      if (!Space.has_value())
        return std::nullopt;
      Params.Space = Space;
    } else if (tryConsumeExpectedToken(TokenKind::kw_visibility)) {
      // `visibility` `=` SHADER_VISIBILITY
      if (Params.Visibility.has_value()) {
        reportDiag(diag::err_hlsl_rootsig_repeat_param) << CurToken.TokKind;
        return std::nullopt;
      }

      if (consumeExpectedToken(TokenKind::pu_equal))
        return std::nullopt;

      auto Visibility = parseShaderVisibility(TokenKind::kw_visibility);
      if (!Visibility.has_value())
        return std::nullopt;
      Params.Visibility = Visibility;
    } else {
      consumeNextToken(); // let diagnostic be at the start of invalid token
      reportDiag(diag::err_hlsl_invalid_token)
          << /*parameter=*/0 << /*param of*/ TokenKind::kw_RootConstants;
      return std::nullopt;
    }

    // ',' denotes another element, otherwise, expected to be at ')'
    if (!tryConsumeExpectedToken(TokenKind::pu_comma))
      break;
  }

  return Params;
}

std::optional<RootSignatureParser::ParsedRootDescriptorParams>
RootSignatureParser::parseRootDescriptorParams(TokenKind DescKind,
                                               TokenKind RegType) {
  assert(CurToken.TokKind == TokenKind::pu_l_paren &&
         "Expects to only be invoked starting at given token");

  ParsedRootDescriptorParams Params;
  while (!peekExpectedToken(TokenKind::pu_r_paren)) {
    if (tryConsumeExpectedToken(RegType)) {
      // ( `b` | `t` | `u`) POS_INT
      if (Params.Reg.has_value()) {
        reportDiag(diag::err_hlsl_rootsig_repeat_param) << CurToken.TokKind;
        return std::nullopt;
      }
      auto Reg = parseRegister();
      if (!Reg.has_value())
        return std::nullopt;
      Params.Reg = Reg;
    } else if (tryConsumeExpectedToken(TokenKind::kw_space)) {
      // `space` `=` POS_INT
      if (Params.Space.has_value()) {
        reportDiag(diag::err_hlsl_rootsig_repeat_param) << CurToken.TokKind;
        return std::nullopt;
      }

      if (consumeExpectedToken(TokenKind::pu_equal))
        return std::nullopt;

      auto Space = parseUIntParam();
      if (!Space.has_value())
        return std::nullopt;
      Params.Space = Space;
    } else if (tryConsumeExpectedToken(TokenKind::kw_visibility)) {
      // `visibility` `=` SHADER_VISIBILITY
      if (Params.Visibility.has_value()) {
        reportDiag(diag::err_hlsl_rootsig_repeat_param) << CurToken.TokKind;
        return std::nullopt;
      }

      if (consumeExpectedToken(TokenKind::pu_equal))
        return std::nullopt;

      auto Visibility = parseShaderVisibility(TokenKind::kw_visibility);
      if (!Visibility.has_value())
        return std::nullopt;
      Params.Visibility = Visibility;
    } else if (tryConsumeExpectedToken(TokenKind::kw_flags)) {
      // `flags` `=` ROOT_DESCRIPTOR_FLAGS
      if (Params.Flags.has_value()) {
        reportDiag(diag::err_hlsl_rootsig_repeat_param) << CurToken.TokKind;
        return std::nullopt;
      }

      if (consumeExpectedToken(TokenKind::pu_equal))
        return std::nullopt;

      auto Flags = parseRootDescriptorFlags(TokenKind::kw_flags);
      if (!Flags.has_value())
        return std::nullopt;
      Params.Flags = Flags;
    } else {
      consumeNextToken(); // let diagnostic be at the start of invalid token
      reportDiag(diag::err_hlsl_invalid_token)
          << /*parameter=*/0 << /*param of*/ DescKind;
      return std::nullopt;
    }

    // ',' denotes another element, otherwise, expected to be at ')'
    if (!tryConsumeExpectedToken(TokenKind::pu_comma))
      break;
  }

  return Params;
}

std::optional<RootSignatureParser::ParsedClauseParams>
RootSignatureParser::parseDescriptorTableClauseParams(TokenKind ClauseKind,
                                                      TokenKind RegType) {
  assert(CurToken.TokKind == TokenKind::pu_l_paren &&
         "Expects to only be invoked starting at given token");

  ParsedClauseParams Params;
  while (!peekExpectedToken(TokenKind::pu_r_paren)) {
    if (tryConsumeExpectedToken(RegType)) {
      // ( `b` | `t` | `u` | `s`) POS_INT
      if (Params.Reg.has_value()) {
        reportDiag(diag::err_hlsl_rootsig_repeat_param) << CurToken.TokKind;
        return std::nullopt;
      }
      auto Reg = parseRegister();
      if (!Reg.has_value())
        return std::nullopt;
      Params.Reg = Reg;
    } else if (tryConsumeExpectedToken(TokenKind::kw_numDescriptors)) {
      // `numDescriptors` `=` POS_INT | unbounded
      if (Params.NumDescriptors.has_value()) {
        reportDiag(diag::err_hlsl_rootsig_repeat_param) << CurToken.TokKind;
        return std::nullopt;
      }

      if (consumeExpectedToken(TokenKind::pu_equal))
        return std::nullopt;

      std::optional<uint32_t> NumDescriptors;
      if (tryConsumeExpectedToken(TokenKind::en_unbounded))
        NumDescriptors = NumDescriptorsUnbounded;
      else {
        NumDescriptors = parseUIntParam();
        if (!NumDescriptors.has_value())
          return std::nullopt;
      }

      Params.NumDescriptors = NumDescriptors;
    } else if (tryConsumeExpectedToken(TokenKind::kw_space)) {
      // `space` `=` POS_INT
      if (Params.Space.has_value()) {
        reportDiag(diag::err_hlsl_rootsig_repeat_param) << CurToken.TokKind;
        return std::nullopt;
      }

      if (consumeExpectedToken(TokenKind::pu_equal))
        return std::nullopt;

      auto Space = parseUIntParam();
      if (!Space.has_value())
        return std::nullopt;
      Params.Space = Space;
    } else if (tryConsumeExpectedToken(TokenKind::kw_offset)) {
      // `offset` `=` POS_INT | DESCRIPTOR_RANGE_OFFSET_APPEND
      if (Params.Offset.has_value()) {
        reportDiag(diag::err_hlsl_rootsig_repeat_param) << CurToken.TokKind;
        return std::nullopt;
      }

      if (consumeExpectedToken(TokenKind::pu_equal))
        return std::nullopt;

      std::optional<uint32_t> Offset;
      if (tryConsumeExpectedToken(TokenKind::en_DescriptorRangeOffsetAppend))
        Offset = DescriptorTableOffsetAppend;
      else {
        Offset = parseUIntParam();
        if (!Offset.has_value())
          return std::nullopt;
      }

      Params.Offset = Offset;
    } else if (tryConsumeExpectedToken(TokenKind::kw_flags)) {
      // `flags` `=` DESCRIPTOR_RANGE_FLAGS
      if (Params.Flags.has_value()) {
        reportDiag(diag::err_hlsl_rootsig_repeat_param) << CurToken.TokKind;
        return std::nullopt;
      }

      if (consumeExpectedToken(TokenKind::pu_equal))
        return std::nullopt;

      auto Flags = parseDescriptorRangeFlags(TokenKind::kw_flags);
      if (!Flags.has_value())
        return std::nullopt;
      Params.Flags = Flags;
    } else {
      consumeNextToken(); // let diagnostic be at the start of invalid token
      reportDiag(diag::err_hlsl_invalid_token)
          << /*parameter=*/0 << /*param of*/ ClauseKind;
      return std::nullopt;
    }

    // ',' denotes another element, otherwise, expected to be at ')'
    if (!tryConsumeExpectedToken(TokenKind::pu_comma))
      break;
  }

  return Params;
}

std::optional<RootSignatureParser::ParsedStaticSamplerParams>
RootSignatureParser::parseStaticSamplerParams() {
  assert(CurToken.TokKind == TokenKind::pu_l_paren &&
         "Expects to only be invoked starting at given token");

  ParsedStaticSamplerParams Params;
  while (!peekExpectedToken(TokenKind::pu_r_paren)) {
    if (tryConsumeExpectedToken(TokenKind::sReg)) {
      // `s` POS_INT
      if (Params.Reg.has_value()) {
        reportDiag(diag::err_hlsl_rootsig_repeat_param) << CurToken.TokKind;
        return std::nullopt;
      }
      auto Reg = parseRegister();
      if (!Reg.has_value())
        return std::nullopt;
      Params.Reg = Reg;
    } else if (tryConsumeExpectedToken(TokenKind::kw_filter)) {
      // `filter` `=` FILTER
      if (Params.Filter.has_value()) {
        reportDiag(diag::err_hlsl_rootsig_repeat_param) << CurToken.TokKind;
        return std::nullopt;
      }

      if (consumeExpectedToken(TokenKind::pu_equal))
        return std::nullopt;

      auto Filter = parseSamplerFilter(TokenKind::kw_filter);
      if (!Filter.has_value())
        return std::nullopt;
      Params.Filter = Filter;
    } else if (tryConsumeExpectedToken(TokenKind::kw_addressU)) {
      // `addressU` `=` TEXTURE_ADDRESS
      if (Params.AddressU.has_value()) {
        reportDiag(diag::err_hlsl_rootsig_repeat_param) << CurToken.TokKind;
        return std::nullopt;
      }

      if (consumeExpectedToken(TokenKind::pu_equal))
        return std::nullopt;

      auto AddressU = parseTextureAddressMode(TokenKind::kw_addressU);
      if (!AddressU.has_value())
        return std::nullopt;
      Params.AddressU = AddressU;
    } else if (tryConsumeExpectedToken(TokenKind::kw_addressV)) {
      // `addressV` `=` TEXTURE_ADDRESS
      if (Params.AddressV.has_value()) {
        reportDiag(diag::err_hlsl_rootsig_repeat_param) << CurToken.TokKind;
        return std::nullopt;
      }

      if (consumeExpectedToken(TokenKind::pu_equal))
        return std::nullopt;

      auto AddressV = parseTextureAddressMode(TokenKind::kw_addressV);
      if (!AddressV.has_value())
        return std::nullopt;
      Params.AddressV = AddressV;
    } else if (tryConsumeExpectedToken(TokenKind::kw_addressW)) {
      // `addressW` `=` TEXTURE_ADDRESS
      if (Params.AddressW.has_value()) {
        reportDiag(diag::err_hlsl_rootsig_repeat_param) << CurToken.TokKind;
        return std::nullopt;
      }

      if (consumeExpectedToken(TokenKind::pu_equal))
        return std::nullopt;

      auto AddressW = parseTextureAddressMode(TokenKind::kw_addressW);
      if (!AddressW.has_value())
        return std::nullopt;
      Params.AddressW = AddressW;
    } else if (tryConsumeExpectedToken(TokenKind::kw_mipLODBias)) {
      // `mipLODBias` `=` NUMBER
      if (Params.MipLODBias.has_value()) {
        reportDiag(diag::err_hlsl_rootsig_repeat_param) << CurToken.TokKind;
        return std::nullopt;
      }

      if (consumeExpectedToken(TokenKind::pu_equal))
        return std::nullopt;

      auto MipLODBias = parseFloatParam();
      if (!MipLODBias.has_value())
        return std::nullopt;
      Params.MipLODBias = MipLODBias;
    } else if (tryConsumeExpectedToken(TokenKind::kw_maxAnisotropy)) {
      // `maxAnisotropy` `=` POS_INT
      if (Params.MaxAnisotropy.has_value()) {
        reportDiag(diag::err_hlsl_rootsig_repeat_param) << CurToken.TokKind;
        return std::nullopt;
      }

      if (consumeExpectedToken(TokenKind::pu_equal))
        return std::nullopt;

      auto MaxAnisotropy = parseUIntParam();
      if (!MaxAnisotropy.has_value())
        return std::nullopt;
      Params.MaxAnisotropy = MaxAnisotropy;
    } else if (tryConsumeExpectedToken(TokenKind::kw_comparisonFunc)) {
      // `comparisonFunc` `=` COMPARISON_FUNC
      if (Params.CompFunc.has_value()) {
        reportDiag(diag::err_hlsl_rootsig_repeat_param) << CurToken.TokKind;
        return std::nullopt;
      }

      if (consumeExpectedToken(TokenKind::pu_equal))
        return std::nullopt;

      auto CompFunc = parseComparisonFunc(TokenKind::kw_comparisonFunc);
      if (!CompFunc.has_value())
        return std::nullopt;
      Params.CompFunc = CompFunc;
    } else if (tryConsumeExpectedToken(TokenKind::kw_borderColor)) {
      // `borderColor` `=` STATIC_BORDER_COLOR
      if (Params.BorderColor.has_value()) {
        reportDiag(diag::err_hlsl_rootsig_repeat_param) << CurToken.TokKind;
        return std::nullopt;
      }

      if (consumeExpectedToken(TokenKind::pu_equal))
        return std::nullopt;

      auto BorderColor = parseStaticBorderColor(TokenKind::kw_borderColor);
      if (!BorderColor.has_value())
        return std::nullopt;
      Params.BorderColor = BorderColor;
    } else if (tryConsumeExpectedToken(TokenKind::kw_minLOD)) {
      // `minLOD` `=` NUMBER
      if (Params.MinLOD.has_value()) {
        reportDiag(diag::err_hlsl_rootsig_repeat_param) << CurToken.TokKind;
        return std::nullopt;
      }

      if (consumeExpectedToken(TokenKind::pu_equal))
        return std::nullopt;

      auto MinLOD = parseFloatParam();
      if (!MinLOD.has_value())
        return std::nullopt;
      Params.MinLOD = MinLOD;
    } else if (tryConsumeExpectedToken(TokenKind::kw_maxLOD)) {
      // `maxLOD` `=` NUMBER
      if (Params.MaxLOD.has_value()) {
        reportDiag(diag::err_hlsl_rootsig_repeat_param) << CurToken.TokKind;
        return std::nullopt;
      }

      if (consumeExpectedToken(TokenKind::pu_equal))
        return std::nullopt;

      auto MaxLOD = parseFloatParam();
      if (!MaxLOD.has_value())
        return std::nullopt;
      Params.MaxLOD = MaxLOD;
    } else if (tryConsumeExpectedToken(TokenKind::kw_space)) {
      // `space` `=` POS_INT
      if (Params.Space.has_value()) {
        reportDiag(diag::err_hlsl_rootsig_repeat_param) << CurToken.TokKind;
        return std::nullopt;
      }

      if (consumeExpectedToken(TokenKind::pu_equal))
        return std::nullopt;

      auto Space = parseUIntParam();
      if (!Space.has_value())
        return std::nullopt;
      Params.Space = Space;
    } else if (tryConsumeExpectedToken(TokenKind::kw_visibility)) {
      // `visibility` `=` SHADER_VISIBILITY
      if (Params.Visibility.has_value()) {
        reportDiag(diag::err_hlsl_rootsig_repeat_param) << CurToken.TokKind;
        return std::nullopt;
      }

      if (consumeExpectedToken(TokenKind::pu_equal))
        return std::nullopt;

      auto Visibility = parseShaderVisibility(TokenKind::kw_visibility);
      if (!Visibility.has_value())
        return std::nullopt;
      Params.Visibility = Visibility;
    } else {
      consumeNextToken(); // let diagnostic be at the start of invalid token
      reportDiag(diag::err_hlsl_invalid_token)
          << /*parameter=*/0 << /*param of*/ TokenKind::kw_StaticSampler;
      return std::nullopt;
    }

    // ',' denotes another element, otherwise, expected to be at ')'
    if (!tryConsumeExpectedToken(TokenKind::pu_comma))
      break;
  }

  return Params;
}

std::optional<uint32_t> RootSignatureParser::parseUIntParam() {
  assert(CurToken.TokKind == TokenKind::pu_equal &&
         "Expects to only be invoked starting at given keyword");
  tryConsumeExpectedToken(TokenKind::pu_plus);
  if (consumeExpectedToken(TokenKind::int_literal, diag::err_expected_after,
                           CurToken.TokKind))
    return std::nullopt;
  return handleUIntLiteral();
}

std::optional<Register> RootSignatureParser::parseRegister() {
  assert((CurToken.TokKind == TokenKind::bReg ||
          CurToken.TokKind == TokenKind::tReg ||
          CurToken.TokKind == TokenKind::uReg ||
          CurToken.TokKind == TokenKind::sReg) &&
         "Expects to only be invoked starting at given keyword");

  Register Reg;
  switch (CurToken.TokKind) {
  default:
    llvm_unreachable("Switch for consumed token was not provided");
  case TokenKind::bReg:
    Reg.ViewType = RegisterType::BReg;
    break;
  case TokenKind::tReg:
    Reg.ViewType = RegisterType::TReg;
    break;
  case TokenKind::uReg:
    Reg.ViewType = RegisterType::UReg;
    break;
  case TokenKind::sReg:
    Reg.ViewType = RegisterType::SReg;
    break;
  }

  auto Number = handleUIntLiteral();
  if (!Number.has_value())
    return std::nullopt; // propogate NumericLiteralParser error

  Reg.Number = *Number;
  return Reg;
}

std::optional<float> RootSignatureParser::parseFloatParam() {
  assert(CurToken.TokKind == TokenKind::pu_equal &&
         "Expects to only be invoked starting at given keyword");
  // Consume sign modifier
  bool Signed =
      tryConsumeExpectedToken({TokenKind::pu_plus, TokenKind::pu_minus});
  bool Negated = Signed && CurToken.TokKind == TokenKind::pu_minus;

  // DXC will treat a postive signed integer as unsigned
  if (!Negated && tryConsumeExpectedToken(TokenKind::int_literal)) {
    std::optional<uint32_t> UInt = handleUIntLiteral();
    if (!UInt.has_value())
      return std::nullopt;
    return float(UInt.value());
  }

  if (Negated && tryConsumeExpectedToken(TokenKind::int_literal)) {
    std::optional<int32_t> Int = handleIntLiteral(Negated);
    if (!Int.has_value())
      return std::nullopt;
    return float(Int.value());
  }

  if (tryConsumeExpectedToken(TokenKind::float_literal)) {
    std::optional<float> Float = handleFloatLiteral(Negated);
    if (!Float.has_value())
      return std::nullopt;
    return Float.value();
  }

  return std::nullopt;
}

std::optional<llvm::dxbc::ShaderVisibility>
RootSignatureParser::parseShaderVisibility(TokenKind Context) {
  assert(CurToken.TokKind == TokenKind::pu_equal &&
         "Expects to only be invoked starting at given keyword");

  TokenKind Expected[] = {
#define SHADER_VISIBILITY_ENUM(NAME, LIT) TokenKind::en_##NAME,
#include "clang/Lex/HLSLRootSignatureTokenKinds.def"
  };

  if (!tryConsumeExpectedToken(Expected)) {
    consumeNextToken(); // consume token to point at invalid token
    reportDiag(diag::err_hlsl_invalid_token)
        << /*value=*/1 << /*value of*/ Context;
    return std::nullopt;
  }

  switch (CurToken.TokKind) {
#define SHADER_VISIBILITY_ENUM(NAME, LIT)                                      \
  case TokenKind::en_##NAME:                                                   \
    return llvm::dxbc::ShaderVisibility::NAME;                                 \
    break;
#include "clang/Lex/HLSLRootSignatureTokenKinds.def"
  default:
    llvm_unreachable("Switch for consumed enum token was not provided");
  }

  return std::nullopt;
}

std::optional<llvm::dxbc::SamplerFilter>
RootSignatureParser::parseSamplerFilter(TokenKind Context) {
  assert(CurToken.TokKind == TokenKind::pu_equal &&
         "Expects to only be invoked starting at given keyword");

  TokenKind Expected[] = {
#define FILTER_ENUM(NAME, LIT) TokenKind::en_##NAME,
#include "clang/Lex/HLSLRootSignatureTokenKinds.def"
  };

  if (!tryConsumeExpectedToken(Expected)) {
    consumeNextToken(); // consume token to point at invalid token
    reportDiag(diag::err_hlsl_invalid_token)
        << /*value=*/1 << /*value of*/ Context;
    return std::nullopt;
  }

  switch (CurToken.TokKind) {
#define FILTER_ENUM(NAME, LIT)                                                 \
  case TokenKind::en_##NAME:                                                   \
    return llvm::dxbc::SamplerFilter::NAME;                                    \
    break;
#include "clang/Lex/HLSLRootSignatureTokenKinds.def"
  default:
    llvm_unreachable("Switch for consumed enum token was not provided");
  }

  return std::nullopt;
}

std::optional<llvm::dxbc::TextureAddressMode>
RootSignatureParser::parseTextureAddressMode(TokenKind Context) {
  assert(CurToken.TokKind == TokenKind::pu_equal &&
         "Expects to only be invoked starting at given keyword");

  TokenKind Expected[] = {
#define TEXTURE_ADDRESS_MODE_ENUM(NAME, LIT) TokenKind::en_##NAME,
#include "clang/Lex/HLSLRootSignatureTokenKinds.def"
  };

  if (!tryConsumeExpectedToken(Expected)) {
    consumeNextToken(); // consume token to point at invalid token
    reportDiag(diag::err_hlsl_invalid_token)
        << /*value=*/1 << /*value of*/ Context;
    return std::nullopt;
  }

  switch (CurToken.TokKind) {
#define TEXTURE_ADDRESS_MODE_ENUM(NAME, LIT)                                   \
  case TokenKind::en_##NAME:                                                   \
    return llvm::dxbc::TextureAddressMode::NAME;                               \
    break;
#include "clang/Lex/HLSLRootSignatureTokenKinds.def"
  default:
    llvm_unreachable("Switch for consumed enum token was not provided");
  }

  return std::nullopt;
}

std::optional<llvm::dxbc::ComparisonFunc>
RootSignatureParser::parseComparisonFunc(TokenKind Context) {
  assert(CurToken.TokKind == TokenKind::pu_equal &&
         "Expects to only be invoked starting at given keyword");

  TokenKind Expected[] = {
#define COMPARISON_FUNC_ENUM(NAME, LIT) TokenKind::en_##NAME,
#include "clang/Lex/HLSLRootSignatureTokenKinds.def"
  };

  if (!tryConsumeExpectedToken(Expected)) {
    consumeNextToken(); // consume token to point at invalid token
    reportDiag(diag::err_hlsl_invalid_token)
        << /*value=*/1 << /*value of*/ Context;
    return std::nullopt;
  }

  switch (CurToken.TokKind) {
#define COMPARISON_FUNC_ENUM(NAME, LIT)                                        \
  case TokenKind::en_##NAME:                                                   \
    return llvm::dxbc::ComparisonFunc::NAME;                                   \
    break;
#include "clang/Lex/HLSLRootSignatureTokenKinds.def"
  default:
    llvm_unreachable("Switch for consumed enum token was not provided");
  }

  return std::nullopt;
}

std::optional<llvm::dxbc::StaticBorderColor>
RootSignatureParser::parseStaticBorderColor(TokenKind Context) {
  assert(CurToken.TokKind == TokenKind::pu_equal &&
         "Expects to only be invoked starting at given keyword");

  TokenKind Expected[] = {
#define STATIC_BORDER_COLOR_ENUM(NAME, LIT) TokenKind::en_##NAME,
#include "clang/Lex/HLSLRootSignatureTokenKinds.def"
  };

  if (!tryConsumeExpectedToken(Expected)) {
    consumeNextToken(); // consume token to point at invalid token
    reportDiag(diag::err_hlsl_invalid_token)
        << /*value=*/1 << /*value of*/ Context;
    return std::nullopt;
  }

  switch (CurToken.TokKind) {
#define STATIC_BORDER_COLOR_ENUM(NAME, LIT)                                    \
  case TokenKind::en_##NAME:                                                   \
    return llvm::dxbc::StaticBorderColor::NAME;                                \
    break;
#include "clang/Lex/HLSLRootSignatureTokenKinds.def"
  default:
    llvm_unreachable("Switch for consumed enum token was not provided");
  }

  return std::nullopt;
}

std::optional<llvm::dxbc::RootDescriptorFlags>
RootSignatureParser::parseRootDescriptorFlags(TokenKind Context) {
  assert(CurToken.TokKind == TokenKind::pu_equal &&
         "Expects to only be invoked starting at given keyword");

  // Handle the edge-case of '0' to specify no flags set
  if (tryConsumeExpectedToken(TokenKind::int_literal)) {
    if (!verifyZeroFlag()) {
      reportDiag(diag::err_hlsl_rootsig_non_zero_flag);
      return std::nullopt;
    }
    return llvm::dxbc::RootDescriptorFlags::None;
  }

  TokenKind Expected[] = {
#define ROOT_DESCRIPTOR_FLAG_ENUM(NAME, LIT) TokenKind::en_##NAME,
#include "clang/Lex/HLSLRootSignatureTokenKinds.def"
  };

  std::optional<llvm::dxbc::RootDescriptorFlags> Flags;

  do {
    if (tryConsumeExpectedToken(Expected)) {
      switch (CurToken.TokKind) {
#define ROOT_DESCRIPTOR_FLAG_ENUM(NAME, LIT)                                   \
  case TokenKind::en_##NAME:                                                   \
    Flags = maybeOrFlag<llvm::dxbc::RootDescriptorFlags>(                      \
        Flags, llvm::dxbc::RootDescriptorFlags::NAME);                         \
    break;
#include "clang/Lex/HLSLRootSignatureTokenKinds.def"
      default:
        llvm_unreachable("Switch for consumed enum token was not provided");
      }
    } else {
      consumeNextToken(); // consume token to point at invalid token
      reportDiag(diag::err_hlsl_invalid_token)
          << /*value=*/1 << /*value of*/ Context;
      return std::nullopt;
    }
  } while (tryConsumeExpectedToken(TokenKind::pu_or));

  return Flags;
}

std::optional<llvm::dxbc::DescriptorRangeFlags>
RootSignatureParser::parseDescriptorRangeFlags(TokenKind Context) {
  assert(CurToken.TokKind == TokenKind::pu_equal &&
         "Expects to only be invoked starting at given keyword");

  // Handle the edge-case of '0' to specify no flags set
  if (tryConsumeExpectedToken(TokenKind::int_literal)) {
    if (!verifyZeroFlag()) {
      reportDiag(diag::err_hlsl_rootsig_non_zero_flag);
      return std::nullopt;
    }
    return llvm::dxbc::DescriptorRangeFlags::None;
  }

  TokenKind Expected[] = {
#define DESCRIPTOR_RANGE_FLAG_ENUM(NAME, LIT, ON) TokenKind::en_##NAME,
#include "clang/Lex/HLSLRootSignatureTokenKinds.def"
  };

  std::optional<llvm::dxbc::DescriptorRangeFlags> Flags;

  do {
    if (tryConsumeExpectedToken(Expected)) {
      switch (CurToken.TokKind) {
#define DESCRIPTOR_RANGE_FLAG_ENUM(NAME, LIT, ON)                              \
  case TokenKind::en_##NAME:                                                   \
    Flags = maybeOrFlag<llvm::dxbc::DescriptorRangeFlags>(                     \
        Flags, llvm::dxbc::DescriptorRangeFlags::NAME);                        \
    break;
#include "clang/Lex/HLSLRootSignatureTokenKinds.def"
      default:
        llvm_unreachable("Switch for consumed enum token was not provided");
      }
    } else {
      consumeNextToken(); // consume token to point at invalid token
      reportDiag(diag::err_hlsl_invalid_token)
          << /*value=*/1 << /*value of*/ Context;
      return std::nullopt;
    }
  } while (tryConsumeExpectedToken(TokenKind::pu_or));

  return Flags;
}

std::optional<uint32_t> RootSignatureParser::handleUIntLiteral() {
  // Parse the numeric value and do semantic checks on its specification
  clang::NumericLiteralParser Literal(
      CurToken.NumSpelling, getTokenLocation(CurToken), PP.getSourceManager(),
      PP.getLangOpts(), PP.getTargetInfo(), PP.getDiagnostics());
  if (Literal.hadError)
    return std::nullopt; // Error has already been reported so just return

  assert(Literal.isIntegerLiteral() &&
         "NumSpelling can only consist of digits");

  llvm::APSInt Val(32, /*IsUnsigned=*/true);
  if (Literal.GetIntegerValue(Val)) {
    // Report that the value has overflowed
    reportDiag(diag::err_hlsl_number_literal_overflow)
        << /*integer type*/ 0 << /*is signed*/ 0;
    return std::nullopt;
  }

  return Val.getExtValue();
}

std::optional<int32_t> RootSignatureParser::handleIntLiteral(bool Negated) {
  // Parse the numeric value and do semantic checks on its specification
  clang::NumericLiteralParser Literal(
      CurToken.NumSpelling, getTokenLocation(CurToken), PP.getSourceManager(),
      PP.getLangOpts(), PP.getTargetInfo(), PP.getDiagnostics());
  if (Literal.hadError)
    return std::nullopt; // Error has already been reported so just return

  assert(Literal.isIntegerLiteral() &&
         "NumSpelling can only consist of digits");

  llvm::APSInt Val(32, /*IsUnsigned=*/true);
  // GetIntegerValue will overwrite Val from the parsed Literal and return
  // true if it overflows as a 32-bit unsigned int
  bool Overflowed = Literal.GetIntegerValue(Val);

  // So we then need to check that it doesn't overflow as a 32-bit signed int:
  int64_t MaxNegativeMagnitude = -int64_t(std::numeric_limits<int32_t>::min());
  Overflowed |= (Negated && MaxNegativeMagnitude < Val.getExtValue());

  int64_t MaxPositiveMagnitude = int64_t(std::numeric_limits<int32_t>::max());
  Overflowed |= (!Negated && MaxPositiveMagnitude < Val.getExtValue());

  if (Overflowed) {
    // Report that the value has overflowed
    reportDiag(diag::err_hlsl_number_literal_overflow)
        << /*integer type*/ 0 << /*is signed*/ 1;
    return std::nullopt;
  }

  if (Negated)
    Val = -Val;

  return int32_t(Val.getExtValue());
}

std::optional<float> RootSignatureParser::handleFloatLiteral(bool Negated) {
  // Parse the numeric value and do semantic checks on its specification
  clang::NumericLiteralParser Literal(
      CurToken.NumSpelling, getTokenLocation(CurToken), PP.getSourceManager(),
      PP.getLangOpts(), PP.getTargetInfo(), PP.getDiagnostics());
  if (Literal.hadError)
    return std::nullopt; // Error has already been reported so just return

  assert(Literal.isFloatingLiteral() &&
         "NumSpelling consists only of [0-9.ef+-]. Any malformed NumSpelling "
         "will be caught and reported by NumericLiteralParser.");

  // DXC used `strtod` to convert the token string to a float which corresponds
  // to:
  auto DXCSemantics = llvm::APFloat::Semantics::S_IEEEdouble;
  auto DXCRoundingMode = llvm::RoundingMode::NearestTiesToEven;

  llvm::APFloat Val(llvm::APFloat::EnumToSemantics(DXCSemantics));
  llvm::APFloat::opStatus Status(Literal.GetFloatValue(Val, DXCRoundingMode));

  // Note: we do not error when opStatus::opInexact by itself as this just
  // denotes that rounding occured but not that it is invalid
  assert(!(Status & llvm::APFloat::opStatus::opInvalidOp) &&
         "NumSpelling consists only of [0-9.ef+-]. Any malformed NumSpelling "
         "will be caught and reported by NumericLiteralParser.");

  assert(!(Status & llvm::APFloat::opStatus::opDivByZero) &&
         "It is not possible for a division to be performed when "
         "constructing an APFloat from a string");

  if (Status & llvm::APFloat::opStatus::opUnderflow) {
    // Report that the value has underflowed
    reportDiag(diag::err_hlsl_number_literal_underflow);
    return std::nullopt;
  }

  if (Status & llvm::APFloat::opStatus::opOverflow) {
    // Report that the value has overflowed
    reportDiag(diag::err_hlsl_number_literal_overflow) << /*float type*/ 1;
    return std::nullopt;
  }

  if (Negated)
    Val = -Val;

  double DoubleVal = Val.convertToDouble();
  double FloatMax = double(std::numeric_limits<float>::max());
  if (FloatMax < DoubleVal || DoubleVal < -FloatMax) {
    // Report that the value has overflowed
    reportDiag(diag::err_hlsl_number_literal_overflow) << /*float type*/ 1;
    return std::nullopt;
  }

  return static_cast<float>(DoubleVal);
}

bool RootSignatureParser::verifyZeroFlag() {
  assert(CurToken.TokKind == TokenKind::int_literal);
  auto X = handleUIntLiteral();
  return X.has_value() && X.value() == 0;
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
  DiagnosticBuilder DB = reportDiag(DiagID);
  switch (DiagID) {
  case diag::err_expected:
    DB << Expected;
    break;
  case diag::err_expected_either:
    DB << Expected << Context;
    break;
  case diag::err_expected_after:
    DB << Context << Expected;
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

bool RootSignatureParser::skipUntilExpectedToken(TokenKind Expected) {
  return skipUntilExpectedToken(ArrayRef{Expected});
}

bool RootSignatureParser::skipUntilExpectedToken(
    ArrayRef<TokenKind> AnyExpected) {

  while (!peekExpectedToken(AnyExpected)) {
    if (peekExpectedToken(TokenKind::end_of_stream))
      return false;
    consumeNextToken();
  }

  return true;
}

bool RootSignatureParser::skipUntilClosedParens(uint32_t NumParens) {
  TokenKind ParenKinds[] = {
      TokenKind::pu_l_paren,
      TokenKind::pu_r_paren,
  };
  while (skipUntilExpectedToken(ParenKinds)) {
    consumeNextToken();
    if (CurToken.TokKind == TokenKind::pu_r_paren)
      NumParens--;
    else
      NumParens++;
    if (NumParens == 0)
      return true;
  }

  return false;
}

SourceLocation RootSignatureParser::getTokenLocation(RootSignatureToken Tok) {
  return Signature->getLocationOfByte(Tok.LocOffset, PP.getSourceManager(),
                                      PP.getLangOpts(), PP.getTargetInfo());
}

IdentifierInfo *ParseHLSLRootSignature(Sema &Actions,
                                       llvm::dxbc::RootSignatureVersion Version,
                                       StringLiteral *Signature) {
  // Construct our identifier
  auto [DeclIdent, Found] =
      Actions.HLSL().ActOnStartRootSignatureDecl(Signature->getString());
  // If we haven't found an already defined DeclIdent then parse the root
  // signature string and construct the in-memory elements
  if (!Found) {
    // Invoke the root signature parser to construct the in-memory constructs
    hlsl::RootSignatureParser Parser(Version, Signature,
                                     Actions.getPreprocessor());
    if (Parser.parse())
      return nullptr;

    // Construct the declaration.
    Actions.HLSL().ActOnFinishRootSignatureDecl(
        Signature->getBeginLoc(), DeclIdent, Parser.getElements());
  }

  return DeclIdent;
}

void HandleRootSignatureTarget(Sema &S, StringRef EntryRootSig) {
  ASTConsumer *Consumer = &S.getASTConsumer();

  // Minimally initalize the parser. This does a couple things:
  // - initializes Sema scope handling
  // - invokes HLSLExternalSemaSource
  // - invokes the preprocessor to lex the macros in the file
  std::unique_ptr<Parser> P(new Parser(S.getPreprocessor(), S, true));
  S.getPreprocessor().EnterMainSourceFile();

  bool HaveLexer = S.getPreprocessor().getCurrentLexer();
  if (HaveLexer) {
    P->Initialize();
    S.ActOnStartOfTranslationUnit();

    // Skim through the file to parse to find the define
    while (P->getCurToken().getKind() != tok::eof)
      P->ConsumeAnyToken();

    HLSLRootSignatureDecl *SignatureDecl =
        S.HLSL().lookupRootSignatureOverrideDecl(
            S.getASTContext().getTranslationUnitDecl());

    if (SignatureDecl)
      Consumer->HandleTopLevelDecl(DeclGroupRef(SignatureDecl));
    else
      S.getDiagnostics().Report(diag::err_hlsl_rootsignature_entry)
          << EntryRootSig;
  }

  Consumer->HandleTranslationUnit(S.getASTContext());
}

} // namespace hlsl
} // namespace clang
