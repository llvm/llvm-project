//=== ParseHLSLRootSignature.cpp - Parse Root Signature -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Parse/ParseHLSLRootSignature.h"

#include "clang/Lex/LiteralSupport.h"

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
  do {
    if (tryConsumeExpectedToken(TokenKind::kw_RootFlags)) {
      auto Flags = parseRootFlags();
      if (!Flags.has_value())
        return true;
      Elements.push_back(*Flags);
    }

    if (tryConsumeExpectedToken(TokenKind::kw_RootConstants)) {
      auto Constants = parseRootConstants();
      if (!Constants.has_value())
        return true;
      Elements.push_back(*Constants);
    }

    if (tryConsumeExpectedToken(TokenKind::kw_DescriptorTable)) {
      auto Table = parseDescriptorTable();
      if (!Table.has_value())
        return true;
      Elements.push_back(*Table);
    }

    if (tryConsumeExpectedToken(
            {TokenKind::kw_CBV, TokenKind::kw_SRV, TokenKind::kw_UAV})) {
      auto Descriptor = parseRootDescriptor();
      if (!Descriptor.has_value())
        return true;
      Elements.push_back(*Descriptor);
    }

    if (tryConsumeExpectedToken(TokenKind::kw_StaticSampler)) {
      auto Sampler = parseStaticSampler();
      if (!Sampler.has_value())
        return true;
      Elements.push_back(*Sampler);
    }
  } while (tryConsumeExpectedToken(TokenKind::pu_comma));

  return consumeExpectedToken(TokenKind::end_of_stream,
                              diag::err_hlsl_unexpected_end_of_params,
                              /*param of=*/TokenKind::kw_RootSignature);
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

  // Handle the edge-case of '0' to specify no flags set
  if (tryConsumeExpectedToken(TokenKind::int_literal)) {
    if (!verifyZeroFlag()) {
      getDiags().Report(CurToken.TokLoc, diag::err_hlsl_rootsig_non_zero_flag);
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
      }
    } while (tryConsumeExpectedToken(TokenKind::pu_or));
  }

  if (consumeExpectedToken(TokenKind::pu_r_paren,
                           diag::err_hlsl_unexpected_end_of_params,
                           /*param of=*/TokenKind::kw_RootFlags))
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

  // Check mandatory parameters where provided
  if (!Params->Num32BitConstants.has_value()) {
    getDiags().Report(CurToken.TokLoc, diag::err_hlsl_rootsig_missing_param)
        << TokenKind::kw_num32BitConstants;
    return std::nullopt;
  }

  Constants.Num32BitConstants = Params->Num32BitConstants.value();

  if (!Params->Reg.has_value()) {
    getDiags().Report(CurToken.TokLoc, diag::err_hlsl_rootsig_missing_param)
        << TokenKind::bReg;
    return std::nullopt;
  }

  Constants.Reg = Params->Reg.value();

  // Fill in optional parameters
  if (Params->Visibility.has_value())
    Constants.Visibility = Params->Visibility.value();

  if (Params->Space.has_value())
    Constants.Space = Params->Space.value();

  if (consumeExpectedToken(TokenKind::pu_r_paren,
                           diag::err_hlsl_unexpected_end_of_params,
                           /*param of=*/TokenKind::kw_RootConstants))
    return std::nullopt;

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
    Descriptor.Type = DescriptorType::CBuffer;
    ExpectedReg = TokenKind::bReg;
    break;
  case TokenKind::kw_SRV:
    Descriptor.Type = DescriptorType::SRV;
    ExpectedReg = TokenKind::tReg;
    break;
  case TokenKind::kw_UAV:
    Descriptor.Type = DescriptorType::UAV;
    ExpectedReg = TokenKind::uReg;
    break;
  }
  Descriptor.setDefaultFlags();

  auto Params = parseRootDescriptorParams(ExpectedReg);
  if (!Params.has_value())
    return std::nullopt;

  // Check mandatory parameters were provided
  if (!Params->Reg.has_value()) {
    getDiags().Report(CurToken.TokLoc, diag::err_hlsl_rootsig_missing_param)
        << ExpectedReg;
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

  if (consumeExpectedToken(TokenKind::pu_r_paren,
                           diag::err_hlsl_unexpected_end_of_params,
                           /*param of=*/TokenKind::kw_RootConstants))
    return std::nullopt;

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

  // Iterate as many Clauses as possible
  do {
    if (tryConsumeExpectedToken({TokenKind::kw_CBV, TokenKind::kw_SRV,
                                 TokenKind::kw_UAV, TokenKind::kw_Sampler})) {
      auto Clause = parseDescriptorTableClause();
      if (!Clause.has_value())
        return std::nullopt;
      Elements.push_back(*Clause);
      Table.NumClauses++;
    }

    if (tryConsumeExpectedToken(TokenKind::kw_visibility)) {
      if (Visibility.has_value()) {
        getDiags().Report(CurToken.TokLoc, diag::err_hlsl_rootsig_repeat_param)
            << CurToken.TokKind;
        return std::nullopt;
      }

      if (consumeExpectedToken(TokenKind::pu_equal))
        return std::nullopt;

      Visibility = parseShaderVisibility();
      if (!Visibility.has_value())
        return std::nullopt;
    }
  } while (tryConsumeExpectedToken(TokenKind::pu_comma));

  // Fill in optional visibility
  if (Visibility.has_value())
    Table.Visibility = Visibility.value();

  if (consumeExpectedToken(TokenKind::pu_r_paren,
                           diag::err_hlsl_unexpected_end_of_params,
                           /*param of=*/TokenKind::kw_DescriptorTable))
    return std::nullopt;

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
    Clause.Type = ClauseType::CBuffer;
    ExpectedReg = TokenKind::bReg;
    break;
  case TokenKind::kw_SRV:
    Clause.Type = ClauseType::SRV;
    ExpectedReg = TokenKind::tReg;
    break;
  case TokenKind::kw_UAV:
    Clause.Type = ClauseType::UAV;
    ExpectedReg = TokenKind::uReg;
    break;
  case TokenKind::kw_Sampler:
    Clause.Type = ClauseType::Sampler;
    ExpectedReg = TokenKind::sReg;
    break;
  }
  Clause.setDefaultFlags();

  auto Params = parseDescriptorTableClauseParams(ExpectedReg);
  if (!Params.has_value())
    return std::nullopt;

  // Check mandatory parameters were provided
  if (!Params->Reg.has_value()) {
    getDiags().Report(CurToken.TokLoc, diag::err_hlsl_rootsig_missing_param)
        << ExpectedReg;
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

  if (consumeExpectedToken(TokenKind::pu_r_paren,
                           diag::err_hlsl_unexpected_end_of_params,
                           /*param of=*/ParamKind))
    return std::nullopt;

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

  // Check mandatory parameters were provided
  if (!Params->Reg.has_value()) {
    getDiags().Report(CurToken.TokLoc, diag::err_hlsl_rootsig_missing_param)
        << TokenKind::sReg;
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

  if (consumeExpectedToken(TokenKind::pu_r_paren,
                           diag::err_hlsl_unexpected_end_of_params,
                           /*param of=*/TokenKind::kw_StaticSampler))
    return std::nullopt;

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
  do {
    // `num32BitConstants` `=` POS_INT
    if (tryConsumeExpectedToken(TokenKind::kw_num32BitConstants)) {
      if (Params.Num32BitConstants.has_value()) {
        getDiags().Report(CurToken.TokLoc, diag::err_hlsl_rootsig_repeat_param)
            << CurToken.TokKind;
        return std::nullopt;
      }

      if (consumeExpectedToken(TokenKind::pu_equal))
        return std::nullopt;

      auto Num32BitConstants = parseUIntParam();
      if (!Num32BitConstants.has_value())
        return std::nullopt;
      Params.Num32BitConstants = Num32BitConstants;
    }

    // `b` POS_INT
    if (tryConsumeExpectedToken(TokenKind::bReg)) {
      if (Params.Reg.has_value()) {
        getDiags().Report(CurToken.TokLoc, diag::err_hlsl_rootsig_repeat_param)
            << CurToken.TokKind;
        return std::nullopt;
      }
      auto Reg = parseRegister();
      if (!Reg.has_value())
        return std::nullopt;
      Params.Reg = Reg;
    }

    // `space` `=` POS_INT
    if (tryConsumeExpectedToken(TokenKind::kw_space)) {
      if (Params.Space.has_value()) {
        getDiags().Report(CurToken.TokLoc, diag::err_hlsl_rootsig_repeat_param)
            << CurToken.TokKind;
        return std::nullopt;
      }

      if (consumeExpectedToken(TokenKind::pu_equal))
        return std::nullopt;

      auto Space = parseUIntParam();
      if (!Space.has_value())
        return std::nullopt;
      Params.Space = Space;
    }

    // `visibility` `=` SHADER_VISIBILITY
    if (tryConsumeExpectedToken(TokenKind::kw_visibility)) {
      if (Params.Visibility.has_value()) {
        getDiags().Report(CurToken.TokLoc, diag::err_hlsl_rootsig_repeat_param)
            << CurToken.TokKind;
        return std::nullopt;
      }

      if (consumeExpectedToken(TokenKind::pu_equal))
        return std::nullopt;

      auto Visibility = parseShaderVisibility();
      if (!Visibility.has_value())
        return std::nullopt;
      Params.Visibility = Visibility;
    }
  } while (tryConsumeExpectedToken(TokenKind::pu_comma));

  return Params;
}

std::optional<RootSignatureParser::ParsedRootDescriptorParams>
RootSignatureParser::parseRootDescriptorParams(TokenKind RegType) {
  assert(CurToken.TokKind == TokenKind::pu_l_paren &&
         "Expects to only be invoked starting at given token");

  ParsedRootDescriptorParams Params;
  do {
    // ( `b` | `t` | `u`) POS_INT
    if (tryConsumeExpectedToken(RegType)) {
      if (Params.Reg.has_value()) {
        getDiags().Report(CurToken.TokLoc, diag::err_hlsl_rootsig_repeat_param)
            << CurToken.TokKind;
        return std::nullopt;
      }
      auto Reg = parseRegister();
      if (!Reg.has_value())
        return std::nullopt;
      Params.Reg = Reg;
    }

    // `space` `=` POS_INT
    if (tryConsumeExpectedToken(TokenKind::kw_space)) {
      if (Params.Space.has_value()) {
        getDiags().Report(CurToken.TokLoc, diag::err_hlsl_rootsig_repeat_param)
            << CurToken.TokKind;
        return std::nullopt;
      }

      if (consumeExpectedToken(TokenKind::pu_equal))
        return std::nullopt;

      auto Space = parseUIntParam();
      if (!Space.has_value())
        return std::nullopt;
      Params.Space = Space;
    }

    // `visibility` `=` SHADER_VISIBILITY
    if (tryConsumeExpectedToken(TokenKind::kw_visibility)) {
      if (Params.Visibility.has_value()) {
        getDiags().Report(CurToken.TokLoc, diag::err_hlsl_rootsig_repeat_param)
            << CurToken.TokKind;
        return std::nullopt;
      }

      if (consumeExpectedToken(TokenKind::pu_equal))
        return std::nullopt;

      auto Visibility = parseShaderVisibility();
      if (!Visibility.has_value())
        return std::nullopt;
      Params.Visibility = Visibility;
    }

    // `flags` `=` ROOT_DESCRIPTOR_FLAGS
    if (tryConsumeExpectedToken(TokenKind::kw_flags)) {
      if (Params.Flags.has_value()) {
        getDiags().Report(CurToken.TokLoc, diag::err_hlsl_rootsig_repeat_param)
            << CurToken.TokKind;
        return std::nullopt;
      }

      if (consumeExpectedToken(TokenKind::pu_equal))
        return std::nullopt;

      auto Flags = parseRootDescriptorFlags();
      if (!Flags.has_value())
        return std::nullopt;
      Params.Flags = Flags;
    }
  } while (tryConsumeExpectedToken(TokenKind::pu_comma));

  return Params;
}

std::optional<RootSignatureParser::ParsedClauseParams>
RootSignatureParser::parseDescriptorTableClauseParams(TokenKind RegType) {
  assert(CurToken.TokKind == TokenKind::pu_l_paren &&
         "Expects to only be invoked starting at given token");

  ParsedClauseParams Params;
  do {
    // ( `b` | `t` | `u` | `s`) POS_INT
    if (tryConsumeExpectedToken(RegType)) {
      if (Params.Reg.has_value()) {
        getDiags().Report(CurToken.TokLoc, diag::err_hlsl_rootsig_repeat_param)
            << CurToken.TokKind;
        return std::nullopt;
      }
      auto Reg = parseRegister();
      if (!Reg.has_value())
        return std::nullopt;
      Params.Reg = Reg;
    }

    // `numDescriptors` `=` POS_INT | unbounded
    if (tryConsumeExpectedToken(TokenKind::kw_numDescriptors)) {
      if (Params.NumDescriptors.has_value()) {
        getDiags().Report(CurToken.TokLoc, diag::err_hlsl_rootsig_repeat_param)
            << CurToken.TokKind;
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
    }

    // `space` `=` POS_INT
    if (tryConsumeExpectedToken(TokenKind::kw_space)) {
      if (Params.Space.has_value()) {
        getDiags().Report(CurToken.TokLoc, diag::err_hlsl_rootsig_repeat_param)
            << CurToken.TokKind;
        return std::nullopt;
      }

      if (consumeExpectedToken(TokenKind::pu_equal))
        return std::nullopt;

      auto Space = parseUIntParam();
      if (!Space.has_value())
        return std::nullopt;
      Params.Space = Space;
    }

    // `offset` `=` POS_INT | DESCRIPTOR_RANGE_OFFSET_APPEND
    if (tryConsumeExpectedToken(TokenKind::kw_offset)) {
      if (Params.Offset.has_value()) {
        getDiags().Report(CurToken.TokLoc, diag::err_hlsl_rootsig_repeat_param)
            << CurToken.TokKind;
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
    }

    // `flags` `=` DESCRIPTOR_RANGE_FLAGS
    if (tryConsumeExpectedToken(TokenKind::kw_flags)) {
      if (Params.Flags.has_value()) {
        getDiags().Report(CurToken.TokLoc, diag::err_hlsl_rootsig_repeat_param)
            << CurToken.TokKind;
        return std::nullopt;
      }

      if (consumeExpectedToken(TokenKind::pu_equal))
        return std::nullopt;

      auto Flags = parseDescriptorRangeFlags();
      if (!Flags.has_value())
        return std::nullopt;
      Params.Flags = Flags;
    }

  } while (tryConsumeExpectedToken(TokenKind::pu_comma));

  return Params;
}

std::optional<RootSignatureParser::ParsedStaticSamplerParams>
RootSignatureParser::parseStaticSamplerParams() {
  assert(CurToken.TokKind == TokenKind::pu_l_paren &&
         "Expects to only be invoked starting at given token");

  ParsedStaticSamplerParams Params;
  do {
    // `s` POS_INT
    if (tryConsumeExpectedToken(TokenKind::sReg)) {
      if (Params.Reg.has_value()) {
        getDiags().Report(CurToken.TokLoc, diag::err_hlsl_rootsig_repeat_param)
            << CurToken.TokKind;
        return std::nullopt;
      }
      auto Reg = parseRegister();
      if (!Reg.has_value())
        return std::nullopt;
      Params.Reg = Reg;
    }

    // `filter` `=` FILTER
    if (tryConsumeExpectedToken(TokenKind::kw_filter)) {
      if (Params.Filter.has_value()) {
        getDiags().Report(CurToken.TokLoc, diag::err_hlsl_rootsig_repeat_param)
            << CurToken.TokKind;
        return std::nullopt;
      }

      if (consumeExpectedToken(TokenKind::pu_equal))
        return std::nullopt;

      auto Filter = parseSamplerFilter();
      if (!Filter.has_value())
        return std::nullopt;
      Params.Filter = Filter;
    }

    // `addressU` `=` TEXTURE_ADDRESS
    if (tryConsumeExpectedToken(TokenKind::kw_addressU)) {
      if (Params.AddressU.has_value()) {
        getDiags().Report(CurToken.TokLoc, diag::err_hlsl_rootsig_repeat_param)
            << CurToken.TokKind;
        return std::nullopt;
      }

      if (consumeExpectedToken(TokenKind::pu_equal))
        return std::nullopt;

      auto AddressU = parseTextureAddressMode();
      if (!AddressU.has_value())
        return std::nullopt;
      Params.AddressU = AddressU;
    }

    // `addressV` `=` TEXTURE_ADDRESS
    if (tryConsumeExpectedToken(TokenKind::kw_addressV)) {
      if (Params.AddressV.has_value()) {
        getDiags().Report(CurToken.TokLoc, diag::err_hlsl_rootsig_repeat_param)
            << CurToken.TokKind;
        return std::nullopt;
      }

      if (consumeExpectedToken(TokenKind::pu_equal))
        return std::nullopt;

      auto AddressV = parseTextureAddressMode();
      if (!AddressV.has_value())
        return std::nullopt;
      Params.AddressV = AddressV;
    }

    // `addressW` `=` TEXTURE_ADDRESS
    if (tryConsumeExpectedToken(TokenKind::kw_addressW)) {
      if (Params.AddressW.has_value()) {
        getDiags().Report(CurToken.TokLoc, diag::err_hlsl_rootsig_repeat_param)
            << CurToken.TokKind;
        return std::nullopt;
      }

      if (consumeExpectedToken(TokenKind::pu_equal))
        return std::nullopt;

      auto AddressW = parseTextureAddressMode();
      if (!AddressW.has_value())
        return std::nullopt;
      Params.AddressW = AddressW;
    }

    // `mipLODBias` `=` NUMBER
    if (tryConsumeExpectedToken(TokenKind::kw_mipLODBias)) {
      if (Params.MipLODBias.has_value()) {
        getDiags().Report(CurToken.TokLoc, diag::err_hlsl_rootsig_repeat_param)
            << CurToken.TokKind;
        return std::nullopt;
      }

      if (consumeExpectedToken(TokenKind::pu_equal))
        return std::nullopt;

      auto MipLODBias = parseFloatParam();
      if (!MipLODBias.has_value())
        return std::nullopt;
      Params.MipLODBias = MipLODBias;
    }

    // `maxAnisotropy` `=` POS_INT
    if (tryConsumeExpectedToken(TokenKind::kw_maxAnisotropy)) {
      if (Params.MaxAnisotropy.has_value()) {
        getDiags().Report(CurToken.TokLoc, diag::err_hlsl_rootsig_repeat_param)
            << CurToken.TokKind;
        return std::nullopt;
      }

      if (consumeExpectedToken(TokenKind::pu_equal))
        return std::nullopt;

      auto MaxAnisotropy = parseUIntParam();
      if (!MaxAnisotropy.has_value())
        return std::nullopt;
      Params.MaxAnisotropy = MaxAnisotropy;
    }

    // `comparisonFunc` `=` COMPARISON_FUNC
    if (tryConsumeExpectedToken(TokenKind::kw_comparisonFunc)) {
      if (Params.CompFunc.has_value()) {
        getDiags().Report(CurToken.TokLoc, diag::err_hlsl_rootsig_repeat_param)
            << CurToken.TokKind;
        return std::nullopt;
      }

      if (consumeExpectedToken(TokenKind::pu_equal))
        return std::nullopt;

      auto CompFunc = parseComparisonFunc();
      if (!CompFunc.has_value())
        return std::nullopt;
      Params.CompFunc = CompFunc;
    }

    // `borderColor` `=` STATIC_BORDER_COLOR
    if (tryConsumeExpectedToken(TokenKind::kw_borderColor)) {
      if (Params.BorderColor.has_value()) {
        getDiags().Report(CurToken.TokLoc, diag::err_hlsl_rootsig_repeat_param)
            << CurToken.TokKind;
        return std::nullopt;
      }

      if (consumeExpectedToken(TokenKind::pu_equal))
        return std::nullopt;

      auto BorderColor = parseStaticBorderColor();
      if (!BorderColor.has_value())
        return std::nullopt;
      Params.BorderColor = BorderColor;
    }

    // `minLOD` `=` NUMBER
    if (tryConsumeExpectedToken(TokenKind::kw_minLOD)) {
      if (Params.MinLOD.has_value()) {
        getDiags().Report(CurToken.TokLoc, diag::err_hlsl_rootsig_repeat_param)
            << CurToken.TokKind;
        return std::nullopt;
      }

      if (consumeExpectedToken(TokenKind::pu_equal))
        return std::nullopt;

      auto MinLOD = parseFloatParam();
      if (!MinLOD.has_value())
        return std::nullopt;
      Params.MinLOD = MinLOD;
    }

    // `maxLOD` `=` NUMBER
    if (tryConsumeExpectedToken(TokenKind::kw_maxLOD)) {
      if (Params.MaxLOD.has_value()) {
        getDiags().Report(CurToken.TokLoc, diag::err_hlsl_rootsig_repeat_param)
            << CurToken.TokKind;
        return std::nullopt;
      }

      if (consumeExpectedToken(TokenKind::pu_equal))
        return std::nullopt;

      auto MaxLOD = parseFloatParam();
      if (!MaxLOD.has_value())
        return std::nullopt;
      Params.MaxLOD = MaxLOD;
    }

    // `space` `=` POS_INT
    if (tryConsumeExpectedToken(TokenKind::kw_space)) {
      if (Params.Space.has_value()) {
        getDiags().Report(CurToken.TokLoc, diag::err_hlsl_rootsig_repeat_param)
            << CurToken.TokKind;
        return std::nullopt;
      }

      if (consumeExpectedToken(TokenKind::pu_equal))
        return std::nullopt;

      auto Space = parseUIntParam();
      if (!Space.has_value())
        return std::nullopt;
      Params.Space = Space;
    }

    // `visibility` `=` SHADER_VISIBILITY
    if (tryConsumeExpectedToken(TokenKind::kw_visibility)) {
      if (Params.Visibility.has_value()) {
        getDiags().Report(CurToken.TokLoc, diag::err_hlsl_rootsig_repeat_param)
            << CurToken.TokKind;
        return std::nullopt;
      }

      if (consumeExpectedToken(TokenKind::pu_equal))
        return std::nullopt;

      auto Visibility = parseShaderVisibility();
      if (!Visibility.has_value())
        return std::nullopt;
      Params.Visibility = Visibility;
    }
  } while (tryConsumeExpectedToken(TokenKind::pu_comma));

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
RootSignatureParser::parseShaderVisibility() {
  assert(CurToken.TokKind == TokenKind::pu_equal &&
         "Expects to only be invoked starting at given keyword");

  TokenKind Expected[] = {
#define SHADER_VISIBILITY_ENUM(NAME, LIT) TokenKind::en_##NAME,
#include "clang/Lex/HLSLRootSignatureTokenKinds.def"
  };

  if (!tryConsumeExpectedToken(Expected))
    return std::nullopt;

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
RootSignatureParser::parseSamplerFilter() {
  assert(CurToken.TokKind == TokenKind::pu_equal &&
         "Expects to only be invoked starting at given keyword");

  TokenKind Expected[] = {
#define FILTER_ENUM(NAME, LIT) TokenKind::en_##NAME,
#include "clang/Lex/HLSLRootSignatureTokenKinds.def"
  };

  if (!tryConsumeExpectedToken(Expected))
    return std::nullopt;

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
RootSignatureParser::parseTextureAddressMode() {
  assert(CurToken.TokKind == TokenKind::pu_equal &&
         "Expects to only be invoked starting at given keyword");

  TokenKind Expected[] = {
#define TEXTURE_ADDRESS_MODE_ENUM(NAME, LIT) TokenKind::en_##NAME,
#include "clang/Lex/HLSLRootSignatureTokenKinds.def"
  };

  if (!tryConsumeExpectedToken(Expected))
    return std::nullopt;

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
RootSignatureParser::parseComparisonFunc() {
  assert(CurToken.TokKind == TokenKind::pu_equal &&
         "Expects to only be invoked starting at given keyword");

  TokenKind Expected[] = {
#define COMPARISON_FUNC_ENUM(NAME, LIT) TokenKind::en_##NAME,
#include "clang/Lex/HLSLRootSignatureTokenKinds.def"
  };

  if (!tryConsumeExpectedToken(Expected))
    return std::nullopt;

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
RootSignatureParser::parseStaticBorderColor() {
  assert(CurToken.TokKind == TokenKind::pu_equal &&
         "Expects to only be invoked starting at given keyword");

  TokenKind Expected[] = {
#define STATIC_BORDER_COLOR_ENUM(NAME, LIT) TokenKind::en_##NAME,
#include "clang/Lex/HLSLRootSignatureTokenKinds.def"
  };

  if (!tryConsumeExpectedToken(Expected))
    return std::nullopt;

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
RootSignatureParser::parseRootDescriptorFlags() {
  assert(CurToken.TokKind == TokenKind::pu_equal &&
         "Expects to only be invoked starting at given keyword");

  // Handle the edge-case of '0' to specify no flags set
  if (tryConsumeExpectedToken(TokenKind::int_literal)) {
    if (!verifyZeroFlag()) {
      getDiags().Report(CurToken.TokLoc, diag::err_hlsl_rootsig_non_zero_flag);
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
    }
  } while (tryConsumeExpectedToken(TokenKind::pu_or));

  return Flags;
}

std::optional<llvm::dxbc::DescriptorRangeFlags>
RootSignatureParser::parseDescriptorRangeFlags() {
  assert(CurToken.TokKind == TokenKind::pu_equal &&
         "Expects to only be invoked starting at given keyword");

  // Handle the edge-case of '0' to specify no flags set
  if (tryConsumeExpectedToken(TokenKind::int_literal)) {
    if (!verifyZeroFlag()) {
      getDiags().Report(CurToken.TokLoc, diag::err_hlsl_rootsig_non_zero_flag);
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
    }
  } while (tryConsumeExpectedToken(TokenKind::pu_or));

  return Flags;
}

std::optional<uint32_t> RootSignatureParser::handleUIntLiteral() {
  // Parse the numeric value and do semantic checks on its specification
  clang::NumericLiteralParser Literal(CurToken.NumSpelling, CurToken.TokLoc,
                                      PP.getSourceManager(), PP.getLangOpts(),
                                      PP.getTargetInfo(), PP.getDiagnostics());
  if (Literal.hadError)
    return std::nullopt; // Error has already been reported so just return

  assert(Literal.isIntegerLiteral() &&
         "NumSpelling can only consist of digits");

  llvm::APSInt Val(32, /*IsUnsigned=*/true);
  if (Literal.GetIntegerValue(Val)) {
    // Report that the value has overflowed
    PP.getDiagnostics().Report(CurToken.TokLoc,
                               diag::err_hlsl_number_literal_overflow)
        << /*integer type*/ 0 << /*is signed*/ 0;
    return std::nullopt;
  }

  return Val.getExtValue();
}

std::optional<int32_t> RootSignatureParser::handleIntLiteral(bool Negated) {
  // Parse the numeric value and do semantic checks on its specification
  clang::NumericLiteralParser Literal(CurToken.NumSpelling, CurToken.TokLoc,
                                      PP.getSourceManager(), PP.getLangOpts(),
                                      PP.getTargetInfo(), PP.getDiagnostics());
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
    PP.getDiagnostics().Report(CurToken.TokLoc,
                               diag::err_hlsl_number_literal_overflow)
        << /*integer type*/ 0 << /*is signed*/ 1;
    return std::nullopt;
  }

  if (Negated)
    Val = -Val;

  return int32_t(Val.getExtValue());
}

std::optional<float> RootSignatureParser::handleFloatLiteral(bool Negated) {
  // Parse the numeric value and do semantic checks on its specification
  clang::NumericLiteralParser Literal(CurToken.NumSpelling, CurToken.TokLoc,
                                      PP.getSourceManager(), PP.getLangOpts(),
                                      PP.getTargetInfo(), PP.getDiagnostics());
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
    PP.getDiagnostics().Report(CurToken.TokLoc,
                               diag::err_hlsl_number_literal_underflow);
    return std::nullopt;
  }

  if (Status & llvm::APFloat::opStatus::opOverflow) {
    // Report that the value has overflowed
    PP.getDiagnostics().Report(CurToken.TokLoc,
                               diag::err_hlsl_number_literal_overflow)
        << /*float type*/ 1;
    return std::nullopt;
  }

  if (Negated)
    Val = -Val;

  double DoubleVal = Val.convertToDouble();
  double FloatMax = double(std::numeric_limits<float>::max());
  if (FloatMax < DoubleVal || DoubleVal < -FloatMax) {
    // Report that the value has overflowed
    PP.getDiagnostics().Report(CurToken.TokLoc,
                               diag::err_hlsl_number_literal_overflow)
        << /*float type*/ 1;
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
