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

std::optional<RootFlags> RootSignatureParser::parseRootFlags() {
  assert(CurToken.TokKind == TokenKind::kw_RootFlags &&
         "Expects to only be invoked starting at given keyword");

  if (consumeExpectedToken(TokenKind::pu_l_paren, diag::err_expected_after,
                           CurToken.TokKind))
    return std::nullopt;

  std::optional<RootFlags> Flags = RootFlags::None;

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
    Flags = maybeOrFlag<RootFlags>(Flags, RootFlags::NAME);                    \
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
  std::optional<ShaderVisibility> Visibility;

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

std::optional<llvm::hlsl::rootsig::ShaderVisibility>
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
    return ShaderVisibility::NAME;                                             \
    break;
#include "clang/Lex/HLSLRootSignatureTokenKinds.def"
  default:
    llvm_unreachable("Switch for consumed enum token was not provided");
  }

  return std::nullopt;
}

std::optional<llvm::hlsl::rootsig::RootDescriptorFlags>
RootSignatureParser::parseRootDescriptorFlags() {
  assert(CurToken.TokKind == TokenKind::pu_equal &&
         "Expects to only be invoked starting at given keyword");

  // Handle the edge-case of '0' to specify no flags set
  if (tryConsumeExpectedToken(TokenKind::int_literal)) {
    if (!verifyZeroFlag()) {
      getDiags().Report(CurToken.TokLoc, diag::err_hlsl_rootsig_non_zero_flag);
      return std::nullopt;
    }
    return RootDescriptorFlags::None;
  }

  TokenKind Expected[] = {
#define ROOT_DESCRIPTOR_FLAG_ENUM(NAME, LIT) TokenKind::en_##NAME,
#include "clang/Lex/HLSLRootSignatureTokenKinds.def"
  };

  std::optional<RootDescriptorFlags> Flags;

  do {
    if (tryConsumeExpectedToken(Expected)) {
      switch (CurToken.TokKind) {
#define ROOT_DESCRIPTOR_FLAG_ENUM(NAME, LIT)                                   \
  case TokenKind::en_##NAME:                                                   \
    Flags =                                                                    \
        maybeOrFlag<RootDescriptorFlags>(Flags, RootDescriptorFlags::NAME);    \
    break;
#include "clang/Lex/HLSLRootSignatureTokenKinds.def"
      default:
        llvm_unreachable("Switch for consumed enum token was not provided");
      }
    }
  } while (tryConsumeExpectedToken(TokenKind::pu_or));

  return Flags;
}

std::optional<llvm::hlsl::rootsig::DescriptorRangeFlags>
RootSignatureParser::parseDescriptorRangeFlags() {
  assert(CurToken.TokKind == TokenKind::pu_equal &&
         "Expects to only be invoked starting at given keyword");

  // Handle the edge-case of '0' to specify no flags set
  if (tryConsumeExpectedToken(TokenKind::int_literal)) {
    if (!verifyZeroFlag()) {
      getDiags().Report(CurToken.TokLoc, diag::err_hlsl_rootsig_non_zero_flag);
      return std::nullopt;
    }
    return DescriptorRangeFlags::None;
  }

  TokenKind Expected[] = {
#define DESCRIPTOR_RANGE_FLAG_ENUM(NAME, LIT, ON) TokenKind::en_##NAME,
#include "clang/Lex/HLSLRootSignatureTokenKinds.def"
  };

  std::optional<DescriptorRangeFlags> Flags;

  do {
    if (tryConsumeExpectedToken(Expected)) {
      switch (CurToken.TokKind) {
#define DESCRIPTOR_RANGE_FLAG_ENUM(NAME, LIT, ON)                              \
  case TokenKind::en_##NAME:                                                   \
    Flags =                                                                    \
        maybeOrFlag<DescriptorRangeFlags>(Flags, DescriptorRangeFlags::NAME);  \
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
    return true; // Error has already been reported so just return

  assert(Literal.isIntegerLiteral() && "IsNumberChar will only support digits");

  llvm::APSInt Val = llvm::APSInt(32, false);
  if (Literal.GetIntegerValue(Val)) {
    // Report that the value has overflowed
    PP.getDiagnostics().Report(CurToken.TokLoc,
                               diag::err_hlsl_number_literal_overflow)
        << 0 << CurToken.NumSpelling;
    return std::nullopt;
  }

  return Val.getExtValue();
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
