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

static std::string FormatTokenKinds(ArrayRef<TokenKind> Kinds) {
  std::string TokenString;
  llvm::raw_string_ostream Out(TokenString);
  bool First = true;
  for (auto Kind : Kinds) {
    if (!First)
      Out << ", ";
    switch (Kind) {
#define TOK(X, SPELLING)                                                       \
  case TokenKind::X:                                                           \
    Out << SPELLING;                                                           \
    break;
#include "clang/Lex/HLSLRootSignatureTokenKinds.def"
    }
    First = false;
  }

  return TokenString;
}

// Parser Definitions

RootSignatureParser::RootSignatureParser(SmallVector<RootElement> &Elements,
                                         RootSignatureLexer &Lexer,
                                         Preprocessor &PP)
    : Elements(Elements), Lexer(Lexer), PP(PP), CurToken(SourceLocation()) {}

bool RootSignatureParser::Parse() {
  // Handle edge-case of empty RootSignature()
  if (Lexer.EndOfBuffer())
    return false;

  // Iterate as many RootElements as possible
  while (!ParseRootElement()) {
    if (Lexer.EndOfBuffer())
      return false;
    if (ConsumeExpectedToken(TokenKind::pu_comma, diag::err_expected_either,
                             "end of root signature string"))
      return true;
  }

  return true;
}

bool RootSignatureParser::ParseRootElement() {
  if (ConsumeExpectedToken(TokenKind::kw_DescriptorTable,
                           diag::err_hlsl_expected, "root element"))
    return true;

  // Dispatch onto the correct parse method
  switch (CurToken.Kind) {
  case TokenKind::kw_DescriptorTable:
    return ParseDescriptorTable();
  default:
    break;
  }
  llvm_unreachable("Switch for an expected token was not provided");
}

bool RootSignatureParser::ParseDescriptorTable() {
  DescriptorTable Table;

  if (ConsumeExpectedToken(TokenKind::pu_l_paren, diag::err_expected_after,
                           "DescriptorTable"))
    return true;

  // Empty case:
  if (TryConsumeExpectedToken(TokenKind::pu_r_paren)) {
    Elements.push_back(Table);
    return false;
  }

  bool SeenVisibility = false;
  // Iterate through all the defined clauses
  do {
    // Handle the visibility parameter
    if (TryConsumeExpectedToken(TokenKind::kw_visibility)) {
      if (SeenVisibility) {
        Diags().Report(CurToken.TokLoc, diag::err_hlsl_rootsig_repeat_param)
            << FormatTokenKinds(CurToken.Kind);
        return true;
      }
      SeenVisibility = true;
      if (ParseParam(&Table.Visibility))
        return true;
      continue;
    }

    // Otherwise, we expect a clause
    if (ParseDescriptorTableClause())
      return true;
    Table.NumClauses++;
  } while (TryConsumeExpectedToken(TokenKind::pu_comma));

  if (ConsumeExpectedToken(TokenKind::pu_r_paren, diag::err_expected_after,
                           "descriptor table clauses"))
    return true;

  Elements.push_back(Table);
  return false;
}

bool RootSignatureParser::ParseDescriptorTableClause() {
  if (ConsumeExpectedToken({TokenKind::kw_CBV, TokenKind::kw_SRV,
                            TokenKind::kw_UAV, TokenKind::kw_Sampler},
                           diag::err_hlsl_expected, "descriptor table clause"))
    return true;

  DescriptorTableClause Clause;
  switch (CurToken.Kind) {
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
  default:
    llvm_unreachable("Switch for an expected token was not provided");
  }
  Clause.SetDefaultFlags();

  if (ConsumeExpectedToken(TokenKind::pu_l_paren, diag::err_expected_after,
                           FormatTokenKinds({CurToken.Kind})))
    return true;

  // Consume mandatory Register paramater
  if (ParseRegister(&Clause.Register))
    return true;

  // Define optional paramaters
  llvm::SmallDenseMap<TokenKind, ParamType> RefMap = {
      {TokenKind::kw_numDescriptors, &Clause.NumDescriptors},
      {TokenKind::kw_space, &Clause.Space},
      {TokenKind::kw_offset, &Clause.Offset},
      {TokenKind::kw_flags, &Clause.Flags},
  };
  if (ParseOptionalParams({RefMap}))
    return true;

  if (ConsumeExpectedToken(TokenKind::pu_r_paren, diag::err_expected_after,
                           "clause parameters"))
    return true;

  Elements.push_back(Clause);
  return false;
}

// Helper struct so that we can use the overloaded notation of std::visit
template <class... Ts> struct ParseMethods : Ts... { using Ts::operator()...; };
template <class... Ts> ParseMethods(Ts...) -> ParseMethods<Ts...>;

bool RootSignatureParser::ParseParam(ParamType Ref) {
  if (ConsumeExpectedToken(TokenKind::pu_equal, diag::err_expected_after,
                           FormatTokenKinds(CurToken.Kind)))
    return true;

  bool Error;
  std::visit(
      ParseMethods{
          [&](uint32_t *X) { Error = ParseUInt(X); },
          [&](DescriptorRangeOffset *X) {
            Error = ParseDescriptorRangeOffset(X);
          },
          [&](ShaderVisibility *Enum) { Error = ParseShaderVisibility(Enum); },
          [&](DescriptorRangeFlags *Flags) {
            Error = ParseDescriptorRangeFlags(Flags);
          },
      },
      Ref);

  return Error;
}

bool RootSignatureParser::ParseOptionalParams(
    llvm::SmallDenseMap<TokenKind, ParamType> &RefMap) {
  SmallVector<TokenKind> ParamKeywords;
  for (auto RefPair : RefMap)
    ParamKeywords.push_back(RefPair.first);

  // Keep track of which keywords have been seen to report duplicates
  llvm::SmallDenseSet<TokenKind> Seen;

  while (TryConsumeExpectedToken(TokenKind::pu_comma)) {
    if (ConsumeExpectedToken(ParamKeywords, diag::err_hlsl_expected,
                             "optional parameter"))
      return true;

    TokenKind ParamKind = CurToken.Kind;
    if (Seen.contains(ParamKind)) {
      Diags().Report(CurToken.TokLoc, diag::err_hlsl_rootsig_repeat_param)
          << FormatTokenKinds({ParamKind});
      return true;
    }
    Seen.insert(ParamKind);

    if (ParseParam(RefMap[ParamKind]))
      return true;
  }

  return false;
}

bool RootSignatureParser::HandleUIntLiteral(uint32_t &X) {
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

bool RootSignatureParser::ParseRegister(Register *Register) {
  if (ConsumeExpectedToken(
          {TokenKind::bReg, TokenKind::tReg, TokenKind::uReg, TokenKind::sReg},
          diag::err_hlsl_expected, "a register"))
    return true;

  switch (CurToken.Kind) {
  case TokenKind::bReg:
    Register->ViewType = RegisterType::BReg;
    break;
  case TokenKind::tReg:
    Register->ViewType = RegisterType::TReg;
    break;
  case TokenKind::uReg:
    Register->ViewType = RegisterType::UReg;
    break;
  case TokenKind::sReg:
    Register->ViewType = RegisterType::SReg;
    break;
  default:
    llvm_unreachable("Switch for an expected token was not provided");
  }

  if (HandleUIntLiteral(Register->Number))
    return true; // propogate NumericLiteralParser error

  return false;
}

bool RootSignatureParser::ParseUInt(uint32_t *X) {
  // Treat a postively signed integer as though it is unsigned to match DXC
  TryConsumeExpectedToken(TokenKind::pu_plus);
  if (ConsumeExpectedToken(TokenKind::int_literal, diag::err_hlsl_expected,
                           "unsigned integer"))
    return true;

  if (HandleUIntLiteral(*X))
    return true; // propogate NumericLiteralParser error

  return false;
}

bool RootSignatureParser::ParseDescriptorRangeOffset(DescriptorRangeOffset *X) {
  if (ConsumeExpectedToken(
          {TokenKind::int_literal, TokenKind::en_DescriptorRangeOffsetAppend},
          diag::err_hlsl_expected, "descriptor range offset"))
    return true;

  // Edge case for the offset enum -> static value
  if (CurToken.Kind == TokenKind::en_DescriptorRangeOffsetAppend) {
    *X = DescriptorTableOffsetAppend;
    return false;
  }

  uint32_t Temp;
  if (HandleUIntLiteral(Temp))
    return true; // propogate NumericLiteralParser error
  *X = DescriptorRangeOffset(Temp);
  return false;
}

template <bool AllowZero, typename EnumType>
bool RootSignatureParser::ParseEnum(
    llvm::SmallDenseMap<TokenKind, EnumType> &EnumMap, EnumType *Enum) {
  SmallVector<TokenKind> EnumToks;
  if (AllowZero)
    EnumToks.push_back(TokenKind::int_literal); //  '0' is a valid flag value
  for (auto EnumPair : EnumMap)
    EnumToks.push_back(EnumPair.first);

  // If invoked we expect to have an enum
  if (ConsumeExpectedToken(EnumToks, diag::err_hlsl_expected,
                           "parameter value"))
    return true;

  // Handle the edge case when '0' is used to specify None
  if (CurToken.Kind == TokenKind::int_literal) {
    uint32_t Temp;
    if (HandleUIntLiteral(Temp))
      return true; // propogate NumericLiteralParser error
    if (Temp != 0) {
      Diags().Report(CurToken.TokLoc, diag::err_hlsl_rootsig_non_zero_flag);
      return true;
    }
    // Set enum to None equivalent
    *Enum = EnumType(0);
    return false;
  }

  // Effectively a switch statement on the token kinds
  for (auto EnumPair : EnumMap)
    if (CurToken.Kind == EnumPair.first) {
      *Enum = EnumPair.second;
      return false;
    }

  llvm_unreachable("Switch for an expected token was not provided");
}

bool RootSignatureParser::ParseShaderVisibility(ShaderVisibility *Enum) {
  // Define the possible flag kinds
  llvm::SmallDenseMap<TokenKind, ShaderVisibility> EnumMap = {
#define SHADER_VISIBILITY_ENUM(NAME, LIT)                                      \
  {TokenKind::en_##NAME, ShaderVisibility::NAME},
#include "clang/Lex/HLSLRootSignatureTokenKinds.def"
  };

  return ParseEnum(EnumMap, Enum);
}

template <typename FlagType>
bool RootSignatureParser::ParseFlags(
    llvm::SmallDenseMap<TokenKind, FlagType> &FlagMap, FlagType *Flags) {
  // Override the default value to 0 so that we can correctly 'or' the values
  *Flags = FlagType(0);

  do {
    FlagType Flag;
    if (ParseEnum<true>(FlagMap, &Flag))
      return true;
    // Store the 'or'
    *Flags |= Flag;
  } while (TryConsumeExpectedToken(TokenKind::pu_or));

  return false;
}

bool RootSignatureParser::ParseDescriptorRangeFlags(
    DescriptorRangeFlags *Flags) {
  // Define the possible flag kinds
  llvm::SmallDenseMap<TokenKind, DescriptorRangeFlags> FlagMap = {
#define DESCRIPTOR_RANGE_FLAG_ENUM(NAME, LIT, ON)                              \
  {TokenKind::en_##NAME, DescriptorRangeFlags::NAME},
#include "clang/Lex/HLSLRootSignatureTokenKinds.def"
  };

  return ParseFlags(FlagMap, Flags);
}

// Is given token one of the expected kinds
static bool IsExpectedToken(TokenKind Kind, ArrayRef<TokenKind> AnyExpected) {
  for (auto Expected : AnyExpected)
    if (Kind == Expected)
      return true;
  return false;
}

bool RootSignatureParser::PeekExpectedToken(TokenKind Expected) {
  return PeekExpectedToken(ArrayRef{Expected});
}

bool RootSignatureParser::PeekExpectedToken(ArrayRef<TokenKind> AnyExpected) {
  RootSignatureToken Result = Lexer.PeekNextToken();
  return IsExpectedToken(Result.Kind, AnyExpected);
}

bool RootSignatureParser::ConsumeExpectedToken(TokenKind Expected,
                                               unsigned DiagID,
                                               StringRef DiagMsg) {
  return ConsumeExpectedToken(ArrayRef{Expected}, DiagID, DiagMsg);
}

bool RootSignatureParser::ConsumeExpectedToken(ArrayRef<TokenKind> AnyExpected,
                                               unsigned DiagID,
                                               StringRef DiagMsg) {
  if (TryConsumeExpectedToken(AnyExpected))
    return false;

  // Report unexpected token kind error
  DiagnosticBuilder DB = Diags().Report(CurToken.TokLoc, DiagID);
  switch (DiagID) {
  case diag::err_expected:
    DB << FormatTokenKinds(AnyExpected);
    break;
  case diag::err_hlsl_expected:
  case diag::err_expected_either:
  case diag::err_expected_after:
    DB << FormatTokenKinds(AnyExpected) << DiagMsg;
    break;
  default:
    DB << DiagMsg;
    break;
  }
  return true;
}

bool RootSignatureParser::TryConsumeExpectedToken(TokenKind Expected) {
  return TryConsumeExpectedToken(ArrayRef{Expected});
}

bool RootSignatureParser::TryConsumeExpectedToken(
    ArrayRef<TokenKind> AnyExpected) {
  // If not the expected token just return
  if (!PeekExpectedToken(AnyExpected))
    return false;
  ConsumeNextToken();
  return true;
}

} // namespace hlsl
} // namespace clang
