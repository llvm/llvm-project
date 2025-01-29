#include "clang/Parse/ParseHLSLRootSignature.h"

#include "llvm/Support/raw_ostream.h"

using namespace llvm::hlsl::root_signature;

namespace clang {
namespace hlsl {

// Helper definitions

static std::string FormatTokenKinds(ArrayRef<TokenKind> Kinds) {
  std::string TokenString;
  llvm::raw_string_ostream Out(TokenString);
  bool First = true;
  for (auto Kind : Kinds) {
    if (!First)
      Out << ", ";
    switch (Kind) {
    case TokenKind::invalid:
      break;
    case TokenKind::int_literal:
      Out << "integer literal";
      break;
    case TokenKind::bReg:
      Out << "b register";
      break;
    case TokenKind::tReg:
      Out << "t register";
      break;
    case TokenKind::uReg:
      Out << "u register";
      break;
    case TokenKind::sReg:
      Out << "s register";
      break;
#define PUNCTUATOR(X, Y)                                                       \
  case TokenKind::pu_##X:                                                      \
    Out << #Y;                                                                 \
    break;
#define KEYWORD(NAME)                                                          \
  case TokenKind::kw_##NAME:                                                   \
    Out << #NAME;                                                              \
    break;
#define ENUM(NAME, LIT)                                                        \
  case TokenKind::en_##NAME:                                                   \
    Out << LIT;                                                                \
    break;
#include "clang/Parse/HLSLRootSignatureTokenKinds.def"
    }
    First = false;
  }

  return TokenString;
}

// Lexer Definitions

static bool IsNumberChar(char C) {
  // TODO(#120472): extend for float support exponents
  return isdigit(C); // integer support
}

bool RootSignatureLexer::LexNumber(RootSignatureToken &Result) {
  // NumericLiteralParser does not handle the sign so we will manually apply it
  bool Negative = Buffer.front() == '-';
  bool Signed = Negative || Buffer.front() == '+';
  if (Signed)
    AdvanceBuffer();

  // Retrieve the possible number
  StringRef NumSpelling = Buffer.take_while(IsNumberChar);

  // Catch this now as the Literal Parser will accept it as valid
  if (NumSpelling.empty()) {
    PP.getDiagnostics().Report(Result.TokLoc,
                               diag::err_hlsl_invalid_number_literal);
    return true;
  }

  // Parse the numeric value and do semantic checks on its specification
  clang::NumericLiteralParser Literal(NumSpelling, SourceLoc,
                                      PP.getSourceManager(), PP.getLangOpts(),
                                      PP.getTargetInfo(), PP.getDiagnostics());
  if (Literal.hadError)
    return true; // Error has already been reported so just return

  if (!Literal.isIntegerLiteral()) {
    // Note: if IsNumberChar allows for hexidecimal we will need to turn this
    // into a diagnostics for potential fixed-point literals
    llvm_unreachable("IsNumberChar will only support digits");
    return true;
  }

  // Retrieve the number value to store into the token
  Result.Kind = TokenKind::int_literal;

  llvm::APSInt X = llvm::APSInt(32, !Signed);
  if (Literal.GetIntegerValue(X)) {
    // Report that the value has overflowed
    PP.getDiagnostics().Report(Result.TokLoc,
                               diag::err_hlsl_number_literal_overflow)
        << (unsigned)Signed << NumSpelling;
    return true;
  }

  X = Negative ? -X : X;
  Result.NumLiteral = APValue(X);

  AdvanceBuffer(NumSpelling.size());
  return false;
}

bool RootSignatureLexer::Lex(SmallVector<RootSignatureToken> &Tokens) {
  // Discard any leading whitespace
  AdvanceBuffer(Buffer.take_while(isspace).size());

  while (!Buffer.empty()) {
    // Record where this token is in the text for usage in parser diagnostics
    RootSignatureToken Result(SourceLoc);
    if (LexToken(Result))
      return true;

    // Successfully Lexed the token so we can store it
    Tokens.push_back(Result);

    // Discard any trailing whitespace
    AdvanceBuffer(Buffer.take_while(isspace).size());
  }

  return false;
}

bool RootSignatureLexer::LexToken(RootSignatureToken &Result) {
  char C = Buffer.front();

  // Punctuators
  switch (C) {
#define PUNCTUATOR(X, Y)                                                       \
  case Y: {                                                                    \
    Result.Kind = TokenKind::pu_##X;                                           \
    AdvanceBuffer();                                                           \
    return false;                                                              \
  }
#include "clang/Parse/HLSLRootSignatureTokenKinds.def"
  default:
    break;
  }

  // Numeric constant
  if (isdigit(C) || C == '-' || C == '+')
    return LexNumber(Result);

  // All following tokens require at least one additional character
  if (Buffer.size() <= 1) {
    PP.getDiagnostics().Report(Result.TokLoc, diag::err_hlsl_invalid_token);
    return true;
  }

  // Peek at the next character to deteremine token type
  char NextC = Buffer[1];

  // Registers: [tsub][0-9+]
  if ((C == 't' || C == 's' || C == 'u' || C == 'b') && isdigit(NextC)) {
    AdvanceBuffer();

    if (LexNumber(Result))
      return true; // Error parsing number which is already reported

    // Lex number could also parse a float so ensure it was an unsigned int
    if (Result.Kind != TokenKind::int_literal ||
        Result.NumLiteral.getInt().isSigned()) {
      // Return invalid number literal for register error
      PP.getDiagnostics().Report(Result.TokLoc,
                                 diag::err_hlsl_invalid_register_literal);
      return true;
    }

    // Convert character to the register type.
    // This is done after LexNumber to override the TokenKind
    switch (C) {
    case 'b':
      Result.Kind = TokenKind::bReg;
      break;
    case 't':
      Result.Kind = TokenKind::tReg;
      break;
    case 'u':
      Result.Kind = TokenKind::uReg;
      break;
    case 's':
      Result.Kind = TokenKind::sReg;
      break;
    default:
      llvm_unreachable("Switch for an expected token was not provided");
      return true;
    }
    return false;
  }

  // Keywords and Enums:
  StringRef TokSpelling =
      Buffer.take_while([](char C) { return isalnum(C) || C == '_'; });

  // Define a large string switch statement for all the keywords and enums
  auto Switch = llvm::StringSwitch<TokenKind>(TokSpelling);
#define KEYWORD(NAME) Switch.Case(#NAME, TokenKind::kw_##NAME);
#define ENUM(NAME, LIT) Switch.CaseLower(LIT, TokenKind::en_##NAME);
#include "clang/Parse/HLSLRootSignatureTokenKinds.def"

  // Then attempt to retreive a string from it
  auto Kind = Switch.Default(TokenKind::invalid);
  if (Kind == TokenKind::invalid) {
    PP.getDiagnostics().Report(Result.TokLoc, diag::err_hlsl_invalid_token);
    return true;
  }

  Result.Kind = Kind;
  AdvanceBuffer(TokSpelling.size());
  return false;
}

// Parser Definitions

RootSignatureParser::RootSignatureParser(
    SmallVector<RootElement> &Elements,
    const SmallVector<RootSignatureToken> &Tokens, DiagnosticsEngine &Diags)
    : Elements(Elements), Diags(Diags) {
  CurTok = Tokens.begin();
  LastTok = Tokens.end();
}

bool RootSignatureParser::Parse() {
  // Handle edge-case of empty RootSignature()
  if (CurTok == LastTok)
    return false;

  bool First = true;
  // Iterate as many RootElements as possible
  while (!ParseRootElement(First)) {
    First = false;
    // Avoid use of ConsumeNextToken here to skip incorrect end of tokens error
    CurTok++;
    if (CurTok == LastTok)
      return false;
    if (EnsureExpectedToken(TokenKind::pu_comma))
      return true;
  }

  return true;
}

bool RootSignatureParser::ParseRootElement(bool First) {
  if (First && EnsureExpectedToken(TokenKind::kw_DescriptorTable))
    return true;
  if (!First && ConsumeExpectedToken(TokenKind::kw_DescriptorTable))
    return true;

  // Dispatch onto the correct parse method
  switch (CurTok->Kind) {
  case TokenKind::kw_DescriptorTable:
    return ParseDescriptorTable();
  default:
    llvm_unreachable("Switch for an expected token was not provided");
    return true;
  }
}

bool RootSignatureParser::ParseDescriptorTable() {
  DescriptorTable Table;

  if (ConsumeExpectedToken(TokenKind::pu_l_paren))
    return true;

  // Empty case:
  if (!TryConsumeExpectedToken(TokenKind::pu_r_paren)) {
    Elements.push_back(Table);
    return false;
  }

  bool SeenVisibility = false;
  // Iterate through all the defined clauses
  do {
    // Handle the visibility parameter
    if (!TryConsumeExpectedToken(TokenKind::kw_visibility)) {
      if (SeenVisibility) {
        Diags.Report(CurTok->TokLoc, diag::err_hlsl_rootsig_repeat_param)
            << FormatTokenKinds(CurTok->Kind);
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
  } while (!TryConsumeExpectedToken(TokenKind::pu_comma));

  if (ConsumeExpectedToken(TokenKind::pu_r_paren))
    return true;

  Elements.push_back(Table);
  return false;
}

bool RootSignatureParser::ParseDescriptorTableClause() {
  if (ConsumeExpectedToken({TokenKind::kw_CBV, TokenKind::kw_SRV,
                            TokenKind::kw_UAV, TokenKind::kw_Sampler}))
    return true;

  DescriptorTableClause Clause;
  switch (CurTok->Kind) {
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
    return true;
  }
  Clause.SetDefaultFlags();

  if (ConsumeExpectedToken(TokenKind::pu_l_paren))
    return true;

  // Consume mandatory Register paramater
  if (ConsumeExpectedToken(
          {TokenKind::bReg, TokenKind::tReg, TokenKind::uReg, TokenKind::sReg}))
    return true;
  if (ParseRegister(&Clause.Register))
    return true;

  // Parse optional paramaters
  llvm::SmallDenseMap<TokenKind, rs::ParamType> RefMap = {
      {TokenKind::kw_numDescriptors, &Clause.NumDescriptors},
      {TokenKind::kw_space, &Clause.Space},
      {TokenKind::kw_offset, &Clause.Offset},
      {TokenKind::kw_flags, &Clause.Flags},
  };
  if (ParseOptionalParams({RefMap}))
    return true;

  if (ConsumeExpectedToken(TokenKind::pu_r_paren))
    return true;

  Elements.push_back(Clause);
  return false;
}

// Helper struct so that we can use the overloaded notation of std::visit
template <class... Ts> struct OverloadedMethods : Ts... {
  using Ts::operator()...;
};
template <class... Ts> OverloadedMethods(Ts...) -> OverloadedMethods<Ts...>;

bool RootSignatureParser::ParseParam(ParamType Ref) {
  if (ConsumeExpectedToken(TokenKind::pu_equal))
    return true;

  bool Error;
  std::visit(
      OverloadedMethods{
          [&](uint32_t *X) { Error = ParseUInt(X); },
          [&](DescriptorRangeOffset *X) {
            Error = ParseDescriptorRangeOffset(X);
          },
          [&](DescriptorRangeFlags *Flags) {
            Error = ParseDescriptorRangeFlags(Flags);
          },
          [&](ShaderVisibility *Enum) { Error = ParseShaderVisibility(Enum); },
      },
      Ref);

  return Error;
}

bool RootSignatureParser::ParseOptionalParams(
    llvm::SmallDenseMap<TokenKind, rs::ParamType> RefMap) {
  SmallVector<TokenKind> ParamKeywords;
  for (auto RefPair : RefMap)
    ParamKeywords.push_back(RefPair.first);

  // Keep track of which keywords have been seen to report duplicates
  llvm::SmallDenseSet<TokenKind> Seen;

  while (!TryConsumeExpectedToken(TokenKind::pu_comma)) {
    if (ConsumeExpectedToken(ParamKeywords))
      return true;

    TokenKind ParamKind = CurTok->Kind;
    if (Seen.contains(ParamKind)) {
      Diags.Report(CurTok->TokLoc, diag::err_hlsl_rootsig_repeat_param)
          << FormatTokenKinds(ParamKind);
      return true;
    }
    Seen.insert(ParamKind);

    if (ParseParam(RefMap[ParamKind]))
      return true;
  }

  return false;
}

bool RootSignatureParser::ParseDescriptorRangeOffset(DescriptorRangeOffset *X) {
  if (ConsumeExpectedToken(
          {TokenKind::int_literal, TokenKind::en_DescriptorRangeOffsetAppend}))
    return true;

  // Edge case for the offset enum -> static value
  if (CurTok->Kind == TokenKind::en_DescriptorRangeOffsetAppend) {
    *X = DescriptorTableOffsetAppend;
    return false;
  }

  *X = DescriptorRangeOffset(CurTok->NumLiteral.getInt().getExtValue());
  return false;
}

bool RootSignatureParser::ParseUInt(uint32_t *X) {
  if (ConsumeExpectedToken(TokenKind::int_literal))
    return true;

  *X = CurTok->NumLiteral.getInt().getExtValue();
  return false;
}

bool RootSignatureParser::ParseRegister(Register *Register) {
  switch (CurTok->Kind) {
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
    return true;
  }

  Register->Number = CurTok->NumLiteral.getInt().getExtValue();

  return false;
}

template <bool AllowZero, typename EnumType>
bool RootSignatureParser::ParseEnum(
    llvm::SmallDenseMap<TokenKind, EnumType> EnumMap, EnumType *Enum) {
  SmallVector<TokenKind> EnumToks;
  if (AllowZero)
    EnumToks.push_back(TokenKind::int_literal); //  '0' is a valid flag value
  for (auto EnumPair : EnumMap)
    EnumToks.push_back(EnumPair.first);

  // If invoked we expect to have an enum
  if (ConsumeExpectedToken(EnumToks))
    return true;

  // Handle the edge case when '0' is used to specify None
  if (CurTok->Kind == TokenKind::int_literal) {
    if (CurTok->NumLiteral.getInt() != 0) {
      Diags.Report(CurTok->TokLoc, diag::err_hlsl_rootsig_non_zero_flag);
      return true;
    }
    // Set enum to None equivalent
    *Enum = EnumType(0);
    return false;
  }

  // Effectively a switch statement on the token kinds
  for (auto EnumPair : EnumMap)
    if (CurTok->Kind == EnumPair.first) {
      *Enum = EnumPair.second;
      return false;
    }

  llvm_unreachable("Switch for an expected token was not provided");
  return true;
}

template <typename FlagType>
bool RootSignatureParser::ParseFlags(
    llvm::SmallDenseMap<TokenKind, FlagType> FlagMap, FlagType *Flags) {
  // Override the default value to 0 so that we can correctly 'or' the values
  *Flags = FlagType(0);

  do {
    FlagType Flag;
    if (ParseEnum<true>(FlagMap, &Flag))
      return true;
    // Store the 'or'
    *Flags |= Flag;

  } while (!TryConsumeExpectedToken(TokenKind::pu_or));

  return false;
}

bool RootSignatureParser::ParseDescriptorRangeFlags(
    DescriptorRangeFlags *Flags) {
  // Define the possible flag kinds
  llvm::SmallDenseMap<TokenKind, DescriptorRangeFlags> FlagMap = {
#define DESCRIPTOR_RANGE_FLAG_ENUM(NAME, LIT, ON)                              \
  {TokenKind::en_##NAME, DescriptorRangeFlags::NAME},
#include "clang/Parse/HLSLRootSignatureTokenKinds.def"
  };

  return ParseFlags(FlagMap, Flags);
}

bool RootSignatureParser::ParseShaderVisibility(ShaderVisibility *Enum) {
  // Define the possible flag kinds
  llvm::SmallDenseMap<TokenKind, ShaderVisibility> EnumMap = {
#define SHADER_VISIBILITY_ENUM(NAME, LIT)                                      \
  {TokenKind::en_##NAME, ShaderVisibility::NAME},
#include "clang/Parse/HLSLRootSignatureTokenKinds.def"
  };

  return ParseEnum(EnumMap, Enum);
}

RootSignatureToken RootSignatureParser::PeekNextToken() {
  // Create an invalid token
  RootSignatureToken Token = RootSignatureToken(SourceLocation());
  if (CurTok != LastTok)
    Token = *(CurTok + 1);
  return Token;
}

bool RootSignatureParser::ConsumeNextToken() {
  SourceLocation EndLoc = CurTok->TokLoc;
  CurTok++;
  if (LastTok == CurTok) {
    // Report unexpected end of tokens error
    Diags.Report(EndLoc, diag::err_hlsl_rootsig_unexpected_eos);
    return true;
  }
  return false;
}

// Is given token one of the expected kinds
static bool IsExpectedToken(TokenKind Kind, ArrayRef<TokenKind> AnyExpected) {
  for (auto Expected : AnyExpected)
    if (Kind == Expected)
      return true;
  return false;
}

bool RootSignatureParser::EnsureExpectedToken(TokenKind Expected) {
  return EnsureExpectedToken(ArrayRef{Expected});
}

bool RootSignatureParser::EnsureExpectedToken(ArrayRef<TokenKind> AnyExpected) {
  if (IsExpectedToken(CurTok->Kind, AnyExpected))
    return false;

  // Report unexpected token kind error
  Diags.Report(CurTok->TokLoc, diag::err_hlsl_rootsig_unexpected_token_kind)
      << (unsigned)(AnyExpected.size() != 1) << FormatTokenKinds(AnyExpected);
  return true;
}

bool RootSignatureParser::PeekExpectedToken(TokenKind Expected) {
  return PeekExpectedToken(ArrayRef{Expected});
}

bool RootSignatureParser::PeekExpectedToken(ArrayRef<TokenKind> AnyExpected) {
  RootSignatureToken Token = PeekNextToken();
  if (Token.Kind == TokenKind::invalid)
    return true;
  if (IsExpectedToken(Token.Kind, AnyExpected))
    return false;
  return true;
}

bool RootSignatureParser::ConsumeExpectedToken(TokenKind Expected) {
  return ConsumeExpectedToken(ArrayRef{Expected});
}

bool RootSignatureParser::ConsumeExpectedToken(
    ArrayRef<TokenKind> AnyExpected) {
  if (ConsumeNextToken())
    return true;

  return EnsureExpectedToken(AnyExpected);
}

bool RootSignatureParser::TryConsumeExpectedToken(TokenKind Expected) {
  return TryConsumeExpectedToken(ArrayRef{Expected});
}

bool RootSignatureParser::TryConsumeExpectedToken(
    ArrayRef<TokenKind> AnyExpected) {
  if (PeekExpectedToken(AnyExpected))
    return true;
  return ConsumeNextToken();
}

} // namespace hlsl
} // namespace clang
