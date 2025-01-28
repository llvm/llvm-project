#include "clang/Parse/ParseHLSLRootSignature.h"

#include "llvm/Support/raw_ostream.h"

using namespace llvm::hlsl::rootsig;

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
    case TokenKind::error:
      break;
    case TokenKind::invalid:
      Out << "invalid identifier";
      break;
    case TokenKind::end_of_stream:
      Out << "end of stream";
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
  // TODO(#126565): extend for float support exponents
  return isdigit(C); // integer support
}

bool RootSignatureLexer::LexNumber(RootSignatureToken &Result) {
  // Retrieve the possible number
  StringRef NumSpelling = Buffer.take_while(IsNumberChar);

  // Parse the numeric value and do semantic checks on its specification
  clang::NumericLiteralParser Literal(NumSpelling, SourceLoc,
                                      PP.getSourceManager(), PP.getLangOpts(),
                                      PP.getTargetInfo(), PP.getDiagnostics());
  if (Literal.hadError)
    return true; // Error has already been reported so just return

  // Note: if IsNumberChar allows for hexidecimal we will need to turn this
  // into a diagnostics for potential fixed-point literals
  assert(Literal.isIntegerLiteral() && "IsNumberChar will only support digits");

  // Retrieve the number value to store into the token

  llvm::APSInt X = llvm::APSInt(32, true);
  if (Literal.GetIntegerValue(X)) {
    // Report that the value has overflowed
    PP.getDiagnostics().Report(Result.TokLoc,
                               diag::err_hlsl_number_literal_overflow)
        << NumSpelling;
    return true;
  }

  Result.Kind = TokenKind::int_literal;
  Result.NumLiteral = APValue(X);

  AdvanceBuffer(NumSpelling.size());
  return false;
}

bool RootSignatureLexer::LexToken(RootSignatureToken &Result) {
  // Discard any leading whitespace
  AdvanceBuffer(Buffer.take_while(isspace).size());

  // Record where this token is in the text for usage in parser diagnostics
  Result = RootSignatureToken(SourceLoc);

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
  if (isdigit(C))
    return LexNumber(Result);

  // All following tokens require at least one additional character
  if (Buffer.size() <= 1) {
    Result = RootSignatureToken(TokenKind::invalid, SourceLoc);
    return false;
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
  Result.Kind = Switch.Default(TokenKind::invalid);
  AdvanceBuffer(TokSpelling.size());
  return false;
}

bool RootSignatureLexer::ConsumeToken() {
  // If we previously peeked then just copy the value over
  if (NextToken && NextToken->Kind != TokenKind::end_of_stream) {
    // Propogate error up if error was encountered during previous peek
    if (NextToken->Kind == TokenKind::error)
      return true;
    CurToken = *NextToken;
    NextToken = std::nullopt;
    return false;
  }

  // This will be implicity be true if NextToken->Kind == end_of_stream
  if (EndOfBuffer()) {
    CurToken = RootSignatureToken(TokenKind::end_of_stream, SourceLoc);
    return false;
  }

  return LexToken(CurToken);
}

std::optional<RootSignatureToken> RootSignatureLexer::PeekNextToken() {
  // Already peeked from the current token
  if (NextToken.has_value())
    return NextToken;

  RootSignatureToken Result;
  if (EndOfBuffer()) {
    Result = RootSignatureToken(TokenKind::end_of_stream, SourceLoc);
  } else if (LexToken(Result)) { // propogate lex error up
    // store error token to prevent further peeking
    NextToken = RootSignatureToken();
    return std::nullopt;
  }
  NextToken = Result;
  return Result;
}

// Parser Definitions

RootSignatureParser::RootSignatureParser(SmallVector<RootElement> &Elements,
                                         RootSignatureLexer &Lexer,
                                         DiagnosticsEngine &Diags)
    : Elements(Elements), Lexer(Lexer), Diags(Diags) {}

bool RootSignatureParser::Parse() {
  // Handle edge-case of empty RootSignature()
  if (Lexer.EndOfBuffer())
    return false;

  // Iterate as many RootElements as possible
  while (!ParseRootElement()) {
    if (Lexer.EndOfBuffer())
      return false;
    if (ConsumeExpectedToken(TokenKind::pu_comma))
      return true;
  }

  return true;
}

bool RootSignatureParser::ParseRootElement() {
  if (ConsumeExpectedToken(TokenKind::kw_DescriptorTable))
    return true;

  // Dispatch onto the correct parse method
  switch (CurToken.Kind) {
  case TokenKind::kw_DescriptorTable:
    return ParseDescriptorTable();
  default:
    llvm_unreachable("Switch for an expected token was not provided");
  }
  return true;
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

  // Iterate through all the defined clauses
  do {
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
  if (ConsumeExpectedToken(TokenKind::pu_l_paren))
    return true;

  // Consume mandatory Register paramater
  if (ConsumeExpectedToken(
          {TokenKind::bReg, TokenKind::tReg, TokenKind::uReg, TokenKind::sReg}))
    return true;
  if (ParseRegister(&Clause.Register))
    return true;

  // Parse optional paramaters
  llvm::SmallDenseMap<TokenKind, ParamType> RefMap = {
      {TokenKind::kw_numDescriptors, &Clause.NumDescriptors},
      {TokenKind::kw_space, &Clause.Space},
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
  std::visit(OverloadedMethods{[&](uint32_t *X) { Error = ParseUInt(X); },
  }, Ref);

  return Error;
}

bool RootSignatureParser::ParseOptionalParams(
    llvm::SmallDenseMap<TokenKind, ParamType> &RefMap) {
  SmallVector<TokenKind> ParamKeywords;
  for (auto RefPair : RefMap)
    ParamKeywords.push_back(RefPair.first);

  // Keep track of which keywords have been seen to report duplicates
  llvm::SmallDenseSet<TokenKind> Seen;

  while (!TryConsumeExpectedToken(TokenKind::pu_comma)) {
    if (ConsumeExpectedToken(ParamKeywords))
      return true;

    TokenKind ParamKind = CurToken.Kind;
    if (Seen.contains(ParamKind)) {
      Diags.Report(CurToken.TokLoc, diag::err_hlsl_rootsig_repeat_param)
          << FormatTokenKinds(ParamKind);
      return true;
    }
    Seen.insert(ParamKind);

    if (ParseParam(RefMap[ParamKind]))
      return true;
  }

  return false;
}

bool RootSignatureParser::ParseUInt(uint32_t *X) {
  // Treat a postively signed integer as though it is unsigned to match DXC
  TryConsumeExpectedToken(TokenKind::pu_plus);
  if (ConsumeExpectedToken(TokenKind::int_literal))
    return true;

  *X = CurToken.NumLiteral.getInt().getExtValue();
  return false;
}

bool RootSignatureParser::ParseRegister(Register *Register) {
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

  Register->Number = CurToken.NumLiteral.getInt().getExtValue();

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
  if (IsExpectedToken(CurToken.Kind, AnyExpected))
    return false;

  // Report unexpected token kind error
  Diags.Report(CurToken.TokLoc, diag::err_hlsl_rootsig_unexpected_token_kind)
      << (unsigned)(AnyExpected.size() != 1) << FormatTokenKinds(AnyExpected);
  return true;
}

bool RootSignatureParser::PeekExpectedToken(TokenKind Expected) {
  return PeekExpectedToken(ArrayRef{Expected});
}

bool RootSignatureParser::PeekExpectedToken(ArrayRef<TokenKind> AnyExpected) {
  auto Result = Lexer.PeekNextToken();
  if (!Result)
    return true;
  if (IsExpectedToken(Result->Kind, AnyExpected))
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
