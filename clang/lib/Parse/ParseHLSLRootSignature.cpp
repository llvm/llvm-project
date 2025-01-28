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
  if (TryConsumeExpectedToken(TokenKind::pu_r_paren)) {
    Elements.push_back(Table);
    return false;
  }

  // Iterate through all the defined clauses
  do {
    if (ParseDescriptorTableClause())
      return true;
    Table.NumClauses++;
  } while (TryConsumeExpectedToken(TokenKind::pu_comma));

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
  if (ParseRegister(&Clause.Register))
    return true;

  // Define optional paramaters
  llvm::SmallDenseMap<TokenKind, ParamType> RefMap = {
      {TokenKind::kw_numDescriptors, &Clause.NumDescriptors},
  };
  if (ParseOptionalParams({RefMap}))
    return true;

  if (ConsumeExpectedToken(TokenKind::pu_r_paren))
    return true;

  Elements.push_back(Clause);
  return false;
}

// Helper struct so that we can use the overloaded notation of std::visit
template <class... Ts> struct ParseMethods : Ts... {
  using Ts::operator()...;
};
template <class... Ts> ParseMethods(Ts...) -> ParseMethods<Ts...>;

bool RootSignatureParser::ParseParam(ParamType Ref) {
  if (ConsumeExpectedToken(TokenKind::pu_equal))
    return true;

  bool Error;
  std::visit(ParseMethods{
    [&](uint32_t *X) { Error = ParseUInt(X); },
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

  while (TryConsumeExpectedToken(TokenKind::pu_comma)) {
    if (ConsumeExpectedToken(ParamKeywords))
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
          {TokenKind::bReg, TokenKind::tReg, TokenKind::uReg, TokenKind::sReg}))
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
  if (ConsumeExpectedToken(TokenKind::int_literal))
    return true;

  if (HandleUIntLiteral(*X))
    return true; // propogate NumericLiteralParser error

  return false;
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

bool RootSignatureParser::ConsumeExpectedToken(TokenKind Expected) {
  return ConsumeExpectedToken(ArrayRef{Expected});
}

bool RootSignatureParser::ConsumeExpectedToken(
    ArrayRef<TokenKind> AnyExpected) {
  ConsumeNextToken();
  if (IsExpectedToken(CurToken.Kind, AnyExpected))
    return false;

  // Report unexpected token kind error
  Diags().Report(CurToken.TokLoc, diag::err_hlsl_rootsig_unexpected_token_kind)
      << (unsigned)(AnyExpected.size() != 1)
      << FormatTokenKinds({CurToken.Kind})
      << FormatTokenKinds(AnyExpected);
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
