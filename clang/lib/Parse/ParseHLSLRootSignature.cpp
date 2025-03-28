#include "clang/Parse/ParseHLSLRootSignature.h"

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
  // Iterate as many RootElements as possible
  while (TryConsumeExpectedToken(TokenKind::kw_DescriptorTable)) {
    // Dispatch onto parser method.
    // We guard against the unreachable here as we just ensured that CurToken
    // will be one of the kinds in the while condition
    switch (CurToken.Kind) {
    case TokenKind::kw_DescriptorTable:
      if (ParseDescriptorTable())
        return true;
      break;
    default:
      llvm_unreachable("Switch for consumed token was not provided");
    }

    if (!TryConsumeExpectedToken(TokenKind::pu_comma))
      break;
  }

  return ConsumeExpectedToken(TokenKind::end_of_stream, diag::err_expected);
}

bool RootSignatureParser::ParseDescriptorTable() {
  assert(CurToken.Kind == TokenKind::kw_DescriptorTable &&
         "Expects to only be invoked starting at given keyword");

  DescriptorTable Table;

  if (ConsumeExpectedToken(TokenKind::pu_l_paren, diag::err_expected_after,
                           CurToken.Kind))
    return true;

  // Iterate as many Clauses as possible
  while (TryConsumeExpectedToken({TokenKind::kw_CBV, TokenKind::kw_SRV,
                                  TokenKind::kw_UAV, TokenKind::kw_Sampler})) {
    if (ParseDescriptorTableClause())
      return true;

    Table.NumClauses++;

    if (!TryConsumeExpectedToken(TokenKind::pu_comma))
      break;
  }

  if (ConsumeExpectedToken(TokenKind::pu_r_paren, diag::err_expected_after,
                           CurToken.Kind))
    return true;

  Elements.push_back(Table);
  return false;
}

bool RootSignatureParser::ParseDescriptorTableClause() {
  assert((CurToken.Kind == TokenKind::kw_CBV ||
          CurToken.Kind == TokenKind::kw_SRV ||
          CurToken.Kind == TokenKind::kw_UAV ||
          CurToken.Kind == TokenKind::kw_Sampler) &&
         "Expects to only be invoked starting at given keyword");

  DescriptorTableClause Clause;
  switch (CurToken.Kind) {
  default:
    break; // Unreachable given Try + assert pattern
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

  if (ConsumeExpectedToken(TokenKind::pu_l_paren, diag::err_expected_after,
                           CurToken.Kind))
    return true;

  if (ConsumeExpectedToken(TokenKind::pu_r_paren, diag::err_expected_after,
                           CurToken.Kind))
    return true;

  Elements.push_back(Clause);
  return false;
}

// Returns true when given token is one of the expected kinds
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
                                               TokenKind Context) {
  return ConsumeExpectedToken(ArrayRef{Expected}, DiagID, Context);
}

bool RootSignatureParser::ConsumeExpectedToken(ArrayRef<TokenKind> AnyExpected,
                                               unsigned DiagID,
                                               TokenKind Context) {
  if (TryConsumeExpectedToken(AnyExpected))
    return false;

  // Report unexpected token kind error
  DiagnosticBuilder DB = Diags().Report(CurToken.TokLoc, DiagID);
  switch (DiagID) {
  case diag::err_expected:
    DB << FormatTokenKinds(AnyExpected);
    break;
  case diag::err_expected_either:
  case diag::err_expected_after:
    DB << FormatTokenKinds(AnyExpected) << FormatTokenKinds({Context});
    break;
  default:
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
