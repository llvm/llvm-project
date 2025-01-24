#include "clang/Parse/ParseHLSLRootSignature.h"

namespace clang {
namespace hlsl {

// Lexer Definitions

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

  // Unable to match on any token type
  PP.getDiagnostics().Report(Result.TokLoc, diag::err_hlsl_invalid_token);
  return true;
}

} // namespace hlsl
} // namespace clang
