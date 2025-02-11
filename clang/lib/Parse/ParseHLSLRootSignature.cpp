#include "clang/Parse/ParseHLSLRootSignature.h"

namespace clang {
namespace hlsl {

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
    Result = RootSignatureToken(SourceLoc);
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
  } else if (LexToken(Result))
    return std::nullopt; // propogate lex error up
  NextToken = Result;
  return Result;
}

} // namespace hlsl
} // namespace clang
