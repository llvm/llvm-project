//=== LexHLSLRootSignature.cpp - Lex Root Signature -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Lex/LexHLSLRootSignature.h"

namespace clang {
namespace hlsl {

using TokenKind = RootSignatureToken::Kind;

// Lexer Definitions

static bool isNumberChar(char C) {
  return isdigit(C)                                      // integer support
         || C == '.'                                     // float support
         || C == 'e' || C == 'E' || C == '-' || C == '+' // exponent support
         || C == 'f' || C == 'F'; // explicit float support
}

RootSignatureToken RootSignatureLexer::lexToken() {
  // Discard any leading whitespace
  advanceBuffer(Buffer.take_while(isspace).size());

  if (isEndOfBuffer())
    return RootSignatureToken(TokenKind::end_of_stream, SourceLoc);

  // Record where this token is in the text for usage in parser diagnostics
  RootSignatureToken Result(SourceLoc);

  char C = Buffer.front();

  // Punctuators
  switch (C) {
#define PUNCTUATOR(X, Y)                                                       \
  case Y: {                                                                    \
    Result.TokKind = TokenKind::pu_##X;                                        \
    advanceBuffer();                                                           \
    return Result;                                                             \
  }
#include "clang/Lex/HLSLRootSignatureTokenKinds.def"
  default:
    break;
  }

  // Number literal
  if (isdigit(C) || C == '.') {
    Result.NumSpelling = Buffer.take_while(isNumberChar);

    // If all values are digits then we have an int literal
    bool IsInteger = Result.NumSpelling.find_if_not(isdigit) == StringRef::npos;

    Result.TokKind =
        IsInteger ? TokenKind::int_literal : TokenKind::float_literal;
    advanceBuffer(Result.NumSpelling.size());
    return Result;
  }

  // All following tokens require at least one additional character
  if (Buffer.size() <= 1) {
    Result = RootSignatureToken(TokenKind::invalid, SourceLoc);
    return Result;
  }

  // Peek at the next character to deteremine token type
  char NextC = Buffer[1];

  // Registers: [tsub][0-9+]
  if ((C == 't' || C == 's' || C == 'u' || C == 'b') && isdigit(NextC)) {
    // Convert character to the register type.
    switch (C) {
    case 'b':
      Result.TokKind = TokenKind::bReg;
      break;
    case 't':
      Result.TokKind = TokenKind::tReg;
      break;
    case 'u':
      Result.TokKind = TokenKind::uReg;
      break;
    case 's':
      Result.TokKind = TokenKind::sReg;
      break;
    default:
      llvm_unreachable("Switch for an expected token was not provided");
    }

    advanceBuffer();

    // Lex the integer literal
    Result.NumSpelling = Buffer.take_while(isNumberChar);
    advanceBuffer(Result.NumSpelling.size());

    return Result;
  }

  // Keywords and Enums:
  StringRef TokSpelling =
      Buffer.take_while([](char C) { return isalnum(C) || C == '_'; });

  // Define a large string switch statement for all the keywords and enums
  auto Switch = llvm::StringSwitch<TokenKind>(TokSpelling);
#define KEYWORD(NAME) Switch.CaseLower(#NAME, TokenKind::kw_##NAME);
#define ENUM(NAME, LIT) Switch.CaseLower(LIT, TokenKind::en_##NAME);
#include "clang/Lex/HLSLRootSignatureTokenKinds.def"

  // Then attempt to retreive a string from it
  Result.TokKind = Switch.Default(TokenKind::invalid);
  advanceBuffer(TokSpelling.size());
  return Result;
}

RootSignatureToken RootSignatureLexer::consumeToken() {
  // If we previously peeked then just return the previous value over
  if (NextToken && NextToken->TokKind != TokenKind::end_of_stream) {
    RootSignatureToken Result = *NextToken;
    NextToken = std::nullopt;
    return Result;
  }
  return lexToken();
}

RootSignatureToken RootSignatureLexer::peekNextToken() {
  // Already peeked from the current token
  if (NextToken)
    return *NextToken;

  NextToken = lexToken();
  return *NextToken;
}

} // namespace hlsl
} // namespace clang
