//===--- BracketTest.cpp -------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang-pseudo/Bracket.h"
#include "clang-pseudo/Token.h"
#include "clang/Basic/LangOptions.h"
#include "llvm/Testing/Annotations/Annotations.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace clang {
namespace pseudo {

// Return a version of Code with each paired bracket marked with ^.
std::string decorate(llvm::StringRef Code, const TokenStream &Stream) {
  std::string Result;
  const char *Pos = Code.data();
  for (const Token &Tok : Stream.tokens()) {
    if (Tok.Pair == 0)
      continue;
    const char *NewPos = Tok.text().begin();
    assert(NewPos >= Code.begin() && NewPos < Code.end());
    Result.append(Pos, NewPos - Pos);
    Result.push_back('^');
    Pos = NewPos;
  }
  Result.append(Pos, Code.end() - Pos);
  return Result;
}

// Checks that the brackets matched in Stream are those annotated in MarkedCode.
void verifyMatchedSet(llvm::StringRef Code, llvm::StringRef MarkedCode,
                      const TokenStream &Stream) {
  EXPECT_EQ(MarkedCode, decorate(Code, Stream));
}

// Checks that paired brackets within the stream nest properly.
void verifyNesting(const TokenStream &Stream) {
  std::vector<const Token *> Stack;
  for (const auto &Tok : Stream.tokens()) {
    if (Tok.Pair > 0)
      Stack.push_back(&Tok);
    else if (Tok.Pair < 0) {
      ASSERT_FALSE(Stack.empty()) << Tok;
      ASSERT_EQ(Stack.back(), Tok.pair())
          << *Stack.back() << " != " << *Tok.pair() << " = pair of " << Tok;
      Stack.pop_back();
    }
  }
  ASSERT_THAT(Stack, testing::IsEmpty());
}

// Checks that ( pairs with a ) on its right, etc.
void verifyMatchKind(const TokenStream &Stream) {
  for (const auto &Tok : Stream.tokens()) {
    if (Tok.Pair == 0)
      continue;
    auto Want = [&]() -> std::pair<bool, tok::TokenKind> {
      switch (Tok.Kind) {
      case tok::l_paren:
        return {true, tok::r_paren};
      case tok::r_paren:
        return {false, tok::l_paren};
      case tok::l_brace:
        return {true, tok::r_brace};
      case tok::r_brace:
        return {false, tok::l_brace};
      case tok::l_square:
        return {true, tok::r_square};
      case tok::r_square:
        return {false, tok::l_square};
      default:
        ADD_FAILURE() << "Paired non-bracket " << Tok;
        return {false, tok::eof};
      }
    }();
    EXPECT_EQ(Tok.Pair > 0, Want.first) << Tok;
    EXPECT_EQ(Tok.pair()->Kind, Want.second) << Tok;
  }
}

// Verifies an expected bracket pairing like:
//   ^( [ ^)
// The input is annotated code, with the brackets expected to be matched marked.
//
// The input doesn't specify which bracket matches with which, but we verify:
//  - exactly the marked subset are paired
//  - ( is paired to a later ), etc
//  - brackets properly nest
// This uniquely determines the bracket structure, so we indirectly verify it.
// If particular tests should emphasize which brackets are paired, use comments.
void verifyBrackets(llvm::StringRef MarkedCode) {
  SCOPED_TRACE(MarkedCode);
  llvm::Annotations A(MarkedCode);
  std::string Code = A.code().str();
  LangOptions LangOpts;
  auto Stream = lex(Code, LangOpts);
  pairBrackets(Stream);

  verifyMatchedSet(Code, MarkedCode, Stream);
  verifyNesting(Stream);
  verifyMatchKind(Stream);
}

TEST(Bracket, SimplePair) {
  verifyBrackets("^{ ^[ ^( ^)  ^( ^) ^] ^}");
  verifyBrackets(") ^{ ^[ ^] ^} (");
  verifyBrackets("{ [ ( ] }"); // FIXME
}

} // namespace pseudo
} // namespace clang
