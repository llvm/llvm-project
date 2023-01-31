//===- unittest/Format/FormatTokenSourceTest.cpp --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../../lib/Format/FormatTokenSource.h"
#include "TestLexer.h"
#include "clang/Basic/TokenKinds.h"
#include "gtest/gtest.h"

namespace clang {
namespace format {
namespace {

class IndexedTokenSourceTest : public ::testing::Test {
protected:
  TokenList lex(llvm::StringRef Code,
                const FormatStyle &Style = getLLVMStyle()) {
    return TestLexer(Allocator, Buffers, Style).lex(Code);
  }
  llvm::SpecificBumpPtrAllocator<FormatToken> Allocator;
  std::vector<std::unique_ptr<llvm::MemoryBuffer>> Buffers;
};

#define EXPECT_TOKEN_KIND(FormatTok, Kind)                                     \
  do {                                                                         \
    FormatToken *Tok = FormatTok;                                              \
    EXPECT_EQ((Tok)->Tok.getKind(), Kind) << *(Tok);                           \
  } while (false);

TEST_F(IndexedTokenSourceTest, EmptyInput) {
  TokenList Tokens = lex("");
  IndexedTokenSource Source(Tokens);
  EXPECT_FALSE(Source.isEOF());
  EXPECT_TOKEN_KIND(Source.getNextToken(), tok::eof);
  EXPECT_TRUE(Source.isEOF());
  EXPECT_TOKEN_KIND(Source.getNextToken(), tok::eof);
  EXPECT_TRUE(Source.isEOF());
  EXPECT_TOKEN_KIND(Source.peekNextToken(/*SkipComment=*/false), tok::eof);
  EXPECT_TOKEN_KIND(Source.peekNextToken(/*SkipComment=*/true), tok::eof);
  EXPECT_EQ(Source.getPreviousToken(), nullptr);
  EXPECT_TRUE(Source.isEOF());
}

TEST_F(IndexedTokenSourceTest, NavigateTokenStream) {
  TokenList Tokens = lex("int a;");
  IndexedTokenSource Source(Tokens);
  EXPECT_TOKEN_KIND(Source.peekNextToken(), tok::kw_int);
  EXPECT_TOKEN_KIND(Source.getNextToken(), tok::kw_int);
  EXPECT_EQ(Source.getPreviousToken(), nullptr);
  EXPECT_TOKEN_KIND(Source.peekNextToken(), tok::identifier);
  EXPECT_TOKEN_KIND(Source.getNextToken(), tok::identifier);
  EXPECT_TOKEN_KIND(Source.getPreviousToken(), tok::kw_int);
  EXPECT_TOKEN_KIND(Source.peekNextToken(), tok::semi);
  EXPECT_TOKEN_KIND(Source.getNextToken(), tok::semi);
  EXPECT_TOKEN_KIND(Source.getPreviousToken(), tok::identifier);
  EXPECT_TOKEN_KIND(Source.peekNextToken(), tok::eof);
  EXPECT_TOKEN_KIND(Source.getNextToken(), tok::eof);
  EXPECT_TOKEN_KIND(Source.getPreviousToken(), tok::semi);
}

TEST_F(IndexedTokenSourceTest, ResetPosition) {
  TokenList Tokens = lex("int a;");
  IndexedTokenSource Source(Tokens);
  Source.getNextToken();
  unsigned Position = Source.getPosition();
  Source.getNextToken();
  Source.getNextToken();
  EXPECT_TOKEN_KIND(Source.getNextToken(), tok::eof);
  EXPECT_TOKEN_KIND(Source.setPosition(Position), tok::kw_int);
}

} // namespace
} // namespace format
} // namespace clang
