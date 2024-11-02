//===---- ModernizeModuleTest.cpp - clang-tidy ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "ClangTidyTest.h"
#include "modernize/IntegralLiteralExpressionMatcher.h"
#include "clang/Lex/Lexer.h"
#include "gtest/gtest.h"

#include <cstring>
#include <iterator>
#include <string>
#include <vector>

namespace clang {
namespace tidy {
namespace test {

static std::vector<Token> tokenify(const char *Text) {
  LangOptions LangOpts;
  std::vector<std::string> Includes;
  LangOptions::setLangDefaults(LangOpts, Language::CXX, llvm::Triple(),
                               Includes, LangStandard::lang_cxx20);
  Lexer Lex(SourceLocation{}, LangOpts, Text, Text, Text + std::strlen(Text));
  std::vector<Token> Tokens;
  bool End = false;
  while (!End) {
    Token Tok;
    End = Lex.LexFromRawLexer(Tok);
    Tokens.push_back(Tok);
  }

  return Tokens;
}

static bool matchText(const char *Text, bool AllowComma) {
  std::vector<Token> Tokens{tokenify(Text)};
  modernize::IntegralLiteralExpressionMatcher Matcher(Tokens, AllowComma);

  return Matcher.match();
}

static modernize::LiteralSize sizeText(const char *Text) {
  std::vector<Token> Tokens{tokenify(Text)};
  modernize::IntegralLiteralExpressionMatcher Matcher(Tokens, true);
  if (Matcher.match())
    return Matcher.largestLiteralSize();
  return modernize::LiteralSize::Unknown;
}

static const char *toString(modernize::LiteralSize Value) {
  switch (Value) {
  case modernize::LiteralSize::Int:
    return "Int";
  case modernize::LiteralSize::UnsignedInt:
    return "UnsignedInt";
  case modernize::LiteralSize::Long:
    return "Long";
  case modernize::LiteralSize::UnsignedLong:
    return "UnsignedLong";
  case modernize::LiteralSize::LongLong:
    return "LongLong";
  case modernize::LiteralSize::UnsignedLongLong:
    return "UnsignedLongLong";
  default:
    return "Unknown";
  }
}

namespace {

struct MatchParam {
  bool AllowComma;
  bool Matched;
  const char *Text;

  friend std::ostream &operator<<(std::ostream &Str, const MatchParam &Value) {
    return Str << "Allow operator,: " << std::boolalpha << Value.AllowComma
               << ", Matched: " << std::boolalpha << Value.Matched
               << ", Text: '" << Value.Text << '\'';
  }
};

struct SizeParam {
  modernize::LiteralSize Size;
  const char *Text;

  friend std::ostream &operator<<(std::ostream &Str, const SizeParam &Value) {
    return Str << "Size: " << toString(Value.Size) << ", Text: '" << Value.Text << '\'';
  }
};

class MatcherTest : public ::testing::TestWithParam<MatchParam> {};

class SizeTest : public ::testing::TestWithParam<SizeParam> {};

} // namespace

static const MatchParam MatchParams[] = {
    // Accept integral literals.
    {true, true, "1"},
    {true, true, "0177"},
    {true, true, "0xdeadbeef"},
    {true, true, "0b1011"},
    {true, true, "'c'"},
    // Reject non-integral literals.
    {true, false, "1.23"},
    {true, false, "0x1p3"},
    {true, false, R"("string")"},
    {true, false, "1i"},

    // Accept literals with these unary operators.
    {true, true, "-1"},
    {true, true, "+1"},
    {true, true, "~1"},
    {true, true, "!1"},
    // Reject invalid unary operators.
    {true, false, "1-"},
    {true, false, "1+"},
    {true, false, "1~"},
    {true, false, "1!"},

    // Accept valid binary operators.
    {true, true, "1+1"},
    {true, true, "1-1"},
    {true, true, "1*1"},
    {true, true, "1/1"},
    {true, true, "1%2"},
    {true, true, "1<<1"},
    {true, true, "1>>1"},
    {true, true, "1<=>1"},
    {true, true, "1<1"},
    {true, true, "1>1"},
    {true, true, "1<=1"},
    {true, true, "1>=1"},
    {true, true, "1==1"},
    {true, true, "1!=1"},
    {true, true, "1&1"},
    {true, true, "1^1"},
    {true, true, "1|1"},
    {true, true, "1&&1"},
    {true, true, "1||1"},
    {true, true, "1+ +1"}, // A space is needed to avoid being tokenized as ++ or --.
    {true, true, "1- -1"},
    // Comma is only valid when inside parentheses.
    {true, true, "(1,1)"},
    // Reject invalid binary operators.
    {true, false, "1+"},
    {true, false, "1-"},
    {true, false, "1*"},
    {true, false, "1/"},
    {true, false, "1%"},
    {true, false, "1<<"},
    {true, false, "1>>"},
    {true, false, "1<=>"},
    {true, false, "1<"},
    {true, false, "1>"},
    {true, false, "1<="},
    {true, false, "1>="},
    {true, false, "1=="},
    {true, false, "1!="},
    {true, false, "1&"},
    {true, false, "1^"},
    {true, false, "1|"},
    {true, false, "1&&"},
    {true, false, "1||"},
    {true, false, "1,"},
    {true, false, ",1"},
    {true, false, "1,1"},

    // Accept valid ternary operators.
    {true, true, "1?1:1"},
    {true, true, "1?:1"}, // A gcc extension treats x ? : y as x ? x : y.
    // Reject invalid ternary operators.
    {true, false, "?"},
    {true, false, "?1"},
    {true, false, "?:"},
    {true, false, "?:1"},
    {true, false, "?1:"},
    {true, false, "?1:1"},
    {true, false, "1?"},
    {true, false, "1?1"},
    {true, false, "1?:"},
    {true, false, "1?1:"},

    // Accept parenthesized expressions.
    {true, true, "(1)"},
    {true, true, "((+1))"},
    {true, true, "((+(1)))"},
    {true, true, "(-1)"},
    {true, true, "-(1)"},
    {true, true, "(+1)"},
    {true, true, "((+1))"},
    {true, true, "+(1)"},
    {true, true, "(~1)"},
    {true, true, "~(1)"},
    {true, true, "(!1)"},
    {true, true, "!(1)"},
    {true, true, "(1+1)"},
    {true, true, "(1-1)"},
    {true, true, "(1*1)"},
    {true, true, "(1/1)"},
    {true, true, "(1%2)"},
    {true, true, "(1<<1)"},
    {true, true, "(1>>1)"},
    {true, true, "(1<=>1)"},
    {true, true, "(1<1)"},
    {true, true, "(1>1)"},
    {true, true, "(1<=1)"},
    {true, true, "(1>=1)"},
    {true, true, "(1==1)"},
    {true, true, "(1!=1)"},
    {true, true, "(1&1)"},
    {true, true, "(1^1)"},
    {true, true, "(1|1)"},
    {true, true, "(1&&1)"},
    {true, true, "(1||1)"},
    {true, true, "(1?1:1)"},

    // Accept more complicated "chained" expressions.
    {true, true, "1+1+1"},
    {true, true, "1+1+1+1"},
    {true, true, "1+1+1+1+1"},
    {true, true, "1*1*1"},
    {true, true, "1*1*1*1"},
    {true, true, "1*1*1*1*1"},
    {true, true, "1<<1<<1"},
    {true, true, "4U>>1>>1"},
    {true, true, "1<1<1"},
    {true, true, "1>1>1"},
    {true, true, "1<=1<=1"},
    {true, true, "1>=1>=1"},
    {true, true, "1==1==1"},
    {true, true, "1!=1!=1"},
    {true, true, "1&1&1"},
    {true, true, "1^1^1"},
    {true, true, "1|1|1"},
    {true, true, "1&&1&&1"},
    {true, true, "1||1||1"},
    {true, true, "(1,1,1)"},

    // Optionally reject comma operator
    {false, false, "1,1"}
};

TEST_P(MatcherTest, MatchResult) {
  const MatchParam &Param = GetParam();
 
  EXPECT_TRUE(matchText(Param.Text, Param.AllowComma) == Param.Matched);
}

INSTANTIATE_TEST_SUITE_P(IntegralLiteralExpressionMatcherTests, MatcherTest,
                         ::testing::ValuesIn(MatchParams));

static const SizeParam SizeParams[] = {
    {modernize::LiteralSize::Int, "1"},
    {modernize::LiteralSize::UnsignedInt, "1U"},
    {modernize::LiteralSize::Long, "1L"},
    {modernize::LiteralSize::UnsignedLong, "1UL"},
    {modernize::LiteralSize::UnsignedLong, "1LU"},
    {modernize::LiteralSize::LongLong, "1LL"},
    {modernize::LiteralSize::UnsignedLongLong, "1ULL"},
    {modernize::LiteralSize::UnsignedLongLong, "1LLU"}};

TEST_P(SizeTest, TokenSize) {
  EXPECT_EQ(sizeText(GetParam().Text), GetParam().Size);
};

INSTANTIATE_TEST_SUITE_P(IntegralLiteralExpressionMatcherTests, SizeTest,
                         ::testing::ValuesIn(SizeParams));

} // namespace test
} // namespace tidy
} // namespace clang
