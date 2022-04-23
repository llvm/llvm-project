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

static bool matchText(const char *Text) {
  std::vector<Token> Tokens{tokenify(Text)};
  modernize::IntegralLiteralExpressionMatcher Matcher(Tokens);

  return Matcher.match();
}

namespace {

struct Param {
  bool Matched;
  const char *Text;
};

class MatcherTest : public ::testing::TestWithParam<Param> {};

} // namespace

static const Param Params[] = {
    // Accept integral literals.
    {true, "1"},
    {true, "0177"},
    {true, "0xdeadbeef"},
    {true, "0b1011"},
    {true, "'c'"},
    // Reject non-integral literals.
    {false, "1.23"},
    {false, "0x1p3"},
    {false, R"("string")"},
    {false, "1i"},

    // Accept literals with these unary operators.
    {true, "-1"},
    {true, "+1"},
    {true, "~1"},
    {true, "!1"},
    // Reject invalid unary operators.
    {false, "1-"},
    {false, "1+"},
    {false, "1~"},
    {false, "1!"},

    // Accept valid binary operators.
    {true, "1+1"},
    {true, "1-1"},
    {true, "1*1"},
    {true, "1/1"},
    {true, "1%2"},
    {true, "1<<1"},
    {true, "1>>1"},
    {true, "1<=>1"},
    {true, "1<1"},
    {true, "1>1"},
    {true, "1<=1"},
    {true, "1>=1"},
    {true, "1==1"},
    {true, "1!=1"},
    {true, "1&1"},
    {true, "1^1"},
    {true, "1|1"},
    {true, "1&&1"},
    {true, "1||1"},
    {true, "1+ +1"}, // A space is needed to avoid being tokenized as ++ or --.
    {true, "1- -1"},
    {true, "1,1"},
    // Reject invalid binary operators.
    {false, "1+"},
    {false, "1-"},
    {false, "1*"},
    {false, "1/"},
    {false, "1%"},
    {false, "1<<"},
    {false, "1>>"},
    {false, "1<=>"},
    {false, "1<"},
    {false, "1>"},
    {false, "1<="},
    {false, "1>="},
    {false, "1=="},
    {false, "1!="},
    {false, "1&"},
    {false, "1^"},
    {false, "1|"},
    {false, "1&&"},
    {false, "1||"},
    {false, "1,"},
    {false, ",1"},

    // Accept valid ternary operators.
    {true, "1?1:1"},
    {true, "1?:1"}, // A gcc extension treats x ? : y as x ? x : y.
    // Reject invalid ternary operators.
    {false, "?"},
    {false, "?1"},
    {false, "?:"},
    {false, "?:1"},
    {false, "?1:"},
    {false, "?1:1"},
    {false, "1?"},
    {false, "1?1"},
    {false, "1?:"},
    {false, "1?1:"},

    // Accept parenthesized expressions.
    {true, "(1)"},
    {true, "((+1))"},
    {true, "((+(1)))"},
    {true, "(-1)"},
    {true, "-(1)"},
    {true, "(+1)"},
    {true, "((+1))"},
    {true, "+(1)"},
    {true, "(~1)"},
    {true, "~(1)"},
    {true, "(!1)"},
    {true, "!(1)"},
    {true, "(1+1)"},
    {true, "(1-1)"},
    {true, "(1*1)"},
    {true, "(1/1)"},
    {true, "(1%2)"},
    {true, "(1<<1)"},
    {true, "(1>>1)"},
    {true, "(1<=>1)"},
    {true, "(1<1)"},
    {true, "(1>1)"},
    {true, "(1<=1)"},
    {true, "(1>=1)"},
    {true, "(1==1)"},
    {true, "(1!=1)"},
    {true, "(1&1)"},
    {true, "(1^1)"},
    {true, "(1|1)"},
    {true, "(1&&1)"},
    {true, "(1||1)"},
    {true, "(1?1:1)"},

    // Accept more complicated "chained" expressions.
    {true, "1+1+1"},
    {true, "1+1+1+1"},
    {true, "1+1+1+1+1"},
    {true, "1*1*1"},
    {true, "1*1*1*1"},
    {true, "1*1*1*1*1"},
    {true, "1<<1<<1"},
    {true, "4U>>1>>1"},
    {true, "1<1<1"},
    {true, "1>1>1"},
    {true, "1<=1<=1"},
    {true, "1>=1>=1"},
    {true, "1==1==1"},
    {true, "1!=1!=1"},
    {true, "1&1&1"},
    {true, "1^1^1"},
    {true, "1|1|1"},
    {true, "1&&1&&1"},
    {true, "1||1||1"},
    {true, "1,1,1"},
};

TEST_P(MatcherTest, MatchResult) {
  EXPECT_TRUE(matchText(GetParam().Text) == GetParam().Matched);
}

INSTANTIATE_TEST_SUITE_P(TokenExpressionParserTests, MatcherTest,
                         ::testing::ValuesIn(Params));

} // namespace test
} // namespace tidy
} // namespace clang

std::ostream &operator<<(std::ostream &Str,
                         const clang::tidy::test::Param &Value) {
  return Str << "Matched: " << std::boolalpha << Value.Matched << ", Text: '"
             << Value.Text << "'";
}
