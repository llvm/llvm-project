//===--- TokenTest.cpp ----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang-pseudo/Token.h"
#include "clang/Basic/LangOptions.h"
#include "clang/Basic/TokenKinds.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace clang {
namespace pseudo {
namespace {

using testing::AllOf;
using testing::ElementsAre;
using testing::ElementsAreArray;
using testing::Not;

MATCHER_P2(token, Text, Kind, "") {
  return arg.Kind == Kind && arg.text() == Text;
}

MATCHER_P(hasFlag, Flag, "") { return arg.flag(Flag); }

MATCHER_P2(lineIndent, Line, Indent, "") {
  return arg.Line == (unsigned)Line && arg.Indent == (unsigned)Indent;
}

MATCHER_P(originalIndex, index, "") {
  return arg.OriginalIndex == (Token::Index)index;
}

TEST(TokenTest, Lex) {
  LangOptions Opts;
  std::string Code = R"cpp(
    #include <stdio.h>
    int main() {
      return 42; // the answer
    }
  )cpp";
  TokenStream Raw = lex(Code, Opts);
  ASSERT_TRUE(Raw.isFinalized());
  EXPECT_THAT(Raw.tokens(),
              ElementsAreArray({
                  // Lexing of directives is weird, especially <angled> strings.
                  token("#", tok::hash),
                  token("include", tok::raw_identifier),
                  token("<", tok::less),
                  token("stdio", tok::raw_identifier),
                  token(".", tok::period),
                  token("h", tok::raw_identifier),
                  token(">", tok::greater),

                  token("int", tok::raw_identifier),
                  token("main", tok::raw_identifier),
                  token("(", tok::l_paren),
                  token(")", tok::r_paren),
                  token("{", tok::l_brace),
                  token("return", tok::raw_identifier),
                  token("42", tok::numeric_constant),
                  token(";", tok::semi),
                  token("// the answer", tok::comment),
                  token("}", tok::r_brace),
              }));

  TokenStream Cooked = cook(Raw, Opts);
  ASSERT_TRUE(Cooked.isFinalized());
  EXPECT_THAT(Cooked.tokens(),
              ElementsAreArray({
                  // Cooked identifier types in directives are not meaningful.
                  token("#", tok::hash),
                  token("include", tok::identifier),
                  token("<", tok::less),
                  token("stdio", tok::identifier),
                  token(".", tok::period),
                  token("h", tok::identifier),
                  token(">", tok::greater),

                  token("int", tok::kw_int),
                  token("main", tok::identifier),
                  token("(", tok::l_paren),
                  token(")", tok::r_paren),
                  token("{", tok::l_brace),
                  token("return", tok::kw_return),
                  token("42", tok::numeric_constant),
                  token(";", tok::semi),
                  token("// the answer", tok::comment),
                  token("}", tok::r_brace),
              }));
  // Check raw tokens point back into original source code.
  EXPECT_EQ(Raw.tokens().front().text().begin(), &Code[Code.find('#')]);
}

TEST(TokenTest, LineContinuation) {
  LangOptions Opts;
  std::string Code = R"cpp(
one_\
token
two \
tokens
  )cpp";
  TokenStream Raw = lex(Code, Opts);
  EXPECT_THAT(
      Raw.tokens(),
      ElementsAre(AllOf(token("one_\\\ntoken", tok::raw_identifier),
                        hasFlag(LexFlags::StartsPPLine),
                        hasFlag(LexFlags::NeedsCleaning), lineIndent(1, 0),
                        originalIndex(0)),
                  AllOf(token("two", tok::raw_identifier),
                        hasFlag(LexFlags::StartsPPLine),
                        Not(hasFlag(LexFlags::NeedsCleaning)),
                        originalIndex(1)),
                  AllOf(token("\\\ntokens", tok::raw_identifier),
                        Not(hasFlag(LexFlags::StartsPPLine)),
                        hasFlag(LexFlags::NeedsCleaning), originalIndex(2))));

  TokenStream Cooked = cook(Raw, Opts);
  EXPECT_THAT(
      Cooked.tokens(),
      ElementsAre(AllOf(token("one_token", tok::identifier), lineIndent(1, 0),
                        originalIndex(0)),
                  AllOf(token("two", tok::identifier), originalIndex(1)),
                  AllOf(token("tokens", tok::identifier), originalIndex(2))));
}

TEST(TokenTest, EncodedCharacters) {
  LangOptions Opts;
  Opts.Trigraphs = true;
  Opts.Digraphs = true;
  Opts.C99 = true; // UCNs
  Opts.CXXOperatorNames = true;
  std::string Code = R"(and <: ??! '??=' \u00E9)";
  TokenStream Raw = lex(Code, Opts);
  EXPECT_THAT(
      Raw.tokens(),
      ElementsAre( // and is not recognized as && until cook().
          AllOf(token("and", tok::raw_identifier),
                Not(hasFlag(LexFlags::NeedsCleaning))),
          // Digraphs are just different spellings of tokens.
          AllOf(token("<:", tok::l_square),
                Not(hasFlag(LexFlags::NeedsCleaning))),
          // Trigraps are interpreted, still need text cleaning.
          AllOf(token(R"(??!)", tok::pipe), hasFlag(LexFlags::NeedsCleaning)),
          // Trigraphs must be substituted inside constants too.
          AllOf(token(R"('??=')", tok::char_constant),
                hasFlag(LexFlags::NeedsCleaning)),
          // UCNs need substitution.
          AllOf(token(R"(\u00E9)", tok::raw_identifier),
                hasFlag(LexFlags::NeedsCleaning))));

  TokenStream Cooked = cook(Raw, Opts);
  EXPECT_THAT(
      Cooked.tokens(),
      ElementsAre(token("and", tok::ampamp), // alternate spelling recognized
                  token("<:", tok::l_square),
                  token("|", tok::pipe),            // trigraph substituted
                  token("'#'", tok::char_constant), // trigraph substituted
                  token("é", tok::identifier)));    // UCN substituted
}

TEST(TokenTest, Indentation) {
  LangOptions Opts;
  std::string Code = R"cpp(   hello world
no_indent \
  line_was_continued
)cpp";
  TokenStream Raw = lex(Code, Opts);
  EXPECT_THAT(Raw.tokens(), ElementsAreArray({
                                lineIndent(0, 3), // hello
                                lineIndent(0, 3), // world
                                lineIndent(1, 0), // no_indent
                                lineIndent(2, 2), // line_was_continued
                            }));
}

TEST(TokenTest, SplitGreaterGreater) {
  LangOptions Opts;
  std::string Code = R"cpp(
>> // split
// >> with an escaped newline in the middle, split
>\
>
>>= // not split
)cpp";
  TokenStream Cook = cook(lex(Code, Opts), Opts);
  TokenStream Split = stripComments(Cook);
  EXPECT_THAT(Split.tokens(),
              ElementsAre(AllOf(token(">", tok::greater), originalIndex(0)),
                          AllOf(token(">", tok::greater), originalIndex(0)),
                          // Token 1 and 2 are comments.
                          AllOf(token(">", tok::greater), originalIndex(3)),
                          AllOf(token(">", tok::greater), originalIndex(3)),
                          AllOf(token(">>=", tok::greatergreaterequal),
                                originalIndex(4))));
}

TEST(TokenTest, DropComments) {
  LangOptions Opts;
  std::string Code = R"cpp(
  // comment
  int /*abc*/;
)cpp";
  TokenStream Raw = cook(lex(Code, Opts), Opts);
  TokenStream Stripped = stripComments(Raw);
  EXPECT_THAT(
      Raw.tokens(),
      ElementsAre(AllOf(token("// comment", tok::comment), originalIndex(0)),
                  AllOf(token("int", tok::kw_int), originalIndex(1)),
                  AllOf(token("/*abc*/", tok::comment), originalIndex(2)),
                  AllOf(token(";", tok::semi), originalIndex(3))));

  EXPECT_THAT(Stripped.tokens(),
              ElementsAre(AllOf(token("int", tok::kw_int), originalIndex(1)),
                          AllOf(token(";", tok::semi), originalIndex(3))));
}

} // namespace
} // namespace pseudo
} // namespace clang
