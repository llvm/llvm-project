//===--- TokenTest.cpp ----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Tooling/Syntax/Pseudo/Preprocess.h"

#include "clang/Basic/LangOptions.h"
#include "clang/Basic/TokenKinds.h"
#include "clang/Tooling/Syntax/Pseudo/Token.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace clang {
namespace syntax {
namespace pseudo {
namespace {

using testing::_;
using testing::ElementsAre;
using testing::Matcher;
using testing::Pair;
using testing::StrEq;
using Chunk = PPStructure::Chunk;

MATCHER_P2(tokensAre, TS, Tokens, "tokens are " + std::string(Tokens)) {
  std::vector<llvm::StringRef> Texts;
  for (const Token &Tok : TS.tokens(arg.Tokens))
    Texts.push_back(Tok.text());
  return Matcher<std::string>(StrEq(Tokens))
      .MatchAndExplain(llvm::join(Texts, " "), result_listener);
}

MATCHER_P(chunkKind, K, "") { return arg.kind() == K; }

TEST(PPStructure, Parse) {
  LangOptions Opts;
  std::string Code = R"cpp(
  #include <foo.h>

  int main() {
  #ifdef HAS_FOO
  #if HAS_BAR
    foo(bar);
  #else
    foo(0)
  #endif
  #elif NEEDS_FOO
    #error missing_foo
  #endif
  }
  )cpp";

  TokenStream S = cook(lex(Code, Opts), Opts);
  PPStructure PP = PPStructure::parse(S);

  ASSERT_THAT(PP.Chunks, ElementsAre(chunkKind(Chunk::K_Directive),
                                     chunkKind(Chunk::K_Code),
                                     chunkKind(Chunk::K_Conditional),
                                     chunkKind(Chunk::K_Code)));

  EXPECT_THAT((const PPStructure::Directive &)PP.Chunks[0],
              tokensAre(S, "# include < foo . h >"));
  EXPECT_THAT((const PPStructure::Code &)PP.Chunks[1],
              tokensAre(S, "int main ( ) {"));
  EXPECT_THAT((const PPStructure::Code &)PP.Chunks[3], tokensAre(S, "}"));

  const PPStructure::Conditional &Ifdef(PP.Chunks[2]);
  EXPECT_THAT(Ifdef.Branches,
              ElementsAre(Pair(tokensAre(S, "# ifdef HAS_FOO"), _),
                          Pair(tokensAre(S, "# elif NEEDS_FOO"), _)));
  EXPECT_THAT(Ifdef.End, tokensAre(S, "# endif"));

  const PPStructure &HasFoo(Ifdef.Branches[0].second);
  const PPStructure &NeedsFoo(Ifdef.Branches[1].second);

  EXPECT_THAT(HasFoo.Chunks, ElementsAre(chunkKind(Chunk::K_Conditional)));
  const PPStructure::Conditional &If(HasFoo.Chunks[0]);
  EXPECT_THAT(If.Branches, ElementsAre(Pair(tokensAre(S, "# if HAS_BAR"), _),
                                       Pair(tokensAre(S, "# else"), _)));
  EXPECT_THAT(If.Branches[0].second.Chunks,
              ElementsAre(chunkKind(Chunk::K_Code)));
  EXPECT_THAT(If.Branches[1].second.Chunks,
              ElementsAre(chunkKind(Chunk::K_Code)));

  EXPECT_THAT(NeedsFoo.Chunks, ElementsAre(chunkKind(Chunk::K_Directive)));
  const PPStructure::Directive &Error(NeedsFoo.Chunks[0]);
  EXPECT_THAT(Error, tokensAre(S, "# error missing_foo"));
  EXPECT_EQ(Error.Kind, tok::pp_error);
}

TEST(PPStructure, ParseUgly) {
  LangOptions Opts;
  std::string Code = R"cpp(
  /*A*/ # /*B*/ \
   /*C*/ \
define \
BAR /*D*/
/*E*/
)cpp";
  TokenStream S = cook(lex(Code, Opts), Opts);
  PPStructure PP = PPStructure::parse(S);

  ASSERT_THAT(PP.Chunks, ElementsAre(chunkKind(Chunk::K_Code),
                                     chunkKind(Chunk::K_Directive),
                                     chunkKind(Chunk::K_Code)));
  EXPECT_THAT((const PPStructure::Code &)PP.Chunks[0], tokensAre(S, "/*A*/"));
  const PPStructure::Directive &Define(PP.Chunks[1]);
  EXPECT_EQ(Define.Kind, tok::pp_define);
  EXPECT_THAT(Define, tokensAre(S, "# /*B*/ /*C*/ define BAR /*D*/"));
  EXPECT_THAT((const PPStructure::Code &)PP.Chunks[2], tokensAre(S, "/*E*/"));
}

TEST(PPStructure, ParseBroken) {
  LangOptions Opts;
  std::string Code = R"cpp(
  a
  #endif // mismatched
  #if X
  b
)cpp";
  TokenStream S = cook(lex(Code, Opts), Opts);
  PPStructure PP = PPStructure::parse(S);

  ASSERT_THAT(PP.Chunks, ElementsAre(chunkKind(Chunk::K_Code),
                                     chunkKind(Chunk::K_Directive),
                                     chunkKind(Chunk::K_Conditional)));
  EXPECT_THAT((const PPStructure::Code &)PP.Chunks[0], tokensAre(S, "a"));
  const PPStructure::Directive &Endif(PP.Chunks[1]);
  EXPECT_EQ(Endif.Kind, tok::pp_endif);
  EXPECT_THAT(Endif, tokensAre(S, "# endif // mismatched"));

  const PPStructure::Conditional &X(PP.Chunks[2]);
  EXPECT_EQ(1u, X.Branches.size());
  // The (only) branch of the broken conditional section runs until eof.
  EXPECT_EQ(tok::pp_if, X.Branches.front().first.Kind);
  EXPECT_THAT(X.Branches.front().second.Chunks,
              ElementsAre(chunkKind(Chunk::K_Code)));
  // The missing terminating directive is marked as pp_not_keyword.
  EXPECT_EQ(tok::pp_not_keyword, X.End.Kind);
  EXPECT_EQ(0u, X.End.Tokens.size());
}

} // namespace
} // namespace pseudo
} // namespace syntax
} // namespace clang
