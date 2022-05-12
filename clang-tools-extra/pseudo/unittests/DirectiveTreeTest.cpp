//===--- DirectiveTreeTest.cpp --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang-pseudo/DirectiveTree.h"

#include "clang-pseudo/Token.h"
#include "clang/Basic/LangOptions.h"
#include "clang/Basic/TokenKinds.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace clang {
namespace pseudo {
namespace {

using testing::_;
using testing::ElementsAre;
using testing::Matcher;
using testing::Pair;
using testing::StrEq;
using Chunk = DirectiveTree::Chunk;

// Matches text of a list of tokens against a string (joined with spaces).
// e.g. EXPECT_THAT(Stream.tokens(), tokens("int main ( ) { }"));
MATCHER_P(tokens, Tokens, "") {
  std::vector<llvm::StringRef> Texts;
  for (const Token &Tok : arg)
    Texts.push_back(Tok.text());
  return Matcher<std::string>(StrEq(Tokens))
      .MatchAndExplain(llvm::join(Texts, " "), result_listener);
}

// Matches tokens covered a directive chunk (with a Tokens property) against a
// string, similar to tokens() above.
// e.g. EXPECT_THAT(SomeDirective, tokensAre(Stream, "# include < vector >"));
MATCHER_P2(tokensAre, TS, Tokens, "tokens are " + std::string(Tokens)) {
  return testing::Matches(tokens(Tokens))(TS.tokens(arg.Tokens));
}

MATCHER_P(chunkKind, K, "") { return arg.kind() == K; }

TEST(DirectiveTree, Parse) {
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
  DirectiveTree PP = DirectiveTree::parse(S);

  ASSERT_THAT(PP.Chunks, ElementsAre(chunkKind(Chunk::K_Directive),
                                     chunkKind(Chunk::K_Code),
                                     chunkKind(Chunk::K_Conditional),
                                     chunkKind(Chunk::K_Code)));

  EXPECT_THAT((const DirectiveTree::Directive &)PP.Chunks[0],
              tokensAre(S, "# include < foo . h >"));
  EXPECT_THAT((const DirectiveTree::Code &)PP.Chunks[1],
              tokensAre(S, "int main ( ) {"));
  EXPECT_THAT((const DirectiveTree::Code &)PP.Chunks[3], tokensAre(S, "}"));

  const DirectiveTree::Conditional &Ifdef(PP.Chunks[2]);
  EXPECT_THAT(Ifdef.Branches,
              ElementsAre(Pair(tokensAre(S, "# ifdef HAS_FOO"), _),
                          Pair(tokensAre(S, "# elif NEEDS_FOO"), _)));
  EXPECT_THAT(Ifdef.End, tokensAre(S, "# endif"));

  const DirectiveTree &HasFoo(Ifdef.Branches[0].second);
  const DirectiveTree &NeedsFoo(Ifdef.Branches[1].second);

  EXPECT_THAT(HasFoo.Chunks, ElementsAre(chunkKind(Chunk::K_Conditional)));
  const DirectiveTree::Conditional &If(HasFoo.Chunks[0]);
  EXPECT_THAT(If.Branches, ElementsAre(Pair(tokensAre(S, "# if HAS_BAR"), _),
                                       Pair(tokensAre(S, "# else"), _)));
  EXPECT_THAT(If.Branches[0].second.Chunks,
              ElementsAre(chunkKind(Chunk::K_Code)));
  EXPECT_THAT(If.Branches[1].second.Chunks,
              ElementsAre(chunkKind(Chunk::K_Code)));

  EXPECT_THAT(NeedsFoo.Chunks, ElementsAre(chunkKind(Chunk::K_Directive)));
  const DirectiveTree::Directive &Error(NeedsFoo.Chunks[0]);
  EXPECT_THAT(Error, tokensAre(S, "# error missing_foo"));
  EXPECT_EQ(Error.Kind, tok::pp_error);
}

TEST(DirectiveTree, ParseUgly) {
  LangOptions Opts;
  std::string Code = R"cpp(
  /*A*/ # /*B*/ \
   /*C*/ \
define \
BAR /*D*/
/*E*/
)cpp";
  TokenStream S = cook(lex(Code, Opts), Opts);
  DirectiveTree PP = DirectiveTree::parse(S);

  ASSERT_THAT(PP.Chunks, ElementsAre(chunkKind(Chunk::K_Code),
                                     chunkKind(Chunk::K_Directive),
                                     chunkKind(Chunk::K_Code)));
  EXPECT_THAT((const DirectiveTree::Code &)PP.Chunks[0], tokensAre(S, "/*A*/"));
  const DirectiveTree::Directive &Define(PP.Chunks[1]);
  EXPECT_EQ(Define.Kind, tok::pp_define);
  EXPECT_THAT(Define, tokensAre(S, "# /*B*/ /*C*/ define BAR /*D*/"));
  EXPECT_THAT((const DirectiveTree::Code &)PP.Chunks[2], tokensAre(S, "/*E*/"));
}

TEST(DirectiveTree, ParseBroken) {
  LangOptions Opts;
  std::string Code = R"cpp(
  a
  #endif // mismatched
  #if X
  b
)cpp";
  TokenStream S = cook(lex(Code, Opts), Opts);
  DirectiveTree PP = DirectiveTree::parse(S);

  ASSERT_THAT(PP.Chunks, ElementsAre(chunkKind(Chunk::K_Code),
                                     chunkKind(Chunk::K_Directive),
                                     chunkKind(Chunk::K_Conditional)));
  EXPECT_THAT((const DirectiveTree::Code &)PP.Chunks[0], tokensAre(S, "a"));
  const DirectiveTree::Directive &Endif(PP.Chunks[1]);
  EXPECT_EQ(Endif.Kind, tok::pp_endif);
  EXPECT_THAT(Endif, tokensAre(S, "# endif // mismatched"));

  const DirectiveTree::Conditional &X(PP.Chunks[2]);
  EXPECT_EQ(1u, X.Branches.size());
  // The (only) branch of the broken conditional section runs until eof.
  EXPECT_EQ(tok::pp_if, X.Branches.front().first.Kind);
  EXPECT_THAT(X.Branches.front().second.Chunks,
              ElementsAre(chunkKind(Chunk::K_Code)));
  // The missing terminating directive is marked as pp_not_keyword.
  EXPECT_EQ(tok::pp_not_keyword, X.End.Kind);
  EXPECT_EQ(0u, X.End.Tokens.size());
}

TEST(DirectiveTree, ChooseBranches) {
  LangOptions Opts;
  const std::string Cases[] = {
      R"cpp(
        // Branches with no alternatives are taken
        #if COND // TAKEN
        int x;
        #endif
      )cpp",

      R"cpp(
        // Empty branches are better than nothing
        #if COND // TAKEN
        #endif
      )cpp",

      R"cpp(
        // Trivially false branches are not taken, even with no alternatives.
        #if 0
        int x;
        #endif
      )cpp",

      R"cpp(
        // Longer branches are preferred over shorter branches
        #if COND // TAKEN
        int x = 1;
        #else
        int x;
        #endif

        #if COND
        int x;
        #else // TAKEN
        int x = 1;
        #endif
      )cpp",

      R"cpp(
        // Trivially true branches are taken if previous branches are trivial.
        #if 1 // TAKEN
        #else
          int x = 1;
        #endif

        #if 0
          int x = 1;
        #elif 0
          int x = 2;
        #elif 1 // TAKEN
          int x;
        #endif

        #if 0
          int x = 1;
        #elif FOO // TAKEN
          int x = 2;
        #elif 1
          int x;
        #endif
      )cpp",

      R"cpp(
        // #else is a trivially true branch
        #if 0
          int x = 1;
        #elif 0
          int x = 2;
        #else // TAKEN
          int x;
        #endif
      )cpp",

      R"cpp(
        // Directives break ties, but nondirective text is more important.
        #if FOO
          #define A 1 2 3
        #else // TAKEN
          #define B 4 5 6
          #define C 7 8 9
        #endif

        #if FOO // TAKEN
          ;
          #define A 1 2 3
        #else
          #define B 4 5 6
          #define C 7 8 9
        #endif
      )cpp",

      R"cpp(
        // Avoid #error directives.
        #if FOO
          int x = 42;
          #error This branch is no good
        #else // TAKEN
        #endif

        #if FOO
          // All paths here lead to errors.
          int x = 42;
          #if 1 // TAKEN
            #if COND // TAKEN
              #error This branch is no good
            #else
              #error This one is no good either
            #endif
          #endif
        #else // TAKEN
        #endif
      )cpp",

      R"cpp(
        // Populate taken branches recursively.
        #if FOO // TAKEN
          int x = 42;
          #if BAR
            ;
          #else // TAKEN
            int y = 43;
          #endif
        #else
          int x;
          #if BAR // TAKEN
            int y;
          #else
            ;
          #endif
        #endif
      )cpp",
  };
  for (const auto &Code : Cases) {
    TokenStream S = cook(lex(Code, Opts), Opts);

    std::function<void(const DirectiveTree &)> Verify =
        [&](const DirectiveTree &M) {
          for (const auto &C : M.Chunks) {
            if (C.kind() != DirectiveTree::Chunk::K_Conditional)
              continue;
            const DirectiveTree::Conditional &Cond(C);
            for (unsigned I = 0; I < Cond.Branches.size(); ++I) {
              auto Directive = S.tokens(Cond.Branches[I].first.Tokens);
              EXPECT_EQ(I == Cond.Taken, Directive.back().text() == "// TAKEN")
                  << "At line " << Directive.front().Line << " of: " << Code;
              Verify(Cond.Branches[I].second);
            }
          }
        };

    DirectiveTree Tree = DirectiveTree::parse(S);
    chooseConditionalBranches(Tree, S);
    Verify(Tree);
  }
}

TEST(DirectiveTree, StripDirectives) {
  LangOptions Opts;
  std::string Code = R"cpp(
    #include <stddef.h>
    a a a
    #warning AAA
    b b b
    #if 1
      c c c
      #warning BBB
      #if 0
        d d d
        #warning CC
      #else
        e e e
      #endif
      f f f
      #if 0
        g g g
      #endif
      h h h
    #else
      i i i
    #endif
    j j j
  )cpp";
  TokenStream S = lex(Code, Opts);

  DirectiveTree Tree = DirectiveTree::parse(S);
  chooseConditionalBranches(Tree, S);
  EXPECT_THAT(Tree.stripDirectives(S).tokens(),
              tokens("a a a b b b c c c e e e f f f h h h j j j"));

  const DirectiveTree &Part =
      ((const DirectiveTree::Conditional &)Tree.Chunks[4]).Branches[0].second;
  EXPECT_THAT(Part.stripDirectives(S).tokens(),
              tokens("c c c e e e f f f h h h"));
}

} // namespace
} // namespace pseudo
} // namespace clang
