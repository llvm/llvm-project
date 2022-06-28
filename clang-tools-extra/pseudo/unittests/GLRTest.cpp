//===--- GLRTest.cpp - Test the GLR parser ----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang-pseudo/GLR.h"
#include "clang-pseudo/Bracket.h"
#include "clang-pseudo/Token.h"
#include "clang-pseudo/grammar/Grammar.h"
#include "clang/Basic/LangOptions.h"
#include "clang/Basic/TokenKinds.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/FormatVariadic.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <memory>

namespace clang {
namespace pseudo {

llvm::raw_ostream &operator<<(llvm::raw_ostream &OS,
                              const std::vector<const GSS::Node *> &Heads) {
  for (const auto *Head : Heads)
    OS << *Head << "\n";
  return OS;
}

namespace {

using Action = LRTable::Action;
using testing::AllOf;
using testing::ElementsAre;
using testing::IsEmpty;
using testing::UnorderedElementsAre;

MATCHER_P(state, StateID, "") { return arg->State == StateID; }
MATCHER_P(parsedSymbol, FNode, "") { return arg->Payload == FNode; }
MATCHER_P(parsedSymbolID, SID, "") { return arg->Payload->symbol() == SID; }
MATCHER_P(start, Start, "") { return arg->Payload->startTokenIndex() == Start; }

testing::Matcher<const GSS::Node *>
parents(llvm::ArrayRef<const GSS::Node *> Parents) {
  return testing::Property(&GSS::Node::parents,
                           testing::UnorderedElementsAreArray(Parents));
}

class GLRTest : public ::testing::Test {
public:
  void build(llvm::StringRef GrammarBNF) {
    std::vector<std::string> Diags;
    G = Grammar::parseBNF(GrammarBNF, Diags);
  }

  void buildGrammar(std::vector<std::string> Nonterminals,
                    std::vector<std::string> Rules) {
    Nonterminals.push_back("_");
    llvm::sort(Nonterminals);
    Nonterminals.erase(std::unique(Nonterminals.begin(), Nonterminals.end()),
                       Nonterminals.end());
    std::string FakeTestBNF;
    for (const auto &NT : Nonterminals)
      FakeTestBNF += llvm::formatv("{0} := {1}\n", "_", NT);
    FakeTestBNF += llvm::join(Rules, "\n");
    build(FakeTestBNF);
  }

  SymbolID id(llvm::StringRef Name) const {
    for (unsigned I = 0; I < NumTerminals; ++I)
      if (G.table().Terminals[I] == Name)
        return tokenSymbol(static_cast<tok::TokenKind>(I));
    for (SymbolID ID = 0; ID < G.table().Nonterminals.size(); ++ID)
      if (G.table().Nonterminals[ID].Name == Name)
        return ID;
    ADD_FAILURE() << "No such symbol found: " << Name;
    return 0;
  }

  RuleID ruleFor(llvm::StringRef NonterminalName) const {
    auto RuleRange = G.table().Nonterminals[id(NonterminalName)].RuleRange;
    if (RuleRange.End - RuleRange.Start == 1)
      return G.table().Nonterminals[id(NonterminalName)].RuleRange.Start;
    ADD_FAILURE() << "Expected a single rule for " << NonterminalName
                  << ", but it has " << RuleRange.End - RuleRange.Start
                  << " rule!\n";
    return 0;
  }

protected:
  Grammar G;
  ForestArena Arena;
  GSS GSStack;
};

TEST_F(GLRTest, ShiftMergingHeads) {
  // Given a test case where we have two heads 1, 2, 3 in the GSS, the heads 1,
  // 2 have shift actions to reach state 4, and the head 3 has a shift action to
  // reach state 5:
  //   0--1
  //   └--2
  //   └--3
  // After the shift action, the GSS (with new heads 4, 5) is:
  //   0---1---4
  //   └---2---┘
  //   └---3---5
  auto *GSSNode0 =
      GSStack.addNode(/*State=*/0, /*ForestNode=*/nullptr, /*Parents=*/{});
  auto *GSSNode1 = GSStack.addNode(/*State=*/1, /*ForestNode=*/nullptr,
                                   /*Parents=*/{GSSNode0});
  auto *GSSNode2 = GSStack.addNode(/*State=*/2, /*ForestNode=*/nullptr,
                                   /*Parents=*/{GSSNode0});
  auto *GSSNode3 = GSStack.addNode(/*State=*/3, /*ForestNode=*/nullptr,
                                   /*Parents=*/{GSSNode0});

  buildGrammar({}, {}); // Create a fake empty grammar.
  LRTable T =
      LRTable::buildForTests(G, /*Entries=*/
                             {
                                 {1, tokenSymbol(tok::semi), Action::shift(4)},
                                 {2, tokenSymbol(tok::semi), Action::shift(4)},
                                 {3, tokenSymbol(tok::semi), Action::shift(5)},
                             },
                             {});

  ForestNode &SemiTerminal = Arena.createTerminal(tok::semi, 0);
  std::vector<const GSS::Node *> NewHeads;
  glrShift({GSSNode1, GSSNode2, GSSNode3}, SemiTerminal, {G, T, Arena, GSStack},
           NewHeads);

  EXPECT_THAT(NewHeads,
              UnorderedElementsAre(AllOf(state(4), parsedSymbol(&SemiTerminal),
                                         parents({GSSNode1, GSSNode2})),
                                   AllOf(state(5), parsedSymbol(&SemiTerminal),
                                         parents({GSSNode3}))))
      << NewHeads;
}

TEST_F(GLRTest, ReduceConflictsSplitting) {
  //  Before (splitting due to R/R conflict):
  //    0--1(IDENTIFIER)
  //  After reducing 1 by `class-name := IDENTIFIER` and
  //                      `enum-name := IDENTIFIER`:
  //    0--2(class-name)    // 2 is goto(0, class-name)
  //    └--3(enum-name)     // 3 is goto(0, enum-name)
  buildGrammar({"class-name", "enum-name"},
               {"class-name := IDENTIFIER", "enum-name := IDENTIFIER"});

  LRTable Table = LRTable::buildForTests(
      G,
      {
          {/*State=*/0, id("class-name"), Action::goTo(2)},
          {/*State=*/0, id("enum-name"), Action::goTo(3)},
      },
      {
          {/*State=*/1, ruleFor("class-name")},
          {/*State=*/1, ruleFor("enum-name")},
      });

  const auto *GSSNode0 =
      GSStack.addNode(/*State=*/0, /*ForestNode=*/nullptr, /*Parents=*/{});
  const auto *GSSNode1 =
      GSStack.addNode(1, &Arena.createTerminal(tok::identifier, 0), {GSSNode0});

  std::vector<const GSS::Node *> Heads = {GSSNode1};
  glrReduce(Heads, tokenSymbol(tok::eof), {G, Table, Arena, GSStack});
  EXPECT_THAT(Heads, UnorderedElementsAre(
                         GSSNode1,
                         AllOf(state(2), parsedSymbolID(id("class-name")),
                               parents({GSSNode0})),
                         AllOf(state(3), parsedSymbolID(id("enum-name")),
                               parents({GSSNode0}))))
      << Heads;
}

TEST_F(GLRTest, ReduceSplittingDueToMultipleBases) {
  //  Before (splitting due to multiple bases):
  //    2(class-name)--4(*)
  //    3(enum-name)---┘
  //  After reducing 4 by `ptr-operator := *`:
  //    2(class-name)--5(ptr-operator)    // 5 is goto(2, ptr-operator)
  //    3(enum-name)---6(ptr-operator)    // 6 is goto(3, ptr-operator)
  buildGrammar({"ptr-operator", "class-name", "enum-name"},
               {"ptr-operator := *"});

  auto *ClassNameNode = &Arena.createOpaque(id("class-name"), /*TokenIndex=*/0);
  auto *EnumNameNode = &Arena.createOpaque(id("enum-name"), /*TokenIndex=*/0);

  const auto *GSSNode2 =
      GSStack.addNode(/*State=*/2, /*ForestNode=*/ClassNameNode, /*Parents=*/{});
  const auto *GSSNode3 =
      GSStack.addNode(/*State=*/3, /*ForestNode=*/EnumNameNode, /*Parents=*/{});
  const auto *GSSNode4 = GSStack.addNode(
      /*State=*/4, &Arena.createTerminal(tok::star, /*TokenIndex=*/1),
      /*Parents=*/{GSSNode2, GSSNode3});

  LRTable Table = LRTable::buildForTests(
      G,
      {
          {/*State=*/2, id("ptr-operator"), Action::goTo(/*NextState=*/5)},
          {/*State=*/3, id("ptr-operator"), Action::goTo(/*NextState=*/6)},
      },
      {
          {/*State=*/4, ruleFor("ptr-operator")},
      });
  std::vector<const GSS::Node *> Heads = {GSSNode4};
  glrReduce(Heads, tokenSymbol(tok::eof), {G, Table, Arena, GSStack});

  EXPECT_THAT(Heads, UnorderedElementsAre(
                         GSSNode4,
                         AllOf(state(5), parsedSymbolID(id("ptr-operator")),
                               parents({GSSNode2})),
                         AllOf(state(6), parsedSymbolID(id("ptr-operator")),
                               parents({GSSNode3}))))
      << Heads;
  // Verify that the payload of the two new heads is shared, only a single
  // ptr-operator node is created in the forest.
  EXPECT_EQ(Heads[1]->Payload, Heads[2]->Payload);
}

TEST_F(GLRTest, ReduceJoiningWithMultipleBases) {
  //  Before (joining due to same goto state, multiple bases):
  //    0--1(cv-qualifier)--3(class-name)
  //    └--2(cv-qualifier)--4(enum-name)
  //  After reducing 3 by `type-name := class-name` and
  //                 4 by `type-name := enum-name`:
  //    0--1(cv-qualifier)--5(type-name)  // 5 is goto(1, type-name) and
  //    └--2(cv-qualifier)--┘             // goto(2, type-name)
  buildGrammar({"type-name", "class-name", "enum-name", "cv-qualifier"},
               {"type-name := class-name", "type-name := enum-name"});

  auto *CVQualifierNode =
      &Arena.createOpaque(id("cv-qualifier"), /*TokenIndex=*/0);
  auto *ClassNameNode = &Arena.createOpaque(id("class-name"), /*TokenIndex=*/1);
  auto *EnumNameNode = &Arena.createOpaque(id("enum-name"), /*TokenIndex=*/1);

  const auto *GSSNode0 =
      GSStack.addNode(/*State=*/0, /*ForestNode=*/nullptr, /*Parents=*/{});
  const auto *GSSNode1 = GSStack.addNode(
      /*State=*/1, /*ForestNode=*/CVQualifierNode, /*Parents=*/{GSSNode0});
  const auto *GSSNode2 = GSStack.addNode(
      /*State=*/2, /*ForestNode=*/CVQualifierNode, /*Parents=*/{GSSNode0});
  const auto *GSSNode3 = GSStack.addNode(
      /*State=*/3, /*ForestNode=*/ClassNameNode,
      /*Parents=*/{GSSNode1});
  const auto *GSSNode4 =
      GSStack.addNode(/*State=*/4, /*ForestNode=*/EnumNameNode,
                      /*Parents=*/{GSSNode2});

  // FIXME: figure out a way to get rid of the hard-coded reduce RuleID!
  LRTable Table = LRTable::buildForTests(
      G,
      {
          {/*State=*/1, id("type-name"), Action::goTo(/*NextState=*/5)},
          {/*State=*/2, id("type-name"), Action::goTo(/*NextState=*/5)},
      },
      {
          {/*State=*/3, /* type-name := class-name */ 0},
          {/*State=*/4, /* type-name := enum-name */ 1},
      });
  std::vector<const GSS::Node *> Heads = {GSSNode3, GSSNode4};
  glrReduce(Heads, tokenSymbol(tok::eof), {G, Table, Arena, GSStack});

  // Verify that the stack heads are joint at state 5 after reduces.
  EXPECT_THAT(Heads, UnorderedElementsAre(GSSNode3, GSSNode4,
                                          AllOf(state(5),
                                                parsedSymbolID(id("type-name")),
                                                parents({GSSNode1, GSSNode2}))))
      << Heads;
  // Verify that we create an ambiguous ForestNode of two parses of `type-name`.
  EXPECT_EQ(Heads.back()->Payload->dumpRecursive(G),
            "[  1, end) type-name := <ambiguous>\n"
            "[  1, end) ├─type-name := class-name\n"
            "[  1, end) │ └─class-name := <opaque>\n"
            "[  1, end) └─type-name := enum-name\n"
            "[  1, end)   └─enum-name := <opaque>\n");
}

TEST_F(GLRTest, ReduceJoiningWithSameBase) {
  //  Before (joining due to same goto state, the same base):
  //    0--1(class-name)--3(*)
  //    └--2(enum-name)--4(*)
  //  After reducing 3 by `pointer := class-name *` and
  //                 2 by `pointer := enum-name *`:
  //    0--5(pointer)  // 5 is goto(0, pointer)
  buildGrammar({"pointer", "class-name", "enum-name"},
               {"pointer := class-name *", "pointer := enum-name *"});

  auto *ClassNameNode = &Arena.createOpaque(id("class-name"), /*TokenIndex=*/0);
  auto *EnumNameNode = &Arena.createOpaque(id("enum-name"), /*TokenIndex=*/0);
  auto *StartTerminal = &Arena.createTerminal(tok::star, /*TokenIndex=*/1);

  const auto *GSSNode0 =
      GSStack.addNode(/*State=*/0, /*ForestNode=*/nullptr, /*Parents=*/{});
  const auto *GSSNode1 =
      GSStack.addNode(/*State=*/1, /*ForestNode=*/ClassNameNode,
                      /*Parents=*/{GSSNode0});
  const auto *GSSNode2 =
      GSStack.addNode(/*State=*/2, /*ForestNode=*/EnumNameNode,
                      /*Parents=*/{GSSNode0});
  const auto *GSSNode3 =
      GSStack.addNode(/*State=*/3, /*ForestNode=*/StartTerminal,
                      /*Parents=*/{GSSNode1});
  const auto *GSSNode4 =
      GSStack.addNode(/*State=*/4, /*ForestNode=*/StartTerminal,
                      /*Parents=*/{GSSNode2});

  // FIXME: figure out a way to get rid of the hard-coded reduce RuleID!
  LRTable Table =
      LRTable::buildForTests(G,
                             {
                                 {/*State=*/0, id("pointer"), Action::goTo(5)},
                             },
                             {
                                 {3, /* pointer := class-name */ 0},
                                 {4, /* pointer := enum-name */ 1},
                             });
  std::vector<const GSS::Node *> Heads = {GSSNode3, GSSNode4};
  glrReduce(Heads, tokenSymbol(tok::eof), {G, Table, Arena, GSStack});

  EXPECT_THAT(
      Heads, UnorderedElementsAre(GSSNode3, GSSNode4,
                                  AllOf(state(5), parsedSymbolID(id("pointer")),
                                        parents({GSSNode0}))))
      << Heads;
  EXPECT_EQ(Heads.back()->Payload->dumpRecursive(G),
            "[  0, end) pointer := <ambiguous>\n"
            "[  0, end) ├─pointer := class-name *\n"
            "[  0,   1) │ ├─class-name := <opaque>\n"
            "[  1, end) │ └─* := tok[1]\n"
            "[  0, end) └─pointer := enum-name *\n"
            "[  0,   1)   ├─enum-name := <opaque>\n"
            "[  1, end)   └─* := tok[1]\n");
}

TEST_F(GLRTest, ReduceLookahead) {
  // A term can be followed by +, but not by -.
  buildGrammar({"sum", "term"}, {"expr := term + term", "term := IDENTIFIER"});
  LRTable Table =
      LRTable::buildForTests(G,
                             {
                                 {/*State=*/0, id("term"), Action::goTo(2)},
                             },
                             {
                                 {/*State=*/1, 0},
                             });

  auto *Identifier = &Arena.createTerminal(tok::identifier, /*Start=*/0);

  const auto *Root =
      GSStack.addNode(/*State=*/0, /*ForestNode=*/nullptr, /*Parents=*/{});
  const auto *GSSNode1 =
      GSStack.addNode(/*State=*/1, /*ForestNode=*/Identifier, {Root});

  // When the lookahead is +, reduce is performed.
  std::vector<const GSS::Node *> Heads = {GSSNode1};
  glrReduce(Heads, tokenSymbol(tok::plus), {G, Table, Arena, GSStack});
  EXPECT_THAT(Heads,
              ElementsAre(GSSNode1, AllOf(state(2), parsedSymbolID(id("term")),
                                          parents(Root))));

  // When the lookahead is -, reduce is not performed.
  Heads = {GSSNode1};
  glrReduce(Heads, tokenSymbol(tok::minus), {G, Table, Arena, GSStack});
  EXPECT_THAT(Heads, ElementsAre(GSSNode1));
}

TEST_F(GLRTest, Recover) {
  // Recovery while parsing "word" inside braces.
  //  Before:
  //    0--1({)--2(?)
  //  After recovering a `word` at state 1:
  //    0--3(word)  // 3 is goto(1, word)
  buildGrammar({"word"}, {});
  LRTable Table = LRTable::buildForTests(
      G, {{/*State=*/1, id("word"), Action::goTo(3)}}, /*Reduce=*/{},
      /*Recovery=*/{{/*State=*/1, RecoveryStrategy::Braces, id("word")}});

  auto *LBrace = &Arena.createTerminal(tok::l_brace, 0);
  auto *Question1 = &Arena.createTerminal(tok::question, 1);
  const auto *Root = GSStack.addNode(0, nullptr, {});
  const auto *OpenedBraces = GSStack.addNode(1, LBrace, {Root});
  const auto *AfterQuestion1 = GSStack.addNode(2, Question1, {OpenedBraces});

  // Need a token stream with paired braces so the strategy works.
  clang::LangOptions LOptions;
  TokenStream Tokens = cook(lex("{ ? ? ? }", LOptions), LOptions);
  pairBrackets(Tokens);
  std::vector<const GSS::Node *> NewHeads;

  unsigned TokenIndex = 2;
  glrRecover({AfterQuestion1}, TokenIndex, Tokens, {G, Table, Arena, GSStack},
             NewHeads);
  EXPECT_EQ(TokenIndex, 4u) << "should skip ahead to matching brace";
  EXPECT_THAT(NewHeads, ElementsAre(
                            AllOf(state(3), parsedSymbolID(id("word")),
                                  parents({OpenedBraces}), start(1u))));
  EXPECT_EQ(NewHeads.front()->Payload->kind(), ForestNode::Opaque);

  // Test recovery failure: omit closing brace so strategy fails
  TokenStream NoRBrace = cook(lex("{ ? ? ? ?", LOptions), LOptions);
  pairBrackets(NoRBrace);
  NewHeads.clear();
  TokenIndex = 2;
  glrRecover({AfterQuestion1}, TokenIndex, NoRBrace,
             {G, Table, Arena, GSStack}, NewHeads);
  EXPECT_EQ(TokenIndex, 3u) << "should advance by 1 by default";
  EXPECT_THAT(NewHeads, IsEmpty());
}

TEST_F(GLRTest, RecoverRightmost) {
  // In a nested block structure, we recover at the innermost possible block.
  //  Before:
  //    0--1({)--1({)--1({)
  //  After recovering a `block` at inside the second braces:
  //    0--1({)--2(body)  // 2 is goto(1, body)
  buildGrammar({"body"}, {});
  LRTable Table = LRTable::buildForTests(
      G, {{/*State=*/1, id("body"), Action::goTo(2)}}, /*Reduce=*/{},
      /*Recovery=*/{{/*State=*/1, RecoveryStrategy::Braces, id("body")}});

  clang::LangOptions LOptions;
  // Innermost brace is unmatched, to test fallback to next brace.
  TokenStream Tokens = cook(lex("{ { { ? ? } }", LOptions), LOptions);
  Tokens.tokens()[0].Pair = 5;
  Tokens.tokens()[1].Pair = 4;
  Tokens.tokens()[4].Pair = 1;
  Tokens.tokens()[5].Pair = 0;

  auto *Brace1 = &Arena.createTerminal(tok::l_brace, 0);
  auto *Brace2 = &Arena.createTerminal(tok::l_brace, 1);
  auto *Brace3 = &Arena.createTerminal(tok::l_brace, 2);
  const auto *Root = GSStack.addNode(0, nullptr, {});
  const auto *In1 = GSStack.addNode(1, Brace1, {Root});
  const auto *In2 = GSStack.addNode(1, Brace2, {In1});
  const auto *In3 = GSStack.addNode(1, Brace3, {In2});

  unsigned TokenIndex = 3;
  std::vector<const GSS::Node *> NewHeads;
  glrRecover({In3}, TokenIndex, Tokens, {G, Table, Arena, GSStack}, NewHeads);
  EXPECT_EQ(TokenIndex, 5u);
  EXPECT_THAT(NewHeads, ElementsAre(AllOf(state(2), parsedSymbolID(id("body")),
                                          parents({In2}), start(2u))));
}

TEST_F(GLRTest, RecoverAlternatives) {
  // Recovery inside braces with multiple equally good options
  //  Before:
  //    0--1({)
  //  After recovering either `word` or `number` inside the braces:
  //    0--1({)--2(word)   // 2 is goto(1, word)
  //          └--3(number) // 3 is goto(1, number)
  buildGrammar({"number", "word"}, {});
  LRTable Table = LRTable::buildForTests(
      G,
      {
          {/*State=*/1, id("number"), Action::goTo(2)},
          {/*State=*/1, id("word"), Action::goTo(3)},
      },
      /*Reduce=*/{},
      /*Recovery=*/
      {
          {/*State=*/1, RecoveryStrategy::Braces, id("number")},
          {/*State=*/1, RecoveryStrategy::Braces, id("word")},
      });
  auto *LBrace = &Arena.createTerminal(tok::l_brace, 0);
  const auto *Root = GSStack.addNode(0, nullptr, {});
  const auto *OpenedBraces = GSStack.addNode(1, LBrace, {Root});

  clang::LangOptions LOptions;
  TokenStream Tokens = cook(lex("{ ? }", LOptions), LOptions);
  pairBrackets(Tokens);
  std::vector<const GSS::Node *> NewHeads;
  unsigned TokenIndex = 1;

  glrRecover({OpenedBraces}, TokenIndex, Tokens, {G, Table, Arena, GSStack},
             NewHeads);
  EXPECT_EQ(TokenIndex, 2u);
  EXPECT_THAT(NewHeads,
              UnorderedElementsAre(AllOf(state(2), parsedSymbolID(id("number")),
                                         parents({OpenedBraces}), start(1u)),
                                   AllOf(state(3), parsedSymbolID(id("word")),
                                         parents({OpenedBraces}), start(1u))));
}

TEST_F(GLRTest, PerfectForestNodeSharing) {
  // Run the GLR on a simple grammar and test that we build exactly one forest
  // node per (SymbolID, token range).

  // This is a grmammar where the original parsing-stack-based forest node
  // sharing approach will fail. In its LR0 graph, it has two states containing
  // item `expr := • IDENTIFIER`, and both have different goto states on the
  // nonterminal `expr`.
  build(R"bnf(
    _ := test

    test := { expr
    test := { IDENTIFIER
    test := left-paren expr
    left-paren := {
    expr := IDENTIFIER
  )bnf");
  clang::LangOptions LOptions;
  const TokenStream &Tokens = cook(lex("{ abc", LOptions), LOptions);
  auto LRTable = LRTable::buildSLR(G);

  const ForestNode &Parsed =
      glrParse(Tokens, {G, LRTable, Arena, GSStack}, id("test"));
  // Verify that there is no duplicated sequence node of `expr := IDENTIFIER`
  // in the forest, see the `#1` and `=#1` in the dump string.
  EXPECT_EQ(Parsed.dumpRecursive(G), "[  0, end) test := <ambiguous>\n"
                                     "[  0, end) ├─test := { expr\n"
                                     "[  0,   1) │ ├─{ := tok[0]\n"
                                     "[  1, end) │ └─expr := IDENTIFIER #1\n"
                                     "[  1, end) │   └─IDENTIFIER := tok[1]\n"
                                     "[  0, end) ├─test := { IDENTIFIER\n"
                                     "[  0,   1) │ ├─{ := tok[0]\n"
                                     "[  1, end) │ └─IDENTIFIER := tok[1]\n"
                                     "[  0, end) └─test := left-paren expr\n"
                                     "[  0,   1)   ├─left-paren := {\n"
                                     "[  0,   1)   │ └─{ := tok[0]\n"
                                     "[  1, end)   └─expr := IDENTIFIER =#1\n"
                                     "[  1, end)     └─IDENTIFIER := tok[1]\n");
}

TEST_F(GLRTest, GLRReduceOrder) {
  // Given the following grammar, and the input `IDENTIFIER`, reductions should
  // be performed in the following order:
  //  1. foo := IDENTIFIER
  //  2. { test := IDENTIFIER, test := foo }
  // foo should be reduced first, so that in step 2 we have completed reduces
  // for test, and form an ambiguous forest node.
  build(R"bnf(
    _ := test

    test := IDENTIFIER
    test := foo
    foo := IDENTIFIER
  )bnf");
  clang::LangOptions LOptions;
  const TokenStream &Tokens = cook(lex("IDENTIFIER", LOptions), LOptions);
  auto LRTable = LRTable::buildSLR(G);

  const ForestNode &Parsed =
      glrParse(Tokens, {G, LRTable, Arena, GSStack}, id("test"));
  EXPECT_EQ(Parsed.dumpRecursive(G), "[  0, end) test := <ambiguous>\n"
                                     "[  0, end) ├─test := IDENTIFIER\n"
                                     "[  0, end) │ └─IDENTIFIER := tok[0]\n"
                                     "[  0, end) └─test := foo\n"
                                     "[  0, end)   └─foo := IDENTIFIER\n"
                                     "[  0, end)     └─IDENTIFIER := tok[0]\n");
}

TEST_F(GLRTest, RecoveryEndToEnd) {
  // Simple example of brace-based recovery showing:
  //  - recovered region includes tokens both ahead of and behind the cursor
  //  - multiple possible recovery rules
  //  - recovery from outer scopes is rejected
  build(R"bnf(
    _ := block

    block := { block }
    block := { numbers }
    numbers := NUMERIC_CONSTANT NUMERIC_CONSTANT
  )bnf");
  auto LRTable = LRTable::buildSLR(G);
  clang::LangOptions LOptions;
  TokenStream Tokens = cook(lex("{ { 42 ? } }", LOptions), LOptions);
  pairBrackets(Tokens);

  const ForestNode &Parsed =
      glrParse(Tokens, {G, LRTable, Arena, GSStack}, id("block"));
  EXPECT_EQ(Parsed.dumpRecursive(G),
            "[  0, end) block := { block [recover=1] }\n"
            "[  0,   1) ├─{ := tok[0]\n"
            "[  1,   5) ├─block := <ambiguous>\n"
            "[  1,   5) │ ├─block := { block [recover=1] }\n"
            "[  1,   2) │ │ ├─{ := tok[1]\n"
            "[  2,   4) │ │ ├─block := <opaque>\n"
            "[  4,   5) │ │ └─} := tok[4]\n"
            "[  1,   5) │ └─block := { numbers [recover=1] }\n"
            "[  1,   2) │   ├─{ := tok[1]\n"
            "[  2,   4) │   ├─numbers := <opaque>\n"
            "[  4,   5) │   └─} := tok[4]\n"
            "[  5, end) └─} := tok[5]\n");
}

TEST_F(GLRTest, NoExplicitAccept) {
  build(R"bnf(
    _ := test

    test := IDENTIFIER test
    test := IDENTIFIER
  )bnf");
  clang::LangOptions LOptions;
  // Given the following input, and the grammar above, we perform two reductions
  // of the nonterminal `test` when the next token is `eof`, verify that the
  // parser stops at the right state.
  const TokenStream &Tokens = cook(lex("id id", LOptions), LOptions);
  auto LRTable = LRTable::buildSLR(G);

  const ForestNode &Parsed =
      glrParse(Tokens, {G, LRTable, Arena, GSStack}, id("test"));
  EXPECT_EQ(Parsed.dumpRecursive(G), "[  0, end) test := IDENTIFIER test\n"
                                     "[  0,   1) ├─IDENTIFIER := tok[0]\n"
                                     "[  1, end) └─test := IDENTIFIER\n"
                                     "[  1, end)   └─IDENTIFIER := tok[1]\n");
}

TEST(GSSTest, GC) {
  //      ┌-A-┬-AB
  //      ├-B-┘
  // Root-+-C
  //      ├-D
  //      └-E
  GSS GSStack;
  auto *Root = GSStack.addNode(0, nullptr, {});
  auto *A = GSStack.addNode(0, nullptr, {Root});
  auto *B = GSStack.addNode(0, nullptr, {Root});
  auto *C = GSStack.addNode(0, nullptr, {Root});
  auto *D = GSStack.addNode(0, nullptr, {Root});
  auto *AB = GSStack.addNode(0, nullptr, {A, B});

  EXPECT_EQ(1u, GSStack.gc({AB, C})) << "D is destroyed";
  EXPECT_EQ(0u, GSStack.gc({AB, C})) << "D is already gone";
  auto *E = GSStack.addNode(0, nullptr, {Root});
  EXPECT_EQ(D, E) << "Storage of GCed node D is reused for E";
  EXPECT_EQ(3u, GSStack.gc({A, E})) << "Destroys B, AB, C";
  EXPECT_EQ(1u, GSStack.gc({E})) << "Destroys A";
}

} // namespace
} // namespace pseudo
} // namespace clang
