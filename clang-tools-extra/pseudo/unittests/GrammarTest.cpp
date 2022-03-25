//===--- GrammarTest.cpp - grammar tests  -----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang-pseudo/Grammar.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <memory>

namespace clang {
namespace pseudo {
namespace {

using testing::AllOf;
using testing::ElementsAre;
using testing::IsEmpty;
using testing::Pair;
using testing::UnorderedElementsAre;

MATCHER_P(TargetID, SID, "") { return arg.Target == SID; }
template <typename... T> testing::Matcher<const Rule &> Sequence(T... IDs) {
  return testing::Property(&Rule::seq, ElementsAre(IDs...));
}

class GrammarTest : public ::testing::Test {
public:
  void build(llvm::StringRef BNF) {
    Diags.clear();
    G = Grammar::parseBNF(BNF, Diags);
  }

  SymbolID id(llvm::StringRef Name) const {
    for (unsigned I = 0; I < NumTerminals; ++I)
      if (G->table().Terminals[I] == Name)
        return tokenSymbol(static_cast<tok::TokenKind>(I));
    for (SymbolID ID = 0; ID < G->table().Nonterminals.size(); ++ID)
      if (G->table().Nonterminals[ID].Name == Name)
        return ID;
    ADD_FAILURE() << "No such symbol found: " << Name;
    return 0;
  }

  RuleID ruleFor(llvm::StringRef NonterminalName) const {
    auto RuleRange = G->table().Nonterminals[id(NonterminalName)].RuleRange;
    if (RuleRange.End - RuleRange.Start == 1)
      return G->table().Nonterminals[id(NonterminalName)].RuleRange.Start;
    ADD_FAILURE() << "Expected a single rule for " << NonterminalName
                  << ", but it has " << RuleRange.End - RuleRange.Start
                  << " rule!\n";
    return 0;
  }

protected:
  std::unique_ptr<Grammar> G;
  std::vector<std::string> Diags;
};

TEST_F(GrammarTest, Basic) {
  build("_ := IDENTIFIER + _ # comment");
  EXPECT_THAT(Diags, IsEmpty());

  auto ExpectedRule =
      AllOf(TargetID(id("_")), Sequence(id("IDENTIFIER"), id("+"), id("_")));
  EXPECT_EQ(G->symbolName(id("_")), "_");
  EXPECT_THAT(G->rulesFor(id("_")), UnorderedElementsAre(ExpectedRule));
  const auto &Rule = G->lookupRule(/*RID=*/0);
  EXPECT_THAT(Rule, ExpectedRule);
  EXPECT_THAT(G->symbolName(Rule.seq()[0]), "IDENTIFIER");
  EXPECT_THAT(G->symbolName(Rule.seq()[1]), "+");
  EXPECT_THAT(G->symbolName(Rule.seq()[2]), "_");
}

TEST_F(GrammarTest, EliminatedOptional) {
  build("_ := CONST_opt INT ;_opt");
  EXPECT_THAT(Diags, IsEmpty());
  EXPECT_THAT(G->table().Rules,
              UnorderedElementsAre(Sequence(id("INT")),
                                   Sequence(id("CONST"), id("INT")),
                                   Sequence(id("CONST"), id("INT"), id(";")),
                                   Sequence(id("INT"), id(";"))));
}

TEST_F(GrammarTest, RuleIDSorted) {
  build(R"bnf(
    _ := x

    x := y
    y := z
    z := IDENTIFIER
  )bnf");
  ASSERT_TRUE(Diags.empty());

  EXPECT_LT(ruleFor("z"), ruleFor("y"));
  EXPECT_LT(ruleFor("y"), ruleFor("x"));
  EXPECT_LT(ruleFor("x"), ruleFor("_"));
}

TEST_F(GrammarTest, Diagnostics) {
  build(R"cpp(
    _ := ,_opt
    _ := undefined-sym
    null :=
    _ := IDENFIFIE # a typo of the terminal IDENFITIER

    invalid
    # cycle
    a := b
    b := a
  )cpp");

  EXPECT_EQ(G->startSymbol(), id("_"));
  EXPECT_THAT(Diags, UnorderedElementsAre(
                         "Rule '_ := ,_opt' has a nullable RHS",
                         "Rule 'null := ' has a nullable RHS",
                         "No rules for nonterminal: undefined-sym",
                         "Failed to parse 'invalid': no separator :=",
                         "Token-like name IDENFIFIE is used as a nonterminal",
                         "No rules for nonterminal: IDENFIFIE",
                         "The grammar contains a cycle involving symbol a"));
}

TEST_F(GrammarTest, FirstAndFollowSets) {
  build(
      R"bnf(
_ := expr
expr := expr - term
expr := term
term := IDENTIFIER
term := ( expr )
)bnf");
  ASSERT_TRUE(Diags.empty());
  auto ToPairs = [](std::vector<llvm::DenseSet<SymbolID>> Input) {
    std::vector<std::pair<SymbolID, llvm::DenseSet<SymbolID>>> Sets;
    for (SymbolID ID = 0; ID < Input.size(); ++ID)
      Sets.emplace_back(ID, std::move(Input[ID]));
    return Sets;
  };

  EXPECT_THAT(
      ToPairs(firstSets(*G)),
      UnorderedElementsAre(
          Pair(id("_"), UnorderedElementsAre(id("IDENTIFIER"), id("("))),
          Pair(id("expr"), UnorderedElementsAre(id("IDENTIFIER"), id("("))),
          Pair(id("term"), UnorderedElementsAre(id("IDENTIFIER"), id("(")))));
  EXPECT_THAT(
      ToPairs(followSets(*G)),
      UnorderedElementsAre(
          Pair(id("_"), UnorderedElementsAre(id("EOF"))),
          Pair(id("expr"), UnorderedElementsAre(id("-"), id("EOF"), id(")"))),
          Pair(id("term"), UnorderedElementsAre(id("-"), id("EOF"), id(")")))));

  build(R"bnf(
# A simplfied C++ decl-specifier-seq.
_ := decl-specifier-seq
decl-specifier-seq := decl-specifier decl-specifier-seq
decl-specifier-seq := decl-specifier
decl-specifier := simple-type-specifier
decl-specifier := INLINE
simple-type-specifier := INT
   )bnf");
  ASSERT_TRUE(Diags.empty());
  EXPECT_THAT(
      ToPairs(firstSets(*G)),
      UnorderedElementsAre(
          Pair(id("_"), UnorderedElementsAre(id("INLINE"), id("INT"))),
          Pair(id("decl-specifier-seq"),
               UnorderedElementsAre(id("INLINE"), id("INT"))),
          Pair(id("simple-type-specifier"), UnorderedElementsAre(id("INT"))),
          Pair(id("decl-specifier"),
               UnorderedElementsAre(id("INLINE"), id("INT")))));
  EXPECT_THAT(
      ToPairs(followSets(*G)),
      UnorderedElementsAre(
          Pair(id("_"), UnorderedElementsAre(id("EOF"))),
          Pair(id("decl-specifier-seq"), UnorderedElementsAre(id("EOF"))),
          Pair(id("decl-specifier"),
               UnorderedElementsAre(id("INLINE"), id("INT"), id("EOF"))),
          Pair(id("simple-type-specifier"),
               UnorderedElementsAre(id("INLINE"), id("INT"), id("EOF")))));
}

} // namespace
} // namespace pseudo
} // namespace clang
