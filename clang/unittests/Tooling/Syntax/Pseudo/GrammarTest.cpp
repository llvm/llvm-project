//===--- GrammarTest.cpp - grammar tests  -----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Tooling/Syntax/Pseudo/Grammar.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <memory>

namespace clang {
namespace syntax {
namespace pseudo {
namespace {

using testing::AllOf;
using testing::ElementsAre;
using testing::IsEmpty;
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

  SymbolID lookup(llvm::StringRef Name) const {
    for (unsigned I = 0; I < NumTerminals; ++I)
      if (G->table().Terminals[I] == Name)
        return tokenSymbol(static_cast<tok::TokenKind>(I));
    for (SymbolID ID = 0; ID < G->table().Nonterminals.size(); ++ID)
      if (G->table().Nonterminals[ID].Name == Name)
        return ID;
    ADD_FAILURE() << "No such symbol found: " << Name;
    return 0;
  }

protected:
  std::unique_ptr<Grammar> G;
  std::vector<std::string> Diags;
};

TEST_F(GrammarTest, Basic) {
  build("expression := IDENTIFIER + expression # comment");
  EXPECT_THAT(Diags, IsEmpty());

  auto ExpectedRule =
      AllOf(TargetID(lookup("expression")),
            Sequence(lookup("IDENTIFIER"), lookup("+"), lookup("expression")));
  auto ExpressionID = lookup("expression");
  EXPECT_EQ(G->symbolName(ExpressionID), "expression");
  EXPECT_THAT(G->rulesFor(ExpressionID), UnorderedElementsAre(ExpectedRule));
  const auto &Rule = G->lookupRule(/*RID=*/0);
  EXPECT_THAT(Rule, ExpectedRule);
  EXPECT_THAT(G->symbolName(Rule.seq()[0]), "IDENTIFIER");
  EXPECT_THAT(G->symbolName(Rule.seq()[1]), "+");
  EXPECT_THAT(G->symbolName(Rule.seq()[2]), "expression");
}

TEST_F(GrammarTest, EliminatedOptional) {
  build("_ := CONST_opt INT ;_opt");
  EXPECT_THAT(Diags, IsEmpty());
  EXPECT_THAT(G->table().Rules,
              UnorderedElementsAre(
                  Sequence(lookup("INT")),
                  Sequence(lookup("CONST"), lookup("INT")),
                  Sequence(lookup("CONST"), lookup("INT"), lookup(";")),
                  Sequence(lookup("INT"), lookup(";"))));
}

TEST_F(GrammarTest, Diagnostics) {
  build(R"cpp(
    _ := ,_opt
    _ := undefined-sym
    null :=
    _ := IDENFIFIE # a typo of the terminal IDENFITIER

    invalid
  )cpp");

  EXPECT_THAT(Diags, UnorderedElementsAre(
                         "Rule '_ := ,_opt' has a nullable RHS",
                         "Rule 'null := ' has a nullable RHS",
                         "No rules for nonterminal: undefined-sym",
                         "Failed to parse 'invalid': no separator :=",
                         "Token-like name IDENFIFIE is used as a nonterminal",
                         "No rules for nonterminal: IDENFIFIE"));
}

} // namespace
} // namespace pseudo
} // namespace syntax
} // namespace clang
