//===--- LRTableTest.cpp - ---------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang-pseudo/grammar/LRTable.h"
#include "clang-pseudo/grammar/Grammar.h"
#include "clang/Basic/TokenKinds.h"
#include "llvm/Testing/Support/SupportHelpers.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <vector>

namespace clang {
namespace pseudo {
namespace {

using llvm::ValueIs;
using testing::ElementsAre;
using StateID = LRTable::StateID;

TEST(LRTable, Builder) {
  std::vector<std::string> GrammarDiags;
  Grammar G = Grammar::parseBNF(R"bnf(
    _ := expr            # rule 0
    expr := term         # rule 1
    expr := expr + term  # rule 2
    term := IDENTIFIER   # rule 3
  )bnf",
                                GrammarDiags);
  EXPECT_THAT(GrammarDiags, testing::IsEmpty());

  SymbolID Term = *G.findNonterminal("term");
  SymbolID Eof = tokenSymbol(tok::eof);
  SymbolID Identifier = tokenSymbol(tok::identifier);
  SymbolID Plus = tokenSymbol(tok::plus);

  LRTable::Builder B(G);
  //           eof  IDENT   term
  // +-------+----+-------+------+
  // |state0 |    | s0    |      |
  // |state1 |    |       | g3   |
  // |state2 |    |       |      |
  // +-------+----+-------+------+-------
  B.Transition[{StateID{0}, Identifier}] = StateID{0};
  B.Transition[{StateID{1}, Term}] = StateID{3};
  B.Reduce[StateID{0}].insert(RuleID{0});
  B.Reduce[StateID{1}].insert(RuleID{2});
  B.Reduce[StateID{2}].insert(RuleID{1});
  LRTable T = std::move(B).build();

  EXPECT_EQ(T.getShiftState(0, Eof), std::nullopt);
  EXPECT_THAT(T.getShiftState(0, Identifier), ValueIs(0));
  EXPECT_THAT(T.getReduceRules(0), ElementsAre(0));

  EXPECT_EQ(T.getShiftState(1, Eof), std::nullopt);
  EXPECT_EQ(T.getShiftState(1, Identifier), std::nullopt);
  EXPECT_THAT(T.getGoToState(1, Term), ValueIs(3));
  EXPECT_THAT(T.getReduceRules(1), ElementsAre(2));

  // Verify the behaivor for other non-available-actions terminals.
  SymbolID Int = tokenSymbol(tok::kw_int);
  EXPECT_EQ(T.getShiftState(2, Int), std::nullopt);

  // Check follow sets.
  EXPECT_TRUE(T.canFollow(Term, Plus));
  EXPECT_TRUE(T.canFollow(Term, Eof));
  EXPECT_FALSE(T.canFollow(Term, Int));
}

} // namespace
} // namespace pseudo
} // namespace clang
