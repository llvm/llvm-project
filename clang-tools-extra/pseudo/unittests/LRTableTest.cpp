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
using Action = LRTable::Action;

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

  //           eof  IDENT   term
  // +-------+----+-------+------+
  // |state0 |    | s0    |      |
  // |state1 |    |       | g3   |
  // |state2 |    |       |      |
  // +-------+----+-------+------+-------
  std::vector<LRTable::Entry> Entries = {
      {/* State */ 0, Identifier, Action::shift(0)},
      {/* State */ 1, Term, Action::goTo(3)},
  };
  std::vector<LRTable::ReduceEntry> ReduceEntries = {
      {/*State=*/0, /*Rule=*/0},
      {/*State=*/1, /*Rule=*/2},
      {/*State=*/2, /*Rule=*/1},
  };
  LRTable T = LRTable::buildForTests(G, Entries, ReduceEntries);
  EXPECT_EQ(T.getShiftState(0, Eof), llvm::None);
  EXPECT_THAT(T.getShiftState(0, Identifier), ValueIs(0));
  EXPECT_THAT(T.getReduceRules(0), ElementsAre(0));

  EXPECT_EQ(T.getShiftState(1, Eof), llvm::None);
  EXPECT_EQ(T.getShiftState(1, Identifier), llvm::None);
  EXPECT_THAT(T.getGoToState(1, Term), ValueIs(3));
  EXPECT_THAT(T.getReduceRules(1), ElementsAre(2));

  // Verify the behaivor for other non-available-actions terminals.
  SymbolID Int = tokenSymbol(tok::kw_int);
  EXPECT_EQ(T.getShiftState(2, Int), llvm::None);

  // Check follow sets.
  EXPECT_TRUE(T.canFollow(Term, Plus));
  EXPECT_TRUE(T.canFollow(Term, Eof));
  EXPECT_FALSE(T.canFollow(Term, Int));
}

} // namespace
} // namespace pseudo
} // namespace clang
