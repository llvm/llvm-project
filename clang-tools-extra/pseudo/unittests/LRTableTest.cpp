//===--- LRTableTest.cpp - ---------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang-pseudo/LRTable.h"
#include "clang-pseudo/Grammar.h"
#include "clang/Basic/TokenKinds.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <vector>

namespace clang {
namespace pseudo {
namespace {

using testing::IsEmpty;
using testing::UnorderedElementsAre;
using Action = LRTable::Action;

TEST(LRTable, Builder) {
  GrammarTable GTable;

  //           eof   semi  ...
  // +-------+----+-------+---
  // |state0 |    | s0,r0 |...
  // |state1 | acc|       |...
  // |state2 |    |  r1   |...
  // +-------+----+-------+---
  std::vector<LRTable::Entry> Entries = {
      {/* State */ 0, tokenSymbol(tok::semi), Action::shift(0)},
      {/* State */ 0, tokenSymbol(tok::semi), Action::reduce(0)},
      {/* State */ 1, tokenSymbol(tok::eof), Action::reduce(2)},
      {/* State */ 2, tokenSymbol(tok::semi), Action::reduce(1)}};
  GrammarTable GT;
  LRTable T = LRTable::buildForTests(GT, Entries);
  EXPECT_THAT(T.find(0, tokenSymbol(tok::eof)), IsEmpty());
  EXPECT_THAT(T.find(0, tokenSymbol(tok::semi)),
              UnorderedElementsAre(Action::shift(0), Action::reduce(0)));
  EXPECT_THAT(T.find(1, tokenSymbol(tok::eof)),
              UnorderedElementsAre(Action::reduce(2)));
  EXPECT_THAT(T.find(1, tokenSymbol(tok::semi)), IsEmpty());
  EXPECT_THAT(T.find(2, tokenSymbol(tok::semi)),
              UnorderedElementsAre(Action::reduce(1)));
  // Verify the behaivor for other non-available-actions terminals.
  EXPECT_THAT(T.find(2, tokenSymbol(tok::kw_int)), IsEmpty());
}

} // namespace
} // namespace pseudo
} // namespace clang
