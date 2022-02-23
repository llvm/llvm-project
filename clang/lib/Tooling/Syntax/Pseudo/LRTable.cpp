//===--- LRTable.cpp - Parsing table for LR parsers --------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Tooling/Syntax/Pseudo/LRTable.h"
#include "clang/Tooling/Syntax/Pseudo/Grammar.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"

namespace clang {
namespace syntax {
namespace pseudo {

llvm::raw_ostream &operator<<(llvm::raw_ostream &OS, const LRTable::Action &A) {
  switch (A.kind()) {
  case LRTable::Action::Shift:
    return OS << llvm::formatv("shift state {0}", A.getShiftState());
  case LRTable::Action::Reduce:
    return OS << llvm::formatv("reduce by rule {0}", A.getReduceRule());
  case LRTable::Action::GoTo:
    return OS << llvm::formatv("go to state {0}", A.getGoToState());
  case LRTable::Action::Accept:
    return OS << "acc";
  case LRTable::Action::Sentinel:
    llvm_unreachable("unexpected Sentinel action kind!");
  }
}

std::string LRTable::dumpStatistics() const {
  StateID NumOfStates = 0;
  for (StateID It : States)
    NumOfStates = std::max(It, NumOfStates);
  return llvm::formatv(R"(
Statistics of the LR parsing table:
    number of states: {0}
    number of actions: {1}
    size of the table (bytes): {2}
)",
                       NumOfStates, Actions.size(), bytes())
      .str();
}

std::string LRTable::dumpForTests(const Grammar &G) const {
  std::string Result;
  llvm::raw_string_ostream OS(Result);
  StateID MaxState = 0;
  for (StateID It : States)
    MaxState = std::max(MaxState, It);
  OS << "LRTable:\n";
  for (StateID S = 0; S <= MaxState; ++S) {
    OS << llvm::formatv("State {0}\n", S);
    for (uint16_t Terminal = 0; Terminal < NumTerminals; ++Terminal) {
      SymbolID TokID = tokenSymbol(static_cast<tok::TokenKind>(Terminal));
      for (auto A : find(S, TokID)) {
        if (A.kind() == LRTable::Action::Shift)
          OS.indent(4) << llvm::formatv("'{0}': shift state {1}\n",
                                        G.symbolName(TokID), A.getShiftState());
        else if (A.kind() == LRTable::Action::Reduce)
          OS.indent(4) << llvm::formatv("'{0}': reduce by rule {1} '{2}'\n",
                                        G.symbolName(TokID), A.getReduceRule(),
                                        G.dumpRule(A.getReduceRule()));
        else if (A.kind() == LRTable::Action::Accept)
          OS.indent(4) << llvm::formatv("'{0}': accept\n", G.symbolName(TokID));
      }
    }
    for (SymbolID NontermID = 0; NontermID < G.table().Nonterminals.size();
         ++NontermID) {
      if (find(S, NontermID).empty())
        continue;
      OS.indent(4) << llvm::formatv("'{0}': go to state {1}\n",
                                    G.symbolName(NontermID),
                                    getGoToState(S, NontermID));
    }
  }
  return OS.str();
}

llvm::ArrayRef<LRTable::Action> LRTable::getActions(StateID State,
                                                    SymbolID Terminal) const {
  assert(pseudo::isToken(Terminal) && "expect terminal symbol!");
  return find(State, Terminal);
}

LRTable::StateID LRTable::getGoToState(StateID State,
                                       SymbolID Nonterminal) const {
  assert(pseudo::isNonterminal(Nonterminal) && "expected nonterminal symbol!");
  auto Result = find(State, Nonterminal);
  assert(Result.size() == 1 && Result.front().kind() == Action::GoTo);
  return Result.front().getGoToState();
}

llvm::ArrayRef<LRTable::Action> LRTable::find(StateID Src, SymbolID ID) const {
  size_t Idx = isToken(ID) ? symbolToToken(ID) : ID;
  assert(isToken(ID) ? Idx + 1 < TerminalOffset.size()
                     : Idx + 1 < NontermOffset.size());
  std::pair<size_t, size_t> TargetStateRange =
      isToken(ID) ? std::make_pair(TerminalOffset[Idx], TerminalOffset[Idx + 1])
                  : std::make_pair(NontermOffset[Idx], NontermOffset[Idx + 1]);
  auto TargetedStates =
      llvm::makeArrayRef(States.data() + TargetStateRange.first,
                         States.data() + TargetStateRange.second);

  assert(llvm::is_sorted(TargetedStates) &&
         "subrange of the StateIdx should be sorted!");
  const LRTable::StateID *It = llvm::partition_point(
      TargetedStates, [&Src](LRTable::StateID S) { return S < Src; });
  if (It == TargetedStates.end())
    return {};
  size_t Start = It - States.data(), End = Start;
  while (End < States.size() && States[End] == Src)
    ++End;
  return llvm::makeArrayRef(&Actions[Start], &Actions[End]);
}

} // namespace pseudo
} // namespace syntax
} // namespace clang
