//===--- LRTable.cpp - Parsing table for LR parsers --------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang-pseudo/LRTable.h"
#include "clang-pseudo/Grammar.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"

namespace clang {
namespace pseudo {

llvm::raw_ostream &operator<<(llvm::raw_ostream &OS, const LRTable::Action &A) {
  switch (A.kind()) {
  case LRTable::Action::Shift:
    return OS << llvm::formatv("shift state {0}", A.getShiftState());
  case LRTable::Action::Reduce:
    return OS << llvm::formatv("reduce by rule {0}", A.getReduceRule());
  case LRTable::Action::GoTo:
    return OS << llvm::formatv("go to state {0}", A.getGoToState());
  case LRTable::Action::Sentinel:
    llvm_unreachable("unexpected Sentinel action kind!");
  }
  llvm_unreachable("unexpected action kind!");
}

std::string LRTable::dumpStatistics() const {
  return llvm::formatv(R"(
Statistics of the LR parsing table:
    number of states: {0}
    number of actions: {1}
    size of the table (bytes): {2}
)",
                       StateOffset.size() - 1, Actions.size(), bytes())
      .str();
}

std::string LRTable::dumpForTests(const Grammar &G) const {
  std::string Result;
  llvm::raw_string_ostream OS(Result);
  OS << "LRTable:\n";
  for (StateID S = 0; S < StateOffset.size() - 1; ++S) {
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
  assert(Src + 1u < StateOffset.size());
  std::pair<size_t, size_t> Range =
      std::make_pair(StateOffset[Src], StateOffset[Src + 1]);
  auto SymbolRange = llvm::makeArrayRef(Symbols.data() + Range.first,
                                        Symbols.data() + Range.second);

  assert(llvm::is_sorted(SymbolRange) &&
         "subrange of the Symbols should be sorted!");
  const LRTable::StateID *Start =
      llvm::partition_point(SymbolRange, [&ID](SymbolID S) { return S < ID; });
  if (Start == SymbolRange.end())
    return {};
  const LRTable::StateID *End = Start;
  while (End != SymbolRange.end() && *End == ID)
    ++End;
  return llvm::makeArrayRef(&Actions[Start - Symbols.data()],
                            /*length=*/End - Start);
}

LRTable::StateID LRTable::getStartState(SymbolID Target) const {
  assert(llvm::is_sorted(StartStates) && "StartStates must be sorted!");
  auto It = llvm::partition_point(
      StartStates, [Target](const std::pair<SymbolID, StateID> &X) {
        return X.first < Target;
      });
  assert(It != StartStates.end() && It->first == Target &&
         "target symbol doesn't have a start state!");
  return It->second;
}

} // namespace pseudo
} // namespace clang
