//===--- LRTable.cpp - Parsing table for LR parsers --------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang-pseudo/grammar/LRTable.h"
#include "clang-pseudo/grammar/Grammar.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"

namespace clang {
namespace pseudo {

llvm::raw_ostream &operator<<(llvm::raw_ostream &OS, const LRTable::Action &A) {
  switch (A.kind()) {
  case LRTable::Action::Shift:
    return OS << llvm::formatv("shift state {0}", A.getShiftState());
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
    number of actions: shift={1} goto={2} reduce={3}
    size of the table (bytes): {4}
)",
                       numStates(), Shifts.size(), Gotos.size(), Reduces.size(),
                       bytes())
      .str();
}

std::string LRTable::dumpForTests(const Grammar &G) const {
  std::string Result;
  llvm::raw_string_ostream OS(Result);
  OS << "LRTable:\n";
  for (StateID S = 0; S < numStates(); ++S) {
    OS << llvm::formatv("State {0}\n", S);
    for (uint16_t Terminal = 0; Terminal < NumTerminals; ++Terminal) {
      SymbolID TokID = tokenSymbol(static_cast<tok::TokenKind>(Terminal));
      if (auto SS = getShiftState(S, TokID))
        OS.indent(4) << llvm::formatv("{0}: shift state {1}\n",
                                      G.symbolName(TokID), SS);
    }
    for (RuleID R : getReduceRules(S)) {
      SymbolID Target = G.lookupRule(R).Target;
      std::vector<llvm::StringRef> Terminals;
      for (unsigned Terminal = 0; Terminal < NumTerminals; ++Terminal) {
        SymbolID TokID = tokenSymbol(static_cast<tok::TokenKind>(Terminal));
        if (canFollow(Target, TokID))
          Terminals.push_back(G.symbolName(TokID));
      }
      OS.indent(4) << llvm::formatv("{0}: reduce by rule {1} '{2}'\n",
                                    llvm::join(Terminals, " "), R,
                                    G.dumpRule(R));
    }
    for (SymbolID NontermID = 0; NontermID < G.table().Nonterminals.size();
         ++NontermID) {
      if (auto GS = getGoToState(S, NontermID)) {
        OS.indent(4) << llvm::formatv("{0}: go to state {1}\n",
                                      G.symbolName(NontermID), *GS);
      }
    }
  }
  return OS.str();
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
