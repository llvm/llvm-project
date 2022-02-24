//===--- LRTableBuild.cpp - Build a LRTable from LRGraph ---------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Basic/TokenKinds.h"
#include "clang/Tooling/Syntax/Pseudo/Grammar.h"
#include "clang/Tooling/Syntax/Pseudo/LRGraph.h"
#include "clang/Tooling/Syntax/Pseudo/LRTable.h"
#include <cstdint>

namespace llvm {
template <> struct DenseMapInfo<clang::syntax::pseudo::LRTable::Entry> {
  using Entry = clang::syntax::pseudo::LRTable::Entry;
  static inline Entry getEmptyKey() {
    static Entry E{static_cast<clang::syntax::pseudo::SymbolID>(-1), 0,
                   clang::syntax::pseudo::LRTable::Action::sentinel()};
    return E;
  }
  static inline Entry getTombstoneKey() {
    static Entry E{static_cast<clang::syntax::pseudo::SymbolID>(-2), 0,
                   clang::syntax::pseudo::LRTable::Action::sentinel()};
    return E;
  }
  static unsigned getHashValue(const Entry &I) {
    return llvm::hash_combine(I.State, I.Symbol, I.Act.opaque());
  }
  static bool isEqual(const Entry &LHS, const Entry &RHS) {
    return LHS.State == RHS.State && LHS.Symbol == RHS.Symbol &&
           LHS.Act == RHS.Act;
  }
};
} // namespace llvm

namespace clang {
namespace syntax {
namespace pseudo {

class LRTable::Builder {
public:
  bool insert(Entry E) { return Entries.insert(std::move(E)).second; }
  LRTable build(const GrammarTable &GT) && {
    // E.g. given the following parsing table with 3 states and 3 terminals:
    //
    //            a    b     c
    // +-------+----+-------+-+
    // |state0 |    | s0,r0 | |
    // |state1 | acc|       | |
    // |state2 |    |  r1   | |
    // +-------+----+-------+-+
    //
    // The final LRTable:
    //  - TerminalOffset: [a] = 0, [b] = 1, [c] = 4, [d] = 4 (d is a sentinel)
    //  -  States:     [ 1,    0,  0,  2]
    //    Actions:     [ acc, s0, r0, r1]
    //                   ~~~ corresponding range for terminal a
    //                        ~~~~~~~~~~ corresponding range for terminal b
    // First step, we sort all entries by (Symbol, State, Action).
    std::vector<Entry> Sorted(Entries.begin(), Entries.end());
    llvm::sort(Sorted, [](const Entry &L, const Entry &R) {
      return std::forward_as_tuple(L.Symbol, L.State, L.Act.opaque()) <
             std::forward_as_tuple(R.Symbol, R.State, R.Act.opaque());
    });

    LRTable Table;
    Table.Actions.reserve(Sorted.size());
    Table.States.reserve(Sorted.size());
    // We are good to finalize the States and Actions.
    for (const auto &E : Sorted) {
      Table.Actions.push_back(E.Act);
      Table.States.push_back(E.State);
    }
    // Initialize the terminal and nonterminal idx, all ranges are empty by
    // default.
    Table.TerminalOffset = std::vector<uint32_t>(GT.Terminals.size() + 1, 0);
    Table.NontermOffset = std::vector<uint32_t>(GT.Nonterminals.size() + 1, 0);
    size_t SortedIndex = 0;
    for (SymbolID NonterminalID = 0; NonterminalID < Table.NontermOffset.size();
         ++NonterminalID) {
      Table.NontermOffset[NonterminalID] = SortedIndex;
      while (SortedIndex < Sorted.size() &&
             Sorted[SortedIndex].Symbol == NonterminalID)
        ++SortedIndex;
    }
    for (size_t Terminal = 0; Terminal < Table.TerminalOffset.size();
         ++Terminal) {
      Table.TerminalOffset[Terminal] = SortedIndex;
      while (SortedIndex < Sorted.size() &&
             Sorted[SortedIndex].Symbol ==
                 tokenSymbol(static_cast<tok::TokenKind>(Terminal)))
        ++SortedIndex;
    }
    return Table;
  }

private:
  llvm::DenseSet<Entry> Entries;
};

LRTable LRTable::buildForTests(const GrammarTable &GT,
                               llvm::ArrayRef<Entry> Entries) {
  Builder Build;
  for (const Entry &E : Entries)
    Build.insert(E);
  return std::move(Build).build(GT);
}

LRTable LRTable::buildSLR(const Grammar &G) {
  Builder Build;
  auto Graph = LRGraph::buildLR0(G);
  for (const auto &T : Graph.edges()) {
    Action Act = isToken(T.Label) ? Action::shift(T.Dst) : Action::goTo(T.Dst);
    Build.insert({T.Src, T.Label, Act});
  }
  assert(Graph.states().size() <= (1 << StateBits) &&
         "Graph states execceds the maximum limit!");
  auto FollowSets = followSets(G);
  for (StateID SID = 0; SID < Graph.states().size(); ++SID) {
    for (const Item &I : Graph.states()[SID].Items) {
      // If we've just parsed the start symbol, we can accept the input.
      if (G.lookupRule(I.rule()).Target == G.startSymbol() && !I.hasNext()) {
        Build.insert({SID, tokenSymbol(tok::eof), Action::accept(I.rule())});
        continue;
      }
      if (!I.hasNext()) {
        // If we've reached the end of a rule A := ..., then we can reduce if
        // the next token is in the follow set of A".
        for (SymbolID Follow : FollowSets[G.lookupRule(I.rule()).Target]) {
          assert(isToken(Follow));
          Build.insert({SID, Follow, Action::reduce(I.rule())});
        }
      }
    }
  }
  return std::move(Build).build(G.table());
}

} // namespace pseudo
} // namespace syntax
} // namespace clang
