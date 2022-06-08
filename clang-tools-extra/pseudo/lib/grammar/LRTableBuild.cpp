//===--- LRTableBuild.cpp - Build a LRTable from LRGraph ---------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang-pseudo/Grammar.h"
#include "clang-pseudo/LRGraph.h"
#include "clang-pseudo/LRTable.h"
#include "clang/Basic/TokenKinds.h"
#include <cstdint>

namespace llvm {
template <> struct DenseMapInfo<clang::pseudo::LRTable::Entry> {
  using Entry = clang::pseudo::LRTable::Entry;
  static inline Entry getEmptyKey() {
    static Entry E{static_cast<clang::pseudo::SymbolID>(-1), 0,
                   clang::pseudo::LRTable::Action::sentinel()};
    return E;
  }
  static inline Entry getTombstoneKey() {
    static Entry E{static_cast<clang::pseudo::SymbolID>(-2), 0,
                   clang::pseudo::LRTable::Action::sentinel()};
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
namespace pseudo {

class LRTable::Builder {
public:
  Builder(llvm::ArrayRef<std::pair<SymbolID, StateID>> StartStates)
      : StartStates(StartStates) {}

  bool insert(Entry E) { return Entries.insert(std::move(E)).second; }
  LRTable build(const GrammarTable &GT, unsigned NumStates) && {
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
    //  - StateOffset: [s0] = 0, [s1] = 2, [s2] = 3, [sentinel] = 4
    //  - Symbols:     [ b,   b,   a,  b]
    //    Actions:     [ s0, r0, acc, r1]
    //                   ~~~~~~ range for state 0
    //                           ~~~~ range for state 1
    //                                ~~ range for state 2
    // First step, we sort all entries by (State, Symbol, Action).
    std::vector<Entry> Sorted(Entries.begin(), Entries.end());
    llvm::sort(Sorted, [](const Entry &L, const Entry &R) {
      return std::forward_as_tuple(L.State, L.Symbol, L.Act.opaque()) <
             std::forward_as_tuple(R.State, R.Symbol, R.Act.opaque());
    });

    LRTable Table;
    Table.Actions.reserve(Sorted.size());
    Table.Symbols.reserve(Sorted.size());
    // We are good to finalize the States and Actions.
    for (const auto &E : Sorted) {
      Table.Actions.push_back(E.Act);
      Table.Symbols.push_back(E.Symbol);
    }
    // Initialize the terminal and nonterminal offset, all ranges are empty by
    // default.
    Table.StateOffset = std::vector<uint32_t>(NumStates + 1, 0);
    size_t SortedIndex = 0;
    for (StateID State = 0; State < Table.StateOffset.size(); ++State) {
      Table.StateOffset[State] = SortedIndex;
      while (SortedIndex < Sorted.size() && Sorted[SortedIndex].State == State)
        ++SortedIndex;
    }
    Table.StartStates = std::move(StartStates);
    return Table;
  }

private:
  llvm::DenseSet<Entry> Entries;
  std::vector<std::pair<SymbolID, StateID>> StartStates;
};

LRTable LRTable::buildForTests(const GrammarTable &GT,
                               llvm::ArrayRef<Entry> Entries) {
  StateID MaxState = 0;
  for (const auto &Entry : Entries)
    MaxState = std::max(MaxState, Entry.State);
  Builder Build({});
  for (const Entry &E : Entries)
    Build.insert(E);
  return std::move(Build).build(GT, /*NumStates=*/MaxState + 1);
}

LRTable LRTable::buildSLR(const Grammar &G) {
  auto Graph = LRGraph::buildLR0(G);
  Builder Build(Graph.startStates());
  for (const auto &T : Graph.edges()) {
    Action Act = isToken(T.Label) ? Action::shift(T.Dst) : Action::goTo(T.Dst);
    Build.insert({T.Src, T.Label, Act});
  }
  assert(Graph.states().size() <= (1 << StateBits) &&
         "Graph states execceds the maximum limit!");
  auto FollowSets = followSets(G);
  for (StateID SID = 0; SID < Graph.states().size(); ++SID) {
    for (const Item &I : Graph.states()[SID].Items) {
      // If we've just parsed the start symbol, this means we successfully parse
      // the input. We don't add the reduce action of `_ := start_symbol` in the
      // LRTable (the GLR parser handles it specifically).
      if (G.lookupRule(I.rule()).Target == G.underscore() && !I.hasNext())
        continue;
      if (!I.hasNext()) {
        // If we've reached the end of a rule A := ..., then we can reduce if
        // the next token is in the follow set of A.
        for (SymbolID Follow : FollowSets[G.lookupRule(I.rule()).Target]) {
          assert(isToken(Follow));
          Build.insert({SID, Follow, Action::reduce(I.rule())});
        }
      }
    }
  }
  return std::move(Build).build(G.table(), Graph.states().size());
}

} // namespace pseudo
} // namespace clang
