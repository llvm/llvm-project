//===--- LRTableBuild.cpp - Build a LRTable from LRGraph ---------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang-pseudo/grammar/Grammar.h"
#include "clang-pseudo/grammar/LRGraph.h"
#include "clang-pseudo/grammar/LRTable.h"
#include "clang/Basic/TokenKinds.h"
#include "llvm/ADT/SmallSet.h"
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

struct LRTable::Builder {
  std::vector<std::pair<SymbolID, StateID>> StartStates;
  llvm::DenseSet<Entry> Entries;
  llvm::DenseMap<StateID, llvm::SmallSet<RuleID, 4>> Reduces;
  std::vector<llvm::DenseSet<SymbolID>> FollowSets;

  LRTable build(unsigned NumStates, unsigned NumNonterminals) && {
    LRTable Table;
    Table.StartStates = std::move(StartStates);

    // Compile the goto and shift actions into transition tables.
    llvm::DenseMap<unsigned, SymbolID> Gotos;
    llvm::DenseMap<unsigned, SymbolID> Shifts;
    for (const auto &E : Entries) {
      if (E.Act.kind() == Action::Shift)
        Shifts.try_emplace(shiftIndex(E.State, E.Symbol, NumStates),
                           E.Act.getShiftState());
      else if (E.Act.kind() == Action::GoTo)
        Gotos.try_emplace(gotoIndex(E.State, E.Symbol, NumStates),
                          E.Act.getGoToState());
    }
    Table.Shifts = TransitionTable(Shifts, NumStates * NumTerminals);
    Table.Gotos = TransitionTable(Gotos, NumStates * NumNonterminals);

    // Compile the follow sets into a bitmap.
    Table.FollowSets.resize(tok::NUM_TOKENS * FollowSets.size());
    for (SymbolID NT = 0; NT < FollowSets.size(); ++NT)
      for (SymbolID Follow : FollowSets[NT])
        Table.FollowSets.set(NT * tok::NUM_TOKENS + symbolToToken(Follow));

    // Store the reduce actions in a vector partitioned by state.
    Table.ReduceOffset.reserve(NumStates + 1);
    std::vector<RuleID> StateRules;
    for (StateID S = 0; S < NumStates; ++S) {
      Table.ReduceOffset.push_back(Table.Reduces.size());
      auto It = Reduces.find(S);
      if (It == Reduces.end())
        continue;
      Table.Reduces.insert(Table.Reduces.end(), It->second.begin(),
                           It->second.end());
      std::sort(Table.Reduces.begin() + Table.ReduceOffset.back(),
                Table.Reduces.end());
    }
    Table.ReduceOffset.push_back(Table.Reduces.size());

    return Table;
  }
};

LRTable LRTable::buildForTests(const Grammar &G, llvm::ArrayRef<Entry> Entries,
                               llvm::ArrayRef<ReduceEntry> Reduces) {
  StateID MaxState = 0;
  for (const auto &Entry : Entries) {
    MaxState = std::max(MaxState, Entry.State);
    if (Entry.Act.kind() == LRTable::Action::Shift)
      MaxState = std::max(MaxState, Entry.Act.getShiftState());
    if (Entry.Act.kind() == LRTable::Action::GoTo)
      MaxState = std::max(MaxState, Entry.Act.getGoToState());
  }
  Builder Build;
  Build.Entries.insert(Entries.begin(), Entries.end());
  for (const ReduceEntry &E : Reduces)
    Build.Reduces[E.State].insert(E.Rule);
  Build.FollowSets = followSets(G);
  return std::move(Build).build(/*NumStates=*/MaxState + 1,
                                G.table().Nonterminals.size());
}

LRTable LRTable::buildSLR(const Grammar &G) {
  auto Graph = LRGraph::buildLR0(G);
  Builder Build;
  Build.StartStates = Graph.startStates();
  for (const auto &T : Graph.edges()) {
    Action Act = isToken(T.Label) ? Action::shift(T.Dst) : Action::goTo(T.Dst);
    Build.Entries.insert({T.Src, T.Label, Act});
  }
  Build.FollowSets = followSets(G);
  assert(Graph.states().size() <= (1 << StateBits) &&
         "Graph states execceds the maximum limit!");
  // Add reduce actions.
  for (StateID SID = 0; SID < Graph.states().size(); ++SID) {
    for (const Item &I : Graph.states()[SID].Items) {
      // If we've just parsed the start symbol, this means we successfully parse
      // the input. We don't add the reduce action of `_ := start_symbol` in the
      // LRTable (the GLR parser handles it specifically).
      if (G.lookupRule(I.rule()).Target == G.underscore() && !I.hasNext())
        continue;
      if (!I.hasNext())
        // If we've reached the end of a rule A := ..., then we can reduce if
        // the next token is in the follow set of A.
        Build.Reduces[SID].insert(I.rule());
    }
  }
  return std::move(Build).build(Graph.states().size(),
                                G.table().Nonterminals.size());
}

} // namespace pseudo
} // namespace clang
