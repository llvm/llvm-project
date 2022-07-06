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

namespace clang {
namespace pseudo {

LRTable LRTable::Builder::build() && {
  assert(NumNonterminals != 0 && "Set NumNonterminals or init with grammar");
  LRTable Table;

  // Count number of states: every state has to be reachable somehow.
  StateID MaxState = 0;
  for (const auto &Entry : StartStates)
    MaxState = std::max(MaxState, Entry.second);
  for (const auto &Entry : Transition)
    MaxState = std::max(MaxState, Entry.second);
  unsigned NumStates = MaxState + 1;

  Table.StartStates = std::move(StartStates);

  // Compile the goto and shift actions into transition tables.
  llvm::DenseMap<unsigned, SymbolID> Gotos;
  llvm::DenseMap<unsigned, SymbolID> Shifts;
  for (const auto &E : Transition) {
    if (isToken(E.first.second))
      Shifts.try_emplace(shiftIndex(E.first.first, E.first.second, NumStates),
                         E.second);
    else
      Gotos.try_emplace(gotoIndex(E.first.first, E.first.second, NumStates),
                        E.second);
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
    auto It = Reduce.find(S);
    if (It == Reduce.end())
      continue;
    Table.Reduces.insert(Table.Reduces.end(), It->second.begin(),
                         It->second.end());
    std::sort(Table.Reduces.begin() + Table.ReduceOffset.back(),
              Table.Reduces.end());
  }
  Table.ReduceOffset.push_back(Table.Reduces.size());

  // Error recovery entries: sort (no dups already), and build offset lookup.
  llvm::sort(Recoveries, [&](const auto &L, const auto &R) {
    return std::tie(L.first, L.second.Result, L.second.Strategy) <
           std::tie(R.first, R.second.Result, R.second.Strategy);
  });
  Table.Recoveries.reserve(Recoveries.size());
  for (const auto &R : Recoveries)
    Table.Recoveries.push_back({R.second.Strategy, R.second.Result});
  Table.RecoveryOffset = std::vector<uint32_t>(NumStates + 1, 0);
  unsigned SortedIndex = 0;
  for (StateID State = 0; State < NumStates; ++State) {
    Table.RecoveryOffset[State] = SortedIndex;
    while (SortedIndex < Recoveries.size() &&
           Recoveries[SortedIndex].first == State)
      SortedIndex++;
  }
  Table.RecoveryOffset[NumStates] = SortedIndex;
  assert(SortedIndex == Recoveries.size());

  return Table;
}

LRTable LRTable::buildSLR(const Grammar &G) {
  auto Graph = LRGraph::buildLR0(G);
  Builder Build(G);
  Build.StartStates = Graph.startStates();
  for (const auto &T : Graph.edges())
    Build.Transition.try_emplace({T.Src, T.Label}, T.Dst);
  for (const auto &Entry : Graph.recoveries())
    Build.Recoveries.push_back(
        {Entry.Src, Recovery{Entry.Strategy, Entry.Result}});
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
        Build.Reduce[SID].insert(I.rule());
    }
  }
  return std::move(Build).build();
}

} // namespace pseudo
} // namespace clang
