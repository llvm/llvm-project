//===--- LRGraph.cpp - -------------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Tooling/Syntax/Pseudo/LRGraph.h"
#include "clang/Tooling/Syntax/Pseudo/Grammar.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"

using ItemSet = std::vector<clang::syntax::pseudo::Item>;

namespace llvm {
// Support clang::syntax::pseudo::Item as DenseMap keys.
template <> struct DenseMapInfo<ItemSet> {
  static inline ItemSet getEmptyKey() {
    return {DenseMapInfo<clang::syntax::pseudo::Item>::getEmptyKey()};
  }
  static inline ItemSet getTombstoneKey() {
    return {DenseMapInfo<clang::syntax::pseudo::Item>::getTombstoneKey()};
  }
  static unsigned getHashValue(const ItemSet &I) {
    return llvm::hash_combine_range(I.begin(), I.end());
  }
  static bool isEqual(const ItemSet &LHS, const ItemSet &RHS) {
    return LHS == RHS;
  }
};
} // namespace llvm

namespace clang {
namespace syntax {
namespace pseudo {
namespace {

struct SortByNextSymbol {
  SortByNextSymbol(const Grammar &G) : G(G) {}
  bool operator()(const Item &L, const Item &R) {
    if (L.hasNext() && R.hasNext() && L.next(G) != R.next(G))
      return L.next(G) < R.next(G);
    if (L.hasNext() != R.hasNext())
      return L.hasNext() < R.hasNext(); //  a trailing dot is minimal.
    return L < R;
  }
  const Grammar &G;
};

// Computes a closure of the given item set S:
//  - extends the given S to contain all options for parsing next token;
//  - nonterminals after a dot are recursively expanded into the begin-state
//    of all production rules that produce that nonterminal;
//
// Given
//   Grammar rules = [ _ := E, E := E - T, E := T, T := n, T := ( E ) ]
//   Input = [ E := . T ]
// returns [ E :=  . T, T := . n, T := . ( E ) ]
State closure(ItemSet Queue, const Grammar &G) {
  llvm::DenseSet<Item> InQueue = {Queue.begin(), Queue.end()};
  // We reuse the passed-by-value Queue as the final result, as it's already
  // initialized to the right elements.
  size_t ItIndex = 0;
  while (ItIndex < Queue.size()) {
    const Item &ExpandingItem = Queue[ItIndex];
    ++ItIndex;
    if (!ExpandingItem.hasNext())
      continue;

    SymbolID NextSym = ExpandingItem.next(G);
    if (pseudo::isToken(NextSym))
      continue;
    auto RRange = G.table().Nonterminals[NextSym].RuleRange;
    for (RuleID RID = RRange.start; RID < RRange.end; ++RID) {
      Item NewItem = Item::start(RID, G);
      if (InQueue.insert(NewItem).second) // new
        Queue.push_back(std::move(NewItem));
    }
  }
  Queue.shrink_to_fit();
  llvm::sort(Queue, SortByNextSymbol(G));
  return {std::move(Queue)};
}

// Returns all next (with a dot advanced) kernel item sets, partitioned by the
// advanced symbol.
//
// Given
//  S = [ E := . a b, E := E . - T ]
// returns [
//   {id(a), [ E := a . b ]},
//   {id(-), [ E := E - . T ]}
// ]
std::vector<std::pair<SymbolID, ItemSet>>
nextAvailableKernelItems(const State &S, const Grammar &G) {
  std::vector<std::pair<SymbolID, ItemSet>> Results;
  llvm::ArrayRef<Item> AllItems = S.Items;
  AllItems = AllItems.drop_while([](const Item &I) { return !I.hasNext(); });
  while (!AllItems.empty()) {
    SymbolID AdvancedSymbol = AllItems.front().next(G);
    auto Batch = AllItems.take_while([AdvancedSymbol, &G](const Item &I) {
      assert(I.hasNext());
      return I.next(G) == AdvancedSymbol;
    });
    assert(!Batch.empty());
    AllItems = AllItems.drop_front(Batch.size());

    // Advance a dot over the Symbol.
    ItemSet Next;
    for (const Item &I : Batch)
      Next.push_back(I.advance());
    // sort the set to keep order determinism for hash computation.
    llvm::sort(Next);
    Results.push_back({AdvancedSymbol, std::move(Next)});
  }
  return Results;
}

} // namespace

std::string Item::dump(const Grammar &G) const {
  const auto &Rule = G.lookupRule(RID);
  auto ToNames = [&](llvm::ArrayRef<SymbolID> Syms) {
    std::vector<llvm::StringRef> Results;
    for (auto SID : Syms)
      Results.push_back(G.symbolName(SID));
    return Results;
  };
  return llvm::formatv("{0} := {1} • {2}", G.symbolName(Rule.Target),
                       llvm::join(ToNames(Rule.seq().take_front(DotPos)), " "),
                       llvm::join(ToNames(Rule.seq().drop_front(DotPos)), " "))
      .str();
}

std::string State::dump(const Grammar &G, unsigned Indent) const {
  std::string Result;
  llvm::raw_string_ostream OS(Result);
  for (const auto &Item : Items)
    OS.indent(Indent) << llvm::formatv("{0}\n", Item.dump(G));
  return OS.str();
}

std::string LRGraph::dumpForTests(const Grammar &G) const {
  std::string Result;
  llvm::raw_string_ostream OS(Result);
  OS << "States:\n";
  for (StateID ID = 0; ID < States.size(); ++ID) {
    OS << llvm::formatv("State {0}\n", ID);
    OS << States[ID].dump(G, /*Indent*/ 4);
  }
  for (const auto &E : Edges) {
    OS << llvm::formatv("{0} ->[{1}] {2}\n", E.Src, G.symbolName(E.Label),
                        E.Dst);
  }
  return OS.str();
}

LRGraph LRGraph::buildLR0(const Grammar &G) {
  class Builder {
  public:
    Builder(const Grammar &G) : G(G) {}

    // Adds a given state if not existed.
    std::pair<StateID, /*inserted*/ bool> insert(ItemSet KernelItems) {
      assert(llvm::is_sorted(KernelItems) &&
             "Item must be sorted before inserting to a hash map!");
      auto It = StatesIndex.find(KernelItems);
      if (It != StatesIndex.end())
        return {It->second, false};
      States.push_back(closure(KernelItems, G));
      StateID NextStateID = States.size() - 1;
      StatesIndex.insert({std::move(KernelItems), NextStateID});
      return {NextStateID, true};
    }

    void insertEdge(StateID Src, StateID Dst, SymbolID Label) {
      Edges.push_back({Src, Dst, Label});
    }

    // Returns a state with the given id.
    const State &find(StateID ID) const {
      assert(ID < States.size());
      return States[ID];
    }

    LRGraph build() && {
      States.shrink_to_fit();
      Edges.shrink_to_fit();
      return LRGraph(std::move(States), std::move(Edges));
    }

  private:
    // Key is the **kernel** item sets.
    llvm::DenseMap<ItemSet, /*index of States*/ size_t> StatesIndex;
    std::vector<State> States;
    std::vector<Edge> Edges;
    const Grammar &G;
  } Builder(G);

  std::vector<StateID> PendingStates;
  // Initialize states with the start symbol.
  auto RRange = G.table().Nonterminals[G.startSymbol()].RuleRange;
  for (RuleID RID = RRange.start; RID < RRange.end; ++RID) {
    auto StartState = std::vector<Item>{Item::start(RID, G)};
    auto Result = Builder.insert(std::move(StartState));
    assert(Result.second && "State must be new");
    PendingStates.push_back(Result.first);
  }

  while (!PendingStates.empty()) {
    auto CurrentStateID = PendingStates.back();
    PendingStates.pop_back();
    for (auto Next :
         nextAvailableKernelItems(Builder.find(CurrentStateID), G)) {
      auto Insert = Builder.insert(Next.second);
      if (Insert.second) // new state, insert to the pending queue.
        PendingStates.push_back(Insert.first);
      Builder.insertEdge(CurrentStateID, Insert.first, Next.first);
    }
  }
  return std::move(Builder).build();
}

} // namespace pseudo
} // namespace syntax
} // namespace clang
