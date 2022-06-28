//===--- LRGraph.h - Build an LR automaton  ------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  LR parsers are bottom-up parsers -- they scan the input from left to right,
//  and collect the right-hand side of a production rule (called handle) on top
//  of the stack, then replace (reduce) the handle with the nonterminal defined
//  by the production rule.
//
//  This file defines LRGraph, a deterministic handle-finding finite-state
//  automaton, which is a key component in LR parsers to recognize any of
//  handles in the grammar efficiently. We build the LR table (ACTION and GOTO
//  Table) based on the LRGraph.
//
//  LRGraph can be constructed for any context-free grammars.
//  Even for a LR-ambiguous grammar, we can construct a deterministic FSA, but
//  interpretation of the FSA is nondeterministic -- we might in a state where
//  we can continue searching an handle and identify a handle (called
//  shift/reduce conflicts), or identify more than one handle (callled
//  reduce/reduce conflicts).
//
//  LRGraph is a common model for all variants of LR automatons, from the most
//  basic one LR(0), the powerful SLR(1), LR(1) which uses a one-token lookahead
//  in making decisions.
//===----------------------------------------------------------------------===//

#ifndef CLANG_PSEUDO_GRAMMAR_LRGRAPH_H
#define CLANG_PSEUDO_GRAMMAR_LRGRAPH_H

#include "clang-pseudo/grammar/Grammar.h"
#include "llvm/ADT/Hashing.h"
#include <vector>

namespace clang {
namespace pseudo {

// An LR item -- a grammar rule with a dot at some position of the body.
// e.g. a production rule A := X Y yields 3 items:
//   A := . X Y
//   A := X . Y
//   A := X Y .
// An item indicates how much of a production rule has been recognized at a
// position (described by dot), for example, A := X . Y indicates that we have
// recognized the X part from the input, and we hope next to see the input
// derivable from Y.
class Item {
public:
  static Item start(RuleID ID, const Grammar &G) {
    Item I;
    I.RID = ID;
    I.RuleLength = G.lookupRule(ID).Size;
    return I;
  }
  static Item sentinel(RuleID ID) {
    Item I;
    I.RID = ID;
    return I;
  }

  RuleID rule() const { return RID; }
  uint8_t dot() const { return DotPos; }

  bool hasNext() const { return DotPos < RuleLength; }
  SymbolID next(const Grammar &G) const {
    assert(hasNext());
    return G.lookupRule(RID).Sequence[DotPos];
  }

  Item advance() const {
    assert(hasNext());
    Item I = *this;
    ++I.DotPos;
    return I;
  }

  std::string dump(const Grammar &G) const;

  bool operator==(const Item &I) const {
    return DotPos == I.DotPos && RID == I.RID;
  }
  bool operator<(const Item &I) const {
    return std::tie(RID, DotPos) < std::tie(I.RID, I.DotPos);
  }
  friend llvm::hash_code hash_value(const Item &I) {
    return llvm::hash_combine(I.RID, I.DotPos);
  }

private:
  RuleID RID = 0;
  uint8_t DotPos = 0;
  uint8_t RuleLength = 0; // the length of rule body.
};

// A state represents a node in the LR automaton graph. It is an item set, which
// contains all possible rules that the LR parser may be parsing in that state.
//
// Conceptually, If we knew in advance what we're parsing, at any point we're
// partway through parsing a production, sitting on a stack of partially parsed
// productions. But because we don't know, there could be *several* productions
// we're partway through. The set of possibilities is the parser state, and we
// precompute all the transitions between these states.
struct State {
  // A full set of items (including non-kernel items) representing the state,
  // in a canonical order (see SortByNextSymbol in the cpp file).
  std::vector<Item> Items;

  std::string dump(const Grammar &G, unsigned Indent = 0) const;
};

// LRGraph is a deterministic finite state automaton for LR parsing.
//
// Intuitively, an LR automaton is a transition graph. The graph has a
// collection of nodes, called States. Each state corresponds to a particular
// item set, which represents a condition that could occur during the process of
// parsing a production. Edges are directed from one state to another. Each edge
// is labeled by a grammar symbol (terminal or nonterminal).
//
// LRGraph is used to construct the LR parsing table which is a core
// data-structure driving the LR parser.
class LRGraph {
public:
  // StateID is the index in States table.
  using StateID = uint16_t;

  // Constructs an LR(0) automaton.
  static LRGraph buildLR0(const Grammar &);

  // An edge in the LR graph, it represents a transition in the LR automaton.
  // If the parser is at state Src, with a lookahead Label, then it
  // transits to state Dst.
  struct Edge {
    StateID Src, Dst;
    SymbolID Label;
  };

  llvm::ArrayRef<State> states() const { return States; }
  llvm::ArrayRef<Edge> edges() const { return Edges; }
  llvm::ArrayRef<std::pair<SymbolID, StateID>> startStates() const {
    return StartStates;
  }

  std::string dumpForTests(const Grammar &) const;

private:
  LRGraph(std::vector<State> States, std::vector<Edge> Edges,
          std::vector<std::pair<SymbolID, StateID>> StartStates)
      : States(std::move(States)), Edges(std::move(Edges)),
        StartStates(std::move(StartStates)) {}

  std::vector<State> States;
  std::vector<Edge> Edges;
  std::vector<std::pair<SymbolID, StateID>> StartStates;
};

} // namespace pseudo
} // namespace clang

namespace llvm {
// Support clang::pseudo::Item as DenseMap keys.
template <> struct DenseMapInfo<clang::pseudo::Item> {
  static inline clang::pseudo::Item getEmptyKey() {
    return clang::pseudo::Item::sentinel(-1);
  }
  static inline clang::pseudo::Item getTombstoneKey() {
    return clang::pseudo::Item::sentinel(-2);
  }
  static unsigned getHashValue(const clang::pseudo::Item &I) {
    return hash_value(I);
  }
  static bool isEqual(const clang::pseudo::Item &LHS,
                      const clang::pseudo::Item &RHS) {
    return LHS == RHS;
  }
};
} // namespace llvm

#endif // CLANG_PSEUDO_GRAMMAR_LRGRAPH_H
