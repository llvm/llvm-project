//===--- LRTable.h - Define LR Parsing Table ---------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  The LRTable (referred as LR parsing table in the LR literature) is the core
//  component in LR parsers, it drives the LR parsers by specifying an action to
//  take given the current state on the top of the stack and the current
//  lookahead token.
//
//  The LRTable can be described as a matrix where the rows represent
//  the states of the LR graph, the columns represent the symbols of the
//  grammar, and each entry of the matrix (called action) represents a
//  state transition in the graph.
//
//  Typically, based on the category of the grammar symbol, the LRTable is
//  broken into two logically separate tables:
//    - ACTION table with terminals as columns -- e.g. ACTION[S, a] specifies
//      next action (shift/reduce) on state S under a lookahead terminal a
//    - GOTO table with nonterminals as columns -- e.g. GOTO[S, X] specifies
//      the state which we transist to from the state S with the nonterminal X
//
//  LRTable is *performance-critial* as it is consulted frequently during a
//  parse. In general, LRTable is very sparse (most of the entries are empty).
//  For example, for the C++ language, the SLR table has ~1500 states and 650
//  symbols which results in a matrix having 975K entries, ~90% of entries are
//  empty.
//
//  This file implements a speed-and-space-efficient LRTable.
//
//===----------------------------------------------------------------------===//

#ifndef CLANG_PSEUDO_GRAMMAR_LRTABLE_H
#define CLANG_PSEUDO_GRAMMAR_LRTABLE_H

#include "clang-pseudo/grammar/Grammar.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/Support/Capacity.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdint>
#include <vector>

namespace clang {
namespace pseudo {

// Represents the LR parsing table, which can efficiently the question "what is
// the next step given the lookahead token and current state on top of the
// stack?".
//
// This is a dense implementation, which only takes an amount of space that is
// proportional to the number of non-empty entries in the table.
//
// Unlike the typical LR parsing table which allows at most one available action
// per entry, conflicted actions are allowed in LRTable. The LRTable is designed
// to be used in nondeterministic LR parsers (e.g. GLR).
class LRTable {
public:
  // StateID is only 13 bits wide.
  using StateID = uint16_t;
  static constexpr unsigned StateBits = 13;

  // Action represents the terminal and nonterminal actions, it combines the
  // entry of the ACTION and GOTO tables from the LR literature.
  //
  // FIXME: as we move away from a homogeneous table structure shared between
  // action types, this class becomes less useful. Remove it.
  class Action {
  public:
    enum Kind : uint8_t {
      Sentinel = 0,
      // Terminal actions, corresponding to entries of ACTION table.

      // Shift to state n: move forward with the lookahead, and push state n
      // onto the state stack.
      // A shift is a forward transition, and the value n is the next state that
      // the parser is to enter.
      Shift,

      // NOTE: there are no typical accept actions in the LRtable, accept
      // actions are handled specifically in the parser -- if the parser
      // reaches to a target state (which is goto(StartState, StartSymbol)) at
      // the EOF token after a reduce, this indicates the input has been parsed
      // as the StartSymbol successfully.

      // Nonterminal actions, corresponding to entry of GOTO table.

      // Go to state n: push state n onto the state stack.
      // Similar to Shift, but it is a nonterminal forward transition.
      GoTo,
    };

    static Action goTo(StateID S) { return Action(GoTo, S); }
    static Action shift(StateID S) { return Action(Shift, S); }
    static Action sentinel() { return Action(Sentinel, 0); }

    StateID getShiftState() const {
      assert(kind() == Shift);
      return Value;
    }
    StateID getGoToState() const {
      assert(kind() == GoTo);
      return Value;
    }
    Kind kind() const { return static_cast<Kind>(K); }

    bool operator==(const Action &L) const { return opaque() == L.opaque(); }
    uint16_t opaque() const { return K << ValueBits | Value; };

  private:
    Action(Kind K1, unsigned Value) : K(K1), Value(Value) {}
    static constexpr unsigned ValueBits = StateBits;
    static constexpr unsigned KindBits = 3;
    static_assert(ValueBits >= RuleBits, "Value must be able to store RuleID");
    static_assert(KindBits + ValueBits <= 16,
                  "Must be able to store kind and value efficiently");
    uint16_t K : KindBits;
    // Either StateID or RuleID, depending on the Kind.
    uint16_t Value : ValueBits;
  };

  // Returns the state after we reduce a nonterminal.
  // Expected to be called by LR parsers.
  // If the nonterminal is invalid here, returns None.
  llvm::Optional<StateID> getGoToState(StateID State,
                                       SymbolID Nonterminal) const {
    return Gotos.get(gotoIndex(State, Nonterminal, numStates()));
  }
  // Returns the state after we shift a terminal.
  // Expected to be called by LR parsers.
  // If the terminal is invalid here, returns None.
  llvm::Optional<StateID> getShiftState(StateID State,
                                        SymbolID Terminal) const {
    return Shifts.get(shiftIndex(State, Terminal, numStates()));
  }

  // Returns the possible reductions from a state.
  //
  // These are not keyed by a lookahead token. Instead, call canFollow() to
  // check whether a reduction should apply in the current context:
  //   for (RuleID R : LR.getReduceRules(S)) {
  //     if (!LR.canFollow(G.lookupRule(R).Target, NextToken))
  //       continue;
  //     // ...apply reduce...
  //   }
  llvm::ArrayRef<RuleID> getReduceRules(StateID State) const {
    assert(State + 1u < ReduceOffset.size());
    return llvm::makeArrayRef(Reduces.data() + ReduceOffset[State],
                              Reduces.data() + ReduceOffset[State+1]);
  }
  // Returns whether Terminal can follow Nonterminal in a valid source file.
  bool canFollow(SymbolID Nonterminal, SymbolID Terminal) const {
    assert(isToken(Terminal));
    assert(isNonterminal(Nonterminal));
    return FollowSets.test(tok::NUM_TOKENS * Nonterminal +
                           symbolToToken(Terminal));
  }

  // Returns the state from which the LR parser should start to parse the input
  // tokens as the given StartSymbol.
  //
  // In LR parsing, the start state of `translation-unit` corresponds to
  // `_ := • translation-unit`.
  //
  // Each start state responds to **a** single grammar rule like `_ := start`.
  // REQUIRE: The given StartSymbol must exist in the grammar (in a form of
  //          `_ := start`).
  StateID getStartState(SymbolID StartSymbol) const;

  size_t bytes() const {
    return sizeof(*this) + Gotos.bytes() + Shifts.bytes() +
           llvm::capacity_in_bytes(Reduces) +
           llvm::capacity_in_bytes(ReduceOffset) +
           llvm::capacity_in_bytes(FollowSets);
  }

  std::string dumpStatistics() const;
  std::string dumpForTests(const Grammar &G) const;

  // Build a SLR(1) parsing table.
  static LRTable buildSLR(const Grammar &G);

  struct Builder;
  // Represents an entry in the table, used for building the LRTable.
  struct Entry {
    StateID State;
    SymbolID Symbol;
    Action Act;
  };
  struct ReduceEntry {
    StateID State;
    RuleID Rule;
  };
  // Build a specifid table for testing purposes.
  static LRTable buildForTests(const Grammar &G, llvm::ArrayRef<Entry>,
                               llvm::ArrayRef<ReduceEntry>);

private:
  unsigned numStates() const { return ReduceOffset.size() - 1; }

  // A map from unsigned key => StateID, used to store actions.
  // The keys should be sequential but the values are somewhat sparse.
  //
  // In practice, the keys encode (origin state, symbol) pairs, and the values
  // are the state we should move to after seeing that symbol.
  //
  // We store one bit for presence/absence of the value for each key.
  // At every 64th key, we store the offset into the table of values.
  //   e.g. key 0x500 is checkpoint 0x500/64 = 20
  //                     Checkpoints[20] = 34
  //        get(0x500) = Values[34]                (assuming it has a value)
  // To look up values in between, we count the set bits:
  //        get(0x509) has a value if HasValue[20] & (1<<9)
  //        #values between 0x500 and 0x509: popcnt(HasValue[20] & (1<<9 - 1))
  //        get(0x509) = Values[34 + popcnt(...)]
  //
  // Overall size is 1.25 bits/key + 16 bits/value.
  // Lookup is constant time with a low factor (no hashing).
  class TransitionTable {
    using Word = uint64_t;
    constexpr static unsigned WordBits = CHAR_BIT * sizeof(Word);

    std::vector<StateID> Values;
    std::vector<Word> HasValue;
    std::vector<uint16_t> Checkpoints;

  public:
    TransitionTable() = default;
    TransitionTable(const llvm::DenseMap<unsigned, StateID> &Entries,
                    unsigned NumKeys) {
      assert(
          Entries.size() <
              std::numeric_limits<decltype(Checkpoints)::value_type>::max() &&
          "16 bits too small for value offsets!");
      unsigned NumWords = (NumKeys + WordBits - 1) / WordBits;
      HasValue.resize(NumWords, 0);
      Checkpoints.reserve(NumWords);
      Values.reserve(Entries.size());
      for (unsigned I = 0; I < NumKeys; ++I) {
        if ((I % WordBits) == 0)
          Checkpoints.push_back(Values.size());
        auto It = Entries.find(I);
        if (It != Entries.end()) {
          HasValue[I / WordBits] |= (Word(1) << (I % WordBits));
          Values.push_back(It->second);
        }
      }
    }

    llvm::Optional<StateID> get(unsigned Key) const {
      // Do we have a value for this key?
      Word KeyMask = Word(1) << (Key % WordBits);
      unsigned KeyWord = Key / WordBits;
      if ((HasValue[KeyWord] & KeyMask) == 0)
        return llvm::None;
      // Count the number of values since the checkpoint.
      Word BelowKeyMask = KeyMask - 1;
      unsigned CountSinceCheckpoint =
          llvm::countPopulation(HasValue[KeyWord] & BelowKeyMask);
      // Find the value relative to the last checkpoint.
      return Values[Checkpoints[KeyWord] + CountSinceCheckpoint];
    }

    unsigned size() const { return Values.size(); }

    size_t bytes() const {
      return llvm::capacity_in_bytes(HasValue) +
             llvm::capacity_in_bytes(Values) +
             llvm::capacity_in_bytes(Checkpoints);
    }
  };
  // Shift and Goto tables are keyed by encoded (State, Symbol).
  static unsigned shiftIndex(StateID State, SymbolID Terminal,
                             unsigned NumStates) {
    return NumStates * symbolToToken(Terminal) + State;
  }
  static unsigned gotoIndex(StateID State, SymbolID Nonterminal,
                            unsigned NumStates) {
    assert(isNonterminal(Nonterminal));
    return NumStates * Nonterminal + State;
  }
  TransitionTable Shifts;
  TransitionTable Gotos;

  // A sorted table, storing the start state for each target parsing symbol.
  std::vector<std::pair<SymbolID, StateID>> StartStates;

  // Given a state ID S, the half-open range of Reduces is
  // [ReduceOffset[S], ReduceOffset[S+1])
  std::vector<uint32_t> ReduceOffset;
  std::vector<RuleID> Reduces;
  // Conceptually this is a bool[SymbolID][Token], each entry describing whether
  // the grammar allows the (nonterminal) symbol to be followed by the token.
  //
  // This is flattened by encoding the (SymbolID Nonterminal, tok::Kind Token)
  // as an index: Nonterminal * NUM_TOKENS + Token.
  llvm::BitVector FollowSets;
};
llvm::raw_ostream &operator<<(llvm::raw_ostream &, const LRTable::Action &);

} // namespace pseudo
} // namespace clang

#endif // CLANG_PSEUDO_GRAMMAR_LRTABLE_H
