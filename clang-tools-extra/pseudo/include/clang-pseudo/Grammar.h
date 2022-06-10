//===--- Grammar.h - grammar used by clang pseudoparser  ---------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  This file defines base structures for parsing & modeling a grammar for a
//  programming language:
//
//    # This is a fake C++ BNF grammar
//    _ := translation-unit
//    translation-unit := declaration-seq_opt
//    declaration-seq := declaration
//    declaration-seq := declaration-seq declaration
//
//  A grammar formally describes a language, and it is constructed by a set of
//  production rules. A rule is of BNF form (AAA := BBB CCC). A symbol is either
//  nonterminal or terminal, identified by a SymbolID.
//
//  Annotations are supported in a syntax form of [key=value]. They specify
//  attributes which are associated with either a grammar symbol (on the
//  right-hand side of the symbol) or a grammar rule (at the end of the rule
//  body).
//  Attributes provide a way to inject custom code into the GLR parser. Each
//  unique attribute value creates an extension point (identified by ExtensionID
//  ), and an extension point corresponds to a piece of native code. For
//  example, C++ grammar has a rule:
//
//    contextual-override := IDENTIFIER [guard=Override]
//
//  GLR parser only conducts the reduction of the rule if the IDENTIFIER
//  content is `override`. This Override guard is implemented in CXX.cpp by
//  binding the ExtensionID for the `Override` value to a specific C++ function
//  that performs the check.
//
//  Notions about the BNF grammar:
//  - "_" is the start symbol of the augmented grammar;
//  - single-line comment is supported, starting with a #
//  - A rule describes how a nonterminal (left side of :=) is constructed, and
//    it is *per line* in the grammar file
//  - Terminals (also called tokens) correspond to the clang::TokenKind; they
//    are written in the grammar like "IDENTIFIER", "USING", "+"
//  - Nonterminals are specified with "lower-case" names in the grammar; they
//    shouldn't be nullable (has an empty sequence)
//  - optional symbols are supported (specified with a _opt suffix), and they
//    will be eliminated during the grammar parsing stage
//
//===----------------------------------------------------------------------===//

#ifndef CLANG_PSEUDO_GRAMMAR_H
#define CLANG_PSEUDO_GRAMMAR_H

#include "clang/Basic/TokenKinds.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/StringRef.h"
#include <cstdint>
#include <vector>

namespace clang {
namespace pseudo {
// A SymbolID uniquely identifies a terminal/nonterminal symbol in a grammar.
// nonterminal IDs are indexes into a table of nonterminal symbols.
// Terminal IDs correspond to the clang TokenKind enum.
using SymbolID = uint16_t;
// SymbolID is only 12 bits wide.
// There are maximum 2^11 terminals (aka tokens) and 2^11 nonterminals.
static constexpr uint16_t SymbolBits = 12;
static constexpr uint16_t NumTerminals = tok::NUM_TOKENS;
// SymbolIDs with the top bit set are tokens/terminals.
static constexpr SymbolID TokenFlag = 1 << (SymbolBits - 1);
inline bool isToken(SymbolID ID) { return ID & TokenFlag; }
inline bool isNonterminal(SymbolID ID) { return !isToken(ID); }
// The terminals are always the clang tok::TokenKind (not all are used).
inline tok::TokenKind symbolToToken(SymbolID SID) {
  assert(isToken(SID));
  SID &= ~TokenFlag;
  assert(SID < NumTerminals);
  return static_cast<tok::TokenKind>(SID);
}
inline SymbolID tokenSymbol(tok::TokenKind TK) {
  return TokenFlag | static_cast<SymbolID>(TK);
}

// An extension is a piece of native code specific to a grammar that modifies
// the behavior of annotated rules. One ExtensionID is assigned for each unique
// attribute value (all attributes share a namespace).
using ExtensionID = uint16_t;

// A RuleID uniquely identifies a production rule in a grammar.
// It is an index into a table of rules.
using RuleID = uint16_t;
// There are maximum 2^12 rules.
static constexpr unsigned RuleBits = 12;

// Represent a production rule in the grammar, e.g.
//   expression := a b c
//   ^Target       ^Sequence
struct Rule {
  Rule(SymbolID Target, llvm::ArrayRef<SymbolID> Seq);

  // We occupy 4 bits for the sequence, in theory, it can be at most 2^4 tokens
  // long, however, we're stricter in order to reduce the size, we limit the max
  // length to 9 (this is the longest sequence in cxx grammar).
  static constexpr unsigned SizeBits = 4;
  static constexpr unsigned MaxElements = 9;
  static_assert(MaxElements <= (1 << SizeBits), "Exceeds the maximum limit");
  static_assert(SizeBits + SymbolBits <= 16,
                "Must be able to store symbol ID + size efficiently");

  // 16 bits for target symbol and size of sequence:
  // SymbolID : 12 | Size : 4
  SymbolID Target : SymbolBits;
  uint8_t Size : SizeBits; // Size of the Sequence
  SymbolID Sequence[MaxElements];

  // A guard extension controls whether a reduction of a rule will be conducted
  // by the GLR parser.
  // 0 is sentinel unset extension ID, indicating there is no guard extension
  // being set for this rule.
  ExtensionID Guard = 0;

  llvm::ArrayRef<SymbolID> seq() const {
    return llvm::ArrayRef<SymbolID>(Sequence, Size);
  }
  friend bool operator==(const Rule &L, const Rule &R) {
    return L.Target == R.Target && L.seq() == R.seq() && L.Guard == R.Guard;
  }
};

struct GrammarTable;

// Grammar that describes a programming language, e.g. C++. It represents the
// contents of the specified grammar.
// It is a building block for constructing a table-based parser.
class Grammar {
public:
  explicit Grammar(std::unique_ptr<GrammarTable>);

  // Parses grammar from a BNF file.
  // Diagnostics emitted during parsing are stored in Diags.
  static std::unique_ptr<Grammar> parseBNF(llvm::StringRef BNF,
                                           std::vector<std::string> &Diags);

  // Returns the SymbolID of the symbol '_'.
  SymbolID underscore() const { return Underscore; };

  // Returns all rules of the given nonterminal symbol.
  llvm::ArrayRef<Rule> rulesFor(SymbolID SID) const;
  const Rule &lookupRule(RuleID RID) const;

  // Gets symbol (terminal or nonterminal) name.
  // Terminals have names like "," (kw_comma) or "OPERATOR" (kw_operator).
  llvm::StringRef symbolName(SymbolID) const;

  // Lookup the SymbolID of the nonterminal symbol by Name.
  llvm::Optional<SymbolID> findNonterminal(llvm::StringRef Name) const;

  // Dumps the whole grammar.
  std::string dump() const;
  // Dumps a particular rule.
  std::string dumpRule(RuleID) const;
  // Dumps all rules of the given nonterminal symbol.
  std::string dumpRules(SymbolID) const;

  const GrammarTable &table() const { return *T; }

private:
  std::unique_ptr<GrammarTable> T;
  // The symbol ID of '_'. (In the LR literature, this is the start symbol of
  // the augmented grammar.)
  SymbolID Underscore;
};
// For each nonterminal X, computes the set of terminals that begin strings
// derived from X. (Known as FIRST sets in grammar-based parsers).
std::vector<llvm::DenseSet<SymbolID>> firstSets(const Grammar &);
// For each nonterminal X, computes the set of terminals that could immediately
// follow X. (Known as FOLLOW sets in grammar-based parsers).
std::vector<llvm::DenseSet<SymbolID>> followSets(const Grammar &);

// Storage for the underlying data of the Grammar.
// It can be constructed dynamically (from compiling BNF file) or statically
// (a compiled data-source).
struct GrammarTable {
  GrammarTable();

  struct Nonterminal {
    std::string Name;
    // Corresponding rules that construct the nonterminal, it is a [Start, End)
    // index range of the Rules table.
    struct {
      RuleID Start;
      RuleID End;
    } RuleRange;
  };

  // RuleID is an index into this table of rule definitions.
  //
  // Rules with the same target symbol (LHS) are grouped into a single range.
  // The relative order of different target symbols is *not* by SymbolID, but
  // rather a topological sort: if S := T then the rules producing T have lower
  // RuleIDs than rules producing S.
  // (This strange order simplifies the GLR parser: for a given token range, if
  // we reduce in increasing RuleID order then we need never backtrack --
  // prerequisite reductions are reached before dependent ones).
  std::vector<Rule> Rules;
  // A table of terminals (aka tokens). It corresponds to the clang::Token.
  // clang::tok::TokenKind is the index of the table.
  llvm::ArrayRef<std::string> Terminals;
  // A table of nonterminals, sorted by name.
  // SymbolID is the index of the table.
  std::vector<Nonterminal> Nonterminals;
  // A table of attribute values, sorted by name.
  // ExtensionID is the index of the table.
  std::vector<std::string> AttributeValues;
};

} // namespace pseudo
} // namespace clang

#endif // CLANG_PSEUDO_GRAMMAR_H
