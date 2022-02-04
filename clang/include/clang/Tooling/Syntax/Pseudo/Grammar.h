//===--- Grammar.h - grammar used by clang pseudo parser  --------*- C++-*-===//
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
//  non-terminal or terminal, identified by a SymbolID.
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

#ifndef LLVM_CLANG_TOOLING_SYNTAX_GRAMMAR_H
#define LLVM_CLANG_TOOLING_SYNTAX_GRAMMAR_H

#include "clang/Basic/TokenKinds.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/StringRef.h"
#include <cstdint>
#include <vector>

namespace clang {
namespace syntax {
namespace pseudo {
// A SymbolID uniquely identifies a terminal/non-terminal symbol in a grammar.
// Non-terminal IDs are indexes into a table of non-terminal symbols.
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
  // lenth to 9 (this is the longest sequence in cxx grammar).
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

  llvm::ArrayRef<SymbolID> seq() const {
    return llvm::ArrayRef<SymbolID>(Sequence, Size);
  }
  friend bool operator==(const Rule &L, const Rule &R) {
    return L.Target == R.Target && L.seq() == R.seq();
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

  // Returns the SymbolID of the start symbol '_'.
  SymbolID startSymbol() const { return StartSymbol; };

  // Returns all rules of the given non-terminal symbol.
  llvm::ArrayRef<Rule> rulesFor(SymbolID SID) const;
  const Rule &lookupRule(RuleID RID) const;

  // Gets symbol (terminal or non-terminal) name.
  // Terminals have names like "," (kw_comma) or "OPERATOR" (kw_operator).
  llvm::StringRef symbolName(SymbolID) const;

  // Dumps the whole grammar.
  std::string dump() const;
  // Dumps a particular rule.
  std::string dumpRule(RuleID) const;
  // Dumps all rules of the given nonterminal symbol.
  std::string dumpRules(SymbolID) const;

  const GrammarTable &table() const { return *T; }

private:
  std::unique_ptr<GrammarTable> T;
  // The start symbol '_' of the augmented grammar.
  SymbolID StartSymbol;
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
  struct Nonterminal {
    std::string Name;
    // Corresponding rules that construct the non-terminal, it is a [start, end)
    // index range of the Rules table.
    struct {
      RuleID start;
      RuleID end;
    } RuleRange;
  };

  // The rules are sorted (and thus grouped) by target symbol.
  // RuleID is the index of the vector.
  std::vector<Rule> Rules;
  // A table of terminals (aka tokens). It correspond to the clang::Token.
  // clang::tok::TokenKind is the index of the table.
  std::vector<std::string> Terminals;
  // A table of nonterminals, sorted by name.
  // SymbolID is the index of the table.
  std::vector<Nonterminal> Nonterminals;
};

} // namespace pseudo
} // namespace syntax
} // namespace clang

#endif // LLVM_CLANG_TOOLING_SYNTAX_GRAMMAR_H
