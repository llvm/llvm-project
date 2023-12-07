//===--- Forest.h - Parse forest, the output of the GLR parser ---*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A parse forest represents a set of possible parse trees efficiently, it is
// produced by the GLR parser.
//
// Despite the name, its data structure is a tree-like DAG with a single root.
// Multiple ways to parse the same tokens are presented as an ambiguous node
// with all possible interpretations as children.
// Common sub-parses are shared: if two interpretations both parse "1 + 1" as
// "expr := expr + expr", they will share a Sequence node representing the expr.
//
//===----------------------------------------------------------------------===//

#ifndef CLANG_PSEUDO_FOREST_H
#define CLANG_PSEUDO_FOREST_H

#include "clang-pseudo/Token.h"
#include "clang-pseudo/grammar/Grammar.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Allocator.h"
#include <cstdint>

namespace clang {
namespace pseudo {

// A node represents ways to parse a sequence of tokens, it interprets a fixed
// range of tokens as a fixed grammar symbol.
//
// There are different kinds of nodes, some nodes have "children" (stored in a
// trailing array) and have pointers to them. "Children" has different semantics
// depending on the node kinds. For an Ambiguous node, it means all
// possible interpretations; for a Sequence node, it means each symbol on the
// right hand side of the production rule.
//
// Since this is a node in a DAG, a node may have multiple parents. And a node
// doesn't have parent pointers.
class alignas(class ForestNode *) ForestNode {
public:
  class RecursiveIterator;
  enum Kind {
    // A Terminal node is a single terminal symbol bound to a token.
    Terminal,
    // A Sequence node is a nonterminal symbol parsed from a grammar rule,
    // elements() are the parses of each symbol on the RHS of the rule.
    // If the rule is A := X Y Z, the node is for nonterminal A, and elements()
    // are [X, Y, Z].
    Sequence,
    // An Ambiguous node exposes multiple ways to interpret the code as the
    // same symbol, alternatives() are all possible parses.
    Ambiguous,
    // An Opaque node is a placeholder. It asserts that tokens match a symbol,
    // without saying how.
    // It is used for lazy-parsing (not parsed yet), or error-recovery (invalid
    // code).
    Opaque,
  };
  Kind kind() const { return K; }

  SymbolID symbol() const { return Symbol; }

  // The start of the token range, it is a poistion within a token stream.
  Token::Index startTokenIndex() const { return StartIndex; }

  // Returns the corresponding grammar rule.
  // REQUIRES: this is a Sequence node.
  RuleID rule() const {
    assert(kind() == Sequence);
    return Data & ((1 << RuleBits) - 1);
  }
  // Returns the parses of each element on the RHS of the rule.
  // REQUIRES: this is a Sequence node;
  llvm::ArrayRef<const ForestNode *> elements() const {
    assert(kind() == Sequence);
    return children(Data >> RuleBits);
  }
  llvm::MutableArrayRef<ForestNode *> elements() {
    assert(kind() == Sequence);
    return children(Data >> RuleBits);
  }

  // Returns all possible interpretations of the code.
  // REQUIRES: this is an Ambiguous node.
  llvm::ArrayRef<const ForestNode *> alternatives() const {
    assert(kind() == Ambiguous);
    return children(Data);
  }
  llvm::MutableArrayRef<ForestNode *> alternatives() {
    assert(kind() == Ambiguous);
    return children(Data);
  }

  llvm::ArrayRef<const ForestNode *> children() const {
    switch (kind()) {
    case Sequence:
      return elements();
    case Ambiguous:
      return alternatives();
    case Terminal:
    case Opaque:
      return {};
    }
    llvm_unreachable("Bad kind");
  }

  // Iteration over all nodes in the forest, including this.
  llvm::iterator_range<RecursiveIterator> descendants() const;

  std::string dump(const Grammar &) const;
  std::string dumpRecursive(const Grammar &, bool Abbreviated = false) const;

private:
  friend class ForestArena;

  ForestNode(Kind K, SymbolID Symbol, Token::Index StartIndex, uint16_t Data)
      : StartIndex(StartIndex), K(K), Symbol(Symbol), Data(Data) {}

  ForestNode(const ForestNode &) = delete;
  ForestNode &operator=(const ForestNode &) = delete;
  ForestNode(ForestNode &&) = delete;
  ForestNode &operator=(ForestNode &&) = delete;

  static uint16_t sequenceData(RuleID Rule,
                               llvm::ArrayRef<const ForestNode *> Elements) {
    assert(Rule < (1 << RuleBits));
    assert(Elements.size() < (1 << (16 - RuleBits)));
    return Rule | Elements.size() << RuleBits;
  }
  static uint16_t
  ambiguousData(llvm::ArrayRef<const ForestNode *> Alternatives) {
    return Alternatives.size();
  }

  // Retrieves the trailing array.
  llvm::ArrayRef<const ForestNode *> children(uint16_t Num) const {
    return llvm::ArrayRef(reinterpret_cast<ForestNode *const *>(this + 1), Num);
  }
  llvm::MutableArrayRef<ForestNode *> children(uint16_t Num) {
    return llvm::MutableArrayRef(reinterpret_cast<ForestNode **>(this + 1),
                                 Num);
  }

  Token::Index StartIndex;
  Kind K : 4;
  SymbolID Symbol : SymbolBits;
  // Sequence - child count : 4 | RuleID : RuleBits (12)
  // Ambiguous - child count : 16
  // Terminal, Opaque - unused
  uint16_t Data;
  // An array of ForestNode* following the object.
};
// ForestNode may not be destroyed (for BumpPtrAllocator).
static_assert(std::is_trivially_destructible<ForestNode>());

// A memory arena for the parse forest.
class ForestArena {
public:
  llvm::ArrayRef<ForestNode> createTerminals(const TokenStream &Code);
  ForestNode &createSequence(SymbolID SID, RuleID RID,
                             llvm::ArrayRef<const ForestNode *> Elements) {
    assert(!Elements.empty());
    return create(ForestNode::Sequence, SID,
                  Elements.front()->startTokenIndex(),
                  ForestNode::sequenceData(RID, Elements), Elements);
  }
  ForestNode &createAmbiguous(SymbolID SID,
                              llvm::ArrayRef<const ForestNode *> Alternatives) {
    assert(!Alternatives.empty());
    assert(llvm::all_of(Alternatives,
                        [SID](const ForestNode *Alternative) {
                          return SID == Alternative->symbol();
                        }) &&
           "Ambiguous alternatives must represent the same symbol!");
    return create(ForestNode::Ambiguous, SID,
                  Alternatives.front()->startTokenIndex(),
                  ForestNode::ambiguousData(Alternatives), Alternatives);
  }
  ForestNode &createOpaque(SymbolID SID, Token::Index Start) {
    return create(ForestNode::Opaque, SID, Start, 0, {});
  }

  ForestNode &createTerminal(tok::TokenKind TK, Token::Index Start) {
    return create(ForestNode::Terminal, tokenSymbol(TK), Start, 0, {});
  }

  size_t nodeCount() const { return NodeCount; }
  size_t bytes() const { return Arena.getBytesAllocated() + sizeof(*this); }

private:
  ForestNode &create(ForestNode::Kind K, SymbolID SID, Token::Index Start,
                     uint16_t Data,
                     llvm::ArrayRef<const ForestNode *> Elements) {
    ++NodeCount;
    ForestNode *New = new (Arena.Allocate(
        sizeof(ForestNode) + Elements.size() * sizeof(ForestNode *),
        alignof(ForestNode))) ForestNode(K, SID, Start, Data);
    if (!Elements.empty())
      llvm::copy(Elements, reinterpret_cast<const ForestNode **>(New + 1));
    return *New;
  }

  llvm::BumpPtrAllocator Arena;
  uint32_t NodeCount = 0;
};

class ForestNode::RecursiveIterator
    : public llvm::iterator_facade_base<ForestNode::RecursiveIterator,
                                        std::input_iterator_tag,
                                        const ForestNode> {
  llvm::DenseSet<const ForestNode *> Seen;
  struct StackFrame {
    const ForestNode *Parent;
    unsigned ChildIndex;
  };
  std::vector<StackFrame> Stack;
  const ForestNode *Cur;

public:
  RecursiveIterator(const ForestNode *N = nullptr) : Cur(N) {}

  const ForestNode &operator*() const { return *Cur; }
  void operator++();
  bool operator==(const RecursiveIterator &I) const { return Cur == I.Cur; }
  bool operator!=(const RecursiveIterator &I) const { return !(*this == I); }
};

} // namespace pseudo
} // namespace clang

#endif // CLANG_PSEUDO_FOREST_H
