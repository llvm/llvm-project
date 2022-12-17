//===--- Forest.cpp - Parse forest  ------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang-pseudo/Forest.h"
#include "clang-pseudo/Token.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/FormatVariadic.h"

namespace clang {
namespace pseudo {

void ForestNode::RecursiveIterator::operator++() {
  auto C = Cur->children();
  // Try to find a child of the current node to descend into.
  for (unsigned I = 0; I < C.size(); ++I) {
    if (Seen.insert(C[I]).second) {
      Stack.push_back({Cur, I});
      Cur = C[I];
      return;
    }
  }
  // Try to find a sibling af an ancestor to advance to.
  for (; !Stack.empty(); Stack.pop_back()) {
    C = Stack.back().Parent->children();
    unsigned &Index = Stack.back().ChildIndex;
    while (++Index < C.size()) {
      if (Seen.insert(C[Index]).second) {
        Cur = C[Index];
        return;
      }
    }
  }
  Cur = nullptr;
}

llvm::iterator_range<ForestNode::RecursiveIterator>
ForestNode::descendants() const {
  return {RecursiveIterator(this), RecursiveIterator()};
}

std::string ForestNode::dump(const Grammar &G) const {
  switch (kind()) {
  case Ambiguous:
    return llvm::formatv("{0} := <ambiguous>", G.symbolName(symbol()));
  case Terminal:
    return llvm::formatv("{0} := tok[{1}]", G.symbolName(symbol()),
                         startTokenIndex());
  case Sequence:
    return G.dumpRule(rule());
  case Opaque:
    return llvm::formatv("{0} := <opaque>", G.symbolName(symbol()));
  }
  llvm_unreachable("Unhandled node kind!");
}

std::string ForestNode::dumpRecursive(const Grammar &G,
                                      bool Abbreviated) const {
  using llvm::formatv;
  Token::Index MaxToken = 0;
  // Count visits of nodes so we can mark those seen multiple times.
  llvm::DenseMap<const ForestNode *, /*VisitCount*/ unsigned> VisitCounts;
  std::function<void(const ForestNode *)> CountVisits =
      [&](const ForestNode *P) {
        MaxToken = std::max(MaxToken, P->startTokenIndex());
        if (VisitCounts[P]++ > 0)
          return; // Don't count children as multiply visited.
        if (P->kind() == Ambiguous)
          llvm::for_each(P->alternatives(), CountVisits);
        else if (P->kind() == Sequence)
          llvm::for_each(P->elements(), CountVisits);
      };
  CountVisits(this);

  unsigned IndexWidth = std::max(3, (int)std::to_string(MaxToken).size());
  // e.g. "[{0,4}, {1,4})" if MaxToken is 5742.
  std::string RangeFormat = formatv("[{{0,{0}}, {{1,{0}}) ", IndexWidth);

  // The box-drawing characters that should be added as a child is rendered.
  struct LineDecoration {
    std::string Prefix;         // Prepended to every line.
    llvm::StringRef First;      // added to the child's line.
    llvm::StringRef Subsequent; // added to descendants' lines.
  };

  // We print a "#<id>" for nonterminal forest nodes that are being dumped
  // multiple times.
  llvm::DenseMap<const ForestNode *, size_t> ReferenceIds;
  std::string Result;
  constexpr Token::Index KEnd = std::numeric_limits<Token::Index>::max();
  std::function<void(const ForestNode *, Token::Index, llvm::Optional<SymbolID>,
                     LineDecoration &LineDec)>
      Dump = [&](const ForestNode *P, Token::Index End,
                 llvm::Optional<SymbolID> ElidedParent,
                 LineDecoration LineDec) {
        bool SharedNode = VisitCounts.find(P)->getSecond() > 1;
        llvm::ArrayRef<const ForestNode *> Children;
        auto EndOfElement = [&](size_t ChildIndex) {
          return ChildIndex + 1 == Children.size()
                     ? End
                     : Children[ChildIndex + 1]->startTokenIndex();
        };
        if (P->kind() == Ambiguous) {
          Children = P->alternatives();
        } else if (P->kind() == Sequence) {
          Children = P->elements();
          if (Abbreviated) {
            // Abbreviate chains of trivial sequence nodes.
            //    A := B, B := C, C := D, D := X Y Z
            // becomes
            //    A~D := X Y Z
            //
            // We can't hide nodes that appear multiple times in the tree,
            // because we need to call out their identity with IDs.
            if (Children.size() == 1 && !SharedNode) {
              assert(Children[0]->startTokenIndex() == P->startTokenIndex() &&
                     EndOfElement(0) == End);
              return Dump(Children[0], End,
                          /*ElidedParent=*/ElidedParent.value_or(P->symbol()),
                          LineDec);
            }
          }
        }

        if (End == KEnd)
          Result += formatv(RangeFormat.c_str(), P->startTokenIndex(), "end");
        else
          Result += formatv(RangeFormat.c_str(), P->startTokenIndex(), End);
        Result += LineDec.Prefix;
        Result += LineDec.First;
        if (ElidedParent) {
          Result += G.symbolName(*ElidedParent);
          Result += "~";
        }

        if (SharedNode && P->kind() != ForestNode::Terminal) {
          auto It = ReferenceIds.try_emplace(P, ReferenceIds.size() + 1);
          bool First = It.second;
          unsigned ID = It.first->second;

          // The first time, print as #1. Later, =#1.
          if (First) {
            Result += formatv("{0} #{1}", P->dump(G), ID);
          } else {
            Result += formatv("{0} =#{1}", G.symbolName(P->symbol()), ID);
            Children = {}; // Don't walk the children again.
          }
        } else {
          Result.append(P->dump(G));
        }
        Result.push_back('\n');

        auto OldPrefixSize = LineDec.Prefix.size();
        LineDec.Prefix += LineDec.Subsequent;
        for (size_t I = 0; I < Children.size(); ++I) {
          if (I == Children.size() - 1) {
            LineDec.First = "└─";
            LineDec.Subsequent = "  ";
          } else {
            LineDec.First = "├─";
            LineDec.Subsequent = "│ ";
          }
          Dump(Children[I], P->kind() == Sequence ? EndOfElement(I) : End,
               std::nullopt, LineDec);
        }
        LineDec.Prefix.resize(OldPrefixSize);
      };
  LineDecoration LineDec;
  Dump(this, KEnd, std::nullopt, LineDec);
  return Result;
}

llvm::ArrayRef<ForestNode>
ForestArena::createTerminals(const TokenStream &Code) {
  ForestNode *Terminals = Arena.Allocate<ForestNode>(Code.tokens().size() + 1);
  size_t Index = 0;
  for (const auto &T : Code.tokens()) {
    new (&Terminals[Index])
        ForestNode(ForestNode::Terminal, tokenSymbol(T.Kind),
                   /*Start=*/Index, /*TerminalData*/ 0);
    ++Index;
  }
  // Include an `eof` terminal.
  // This is important to drive the final shift/recover/reduce loop.
  new (&Terminals[Index])
      ForestNode(ForestNode::Terminal, tokenSymbol(tok::eof),
                 /*Start=*/Index, /*TerminalData*/ 0);
  ++Index;
  NodeCount = Index;
  return llvm::makeArrayRef(Terminals, Index);
}

} // namespace pseudo
} // namespace clang
