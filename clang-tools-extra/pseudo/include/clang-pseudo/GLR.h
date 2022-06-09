//===--- GLR.h - Implement a GLR parsing algorithm ---------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This implements a standard Generalized LR (GLR) parsing algorithm.
//
// The GLR parser behaves as a normal LR parser until it encounters a conflict.
// To handle a conflict (where there are multiple actions could perform), the
// parser will simulate nondeterminism by doing a breadth-first search
// over all the possibilities.
//
// Basic mechanisims of the GLR parser:
//  - A number of processes are operated in parallel.
//  - Each process has its own parsing stack and behaves as a standard
//    determinism LR parser.
//  - When a process encounters a conflict, it will be fork (one for each
//    avaiable action).
//  - When a process encounters an error, it is abandoned.
//  - All process are synchronized by the lookahead token: they perfrom shift
//    action at the same time, which means some processes need wait until other
//    processes have performed all reduce actions.
//
//===----------------------------------------------------------------------===//

#ifndef CLANG_PSEUDO_GLR_H
#define CLANG_PSEUDO_GLR_H

#include "clang-pseudo/Forest.h"
#include "clang-pseudo/grammar/Grammar.h"
#include "clang-pseudo/grammar/LRTable.h"
#include "llvm/Support/Allocator.h"
#include <vector>

namespace clang {
namespace pseudo {

// A Graph-Structured Stack efficiently represents all parse stacks of a GLR
// parser.
//
// Each node stores a parse state, the last parsed ForestNode, and the parent
// node. There may be several heads (top of stack), and the parser operates by:
// - shift: pushing terminal symbols on top of the stack
// - reduce: replace N symbols on top of the stack with one nonterminal
//
// The structure is a DAG rather than a linear stack:
// - GLR allows multiple actions (conflicts) on the same head, producing forks
//   where several nodes have the same parent
// - The parser merges nodes with the same (state, ForestNode), producing joins
//   where one node has multiple parents
//
// The parser is responsible for creating nodes and keeping track of the set of
// heads. The GSS class is mostly an arena for them.
struct GSS {
  // A node represents a partial parse of the input up to some point.
  //
  // It is the equivalent of a frame in an LR parse stack.
  // Like such a frame, it has an LR parse state and a syntax-tree node
  // representing the last parsed symbol (a ForestNode in our case).
  // Unlike a regular LR stack frame, it may have multiple parents.
  //
  // Nodes are not exactly pushed and popped on the stack: pushing is just
  // allocating a new head node with a parent pointer to the old head. Popping
  // is just forgetting about a node and remembering its parent instead.
  struct alignas(struct Node *) Node {
    // LR state describing how parsing should continue from this head.
    LRTable::StateID State;
    // Used internally to track reachability during garbage collection.
    bool GCParity;
    // Number of the parents of this node.
    // The parents hold previous parsed symbols, and may resume control after
    // this node is reduced.
    unsigned ParentCount;
    // The parse node for the last parsed symbol.
    // This symbol appears on the left of the dot in the parse state's items.
    // (In the literature, the node is attached to the *edge* to the parent).
    const ForestNode *Payload = nullptr;

    llvm::ArrayRef<const Node *> parents() const {
      return llvm::makeArrayRef(reinterpret_cast<const Node *const *>(this + 1),
                                ParentCount);
    };
    // Parents are stored as a trailing array of Node*.
  };

  // Allocates a new node in the graph.
  const Node *addNode(LRTable::StateID State, const ForestNode *Symbol,
                      llvm::ArrayRef<const Node *> Parents);
  // Frees all nodes not reachable as ancestors of Roots, and returns the count.
  // Calling this periodically prevents steady memory growth of the GSS.
  unsigned gc(std::vector<const Node *> &&Roots);

  size_t bytes() const { return Arena.getTotalMemory() + sizeof(*this); }
  size_t nodesCreated() const { return NodesCreated; }

private:
  // Nodes are recycled using freelists.
  // They are variable size, so use one free-list per distinct #parents.
  std::vector<std::vector<Node *>> FreeList;
  Node *allocate(unsigned Parents);
  void destroy(Node *N);
  // The list of nodes created and not destroyed - our candidates for gc().
  std::vector<Node *> Alive;
  bool GCParity = false; // All nodes should match this, except during GC.

  llvm::BumpPtrAllocator Arena;
  unsigned NodesCreated = 0;
};
llvm::raw_ostream &operator<<(llvm::raw_ostream &, const GSS::Node &);

// Parameters for the GLR parsing.
struct ParseParams {
  // The grammar of the language we're going to parse.
  const Grammar &G;
  // The LR table which GLR uses to parse the input, should correspond to the
  // Grammar G.
  const LRTable &Table;

  // Arena for data structure used by the GLR algorithm.
  ForestArena &Forest;  // Storage for the output forest.
  GSS &GSStack;         // Storage for parsing stacks.
};
// Parses the given token stream as the start symbol with the GLR algorithm,
// and returns a forest node of the start symbol.
//
// A rule `_ := StartSymbol` must exit for the chosen start symbol.
//
// If the parsing fails, we model it as an opaque node in the forest.
const ForestNode &glrParse(const TokenStream &Code, const ParseParams &Params,
                           SymbolID StartSymbol);

// An active stack head can have multiple available actions (reduce/reduce
// actions, reduce/shift actions).
// A step is any one action applied to any one stack head.
struct ParseStep {
  // A specific stack head.
  const GSS::Node *Head = nullptr;
  // An action associated with the head.
  LRTable::Action Action = LRTable::Action::sentinel();
};
// A callback is invoked whenever a new GSS head is created during the GLR
// parsing process (glrShift, or glrReduce).
using NewHeadCallback = std::function<void(const GSS::Node *)>;
// Apply all PendingShift actions on a given GSS state, newly-created heads are
// passed to the callback.
//
// When this function returns, PendingShift is empty.
//
// Exposed for testing only.
void glrShift(std::vector<ParseStep> &PendingShift, const ForestNode &NextTok,
              const ParseParams &Params, NewHeadCallback NewHeadCB);
// Applies PendingReduce actions, until no more reduce actions are available.
//
// When this function returns, PendingReduce is empty. Calls to NewHeadCB may
// add elements to PendingReduce
//
// Exposed for testing only.
void glrReduce(std::vector<ParseStep> &PendingReduce, const ParseParams &Params,
               NewHeadCallback NewHeadCB);

} // namespace pseudo
} // namespace clang

#endif // CLANG_PSEUDO_GLR_H
