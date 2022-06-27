//===--- GLR.cpp   -----------------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang-pseudo/GLR.h"
#include "clang-pseudo/grammar/Grammar.h"
#include "clang-pseudo/grammar/LRTable.h"
#include "clang/Basic/TokenKinds.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FormatVariadic.h"
#include <algorithm>
#include <memory>
#include <queue>

#define DEBUG_TYPE "GLR.cpp"

namespace clang {
namespace pseudo {

using StateID = LRTable::StateID;

llvm::raw_ostream &operator<<(llvm::raw_ostream &OS, const GSS::Node &N) {
  std::vector<std::string> ParentStates;
  for (const auto *Parent : N.parents())
    ParentStates.push_back(llvm::formatv("{0}", Parent->State));
  OS << llvm::formatv("state {0}, parsed symbol {1}, parents {2}", N.State,
                      N.Payload->symbol(), llvm::join(ParentStates, ", "));
  return OS;
}

// Apply all pending shift actions.
// In theory, LR parsing doesn't have shift/shift conflicts on a single head.
// But we may have multiple active heads, and each head has a shift action.
//
// We merge the stack -- if multiple heads will reach the same state after
// shifting a token, we shift only once by combining these heads.
//
// E.g. we have two heads (2, 3) in the GSS, and will shift both to reach 4:
//   0---1---2
//       └---3
// After the shift action, the GSS is:
//   0---1---2---4
//       └---3---┘
void glrShift(llvm::ArrayRef<const GSS::Node *> OldHeads,
              const ForestNode &NewTok, const ParseParams &Params,
              std::vector<const GSS::Node *> &NewHeads) {
  assert(NewTok.kind() == ForestNode::Terminal);
  LLVM_DEBUG(llvm::dbgs() << llvm::formatv("  Shift {0} ({1} active heads):\n",
                                           Params.G.symbolName(NewTok.symbol()),
                                           OldHeads.size()));

  // We group pending shifts by their target state so we can merge them.
  llvm::SmallVector<std::pair<StateID, const GSS::Node *>, 8> Shifts;
  for (const auto *H : OldHeads)
    if (auto S = Params.Table.getShiftState(H->State, NewTok.symbol()))
      Shifts.push_back({*S, H});
  llvm::stable_sort(Shifts, llvm::less_first{});

  auto Rest = llvm::makeArrayRef(Shifts);
  llvm::SmallVector<const GSS::Node *> Parents;
  while (!Rest.empty()) {
    // Collect the batch of PendingShift that have compatible shift states.
    // Their heads become TempParents, the parents of the new GSS node.
    StateID NextState = Rest.front().first;

    Parents.clear();
    for (const auto &Base : Rest) {
      if (Base.first != NextState)
        break;
      Parents.push_back(Base.second);
    }
    Rest = Rest.drop_front(Parents.size());

    LLVM_DEBUG(llvm::dbgs() << llvm::formatv("    --> S{0} ({1} heads)\n",
                                             NextState, Parents.size()));
    NewHeads.push_back(Params.GSStack.addNode(NextState, &NewTok, Parents));
  }
}

namespace {
// A KeyedQueue yields pairs of keys and values in order of the keys.
template <typename Key, typename Value>
using KeyedQueue =
    std::priority_queue<std::pair<Key, Value>,
                        std::vector<std::pair<Key, Value>>, llvm::less_first>;

template <typename T> void sortAndUnique(std::vector<T> &Vec) {
  llvm::sort(Vec);
  Vec.erase(std::unique(Vec.begin(), Vec.end()), Vec.end());
}

// Perform reduces until no more are possible.
//
// Generally this means walking up from the heads gathering ForestNodes that
// will match the RHS of the rule we're reducing into a sequence ForestNode,
// and ending up at a base node.
// Then we push a new GSS node onto that base, taking care to:
//  - pack alternative sequence ForestNodes into an ambiguous ForestNode.
//  - use the same GSS node for multiple heads if the parse state matches.
//
// Examples of reduction:
//   Before (simple):
//     0--1(expr)--2(semi)
//   After reducing 2 by `stmt := expr semi`:
//     0--3(stmt)                // 3 is goto(0, stmt)
//
//   Before (splitting due to R/R conflict):
//     0--1(IDENTIFIER)
//   After reducing 1 by `class-name := IDENTIFIER` & `enum-name := IDENTIFIER`:
//     0--2(class-name)          // 2 is goto(0, class-name)
//     └--3(enum-name)           // 3 is goto(0, enum-name)
//
//   Before (splitting due to multiple bases):
//     0--2(class-name)--4(STAR)
//     └--3(enum-name)---┘
//   After reducing 4 by `ptr-operator := STAR`:
//     0--2(class-name)--5(ptr-operator)    // 5 is goto(2, ptr-operator)
//     └--3(enum-name)---6(ptr-operator)    // 6 is goto(3, ptr-operator)
//
//   Before (joining due to same goto state, multiple bases):
//     0--1(cv-qualifier)--3(class-name)
//     └--2(cv-qualifier)--4(enum-name)
//   After reducing 3 by `type-name := class-name` and
//                  4 by `type-name := enum-name`:
//     0--1(cv-qualifier)--5(type-name)  // 5 is goto(1, type-name) and
//     └--2(cv-qualifier)--┘             //      goto(2, type-name)
//
//   Before (joining due to same goto state, the same base):
//     0--1(class-name)--3(STAR)
//     └--2(enum-name)--4(STAR)
//   After reducing 3 by `pointer := class-name STAR` and
//                  2 by`enum-name := class-name STAR`:
//     0--5(pointer)       // 5 is goto(0, pointer)
//
// (This is a functor rather than a function to allow it to reuse scratch
// storage across calls).
class GLRReduce {
  const ParseParams &Params;

  // There are two interacting complications:
  // 1.  Performing one reduce can unlock new reduces on the newly-created head.
  // 2a. The ambiguous ForestNodes must be complete (have all sequence nodes).
  //     This means we must have unlocked all the reduces that contribute to it.
  // 2b. Similarly, the new GSS nodes must be complete (have all parents).
  //
  // We define a "family" of reduces as those that produce the same symbol and
  // cover the same range of tokens. These are exactly the set of reductions
  // whose sequence nodes would be covered by the same ambiguous node.
  // We wish to process a whole family at a time (to satisfy complication 2),
  // and can address complication 1 by carefully ordering the families:
  // - Process families covering fewer tokens first.
  //   A reduce can't depend on a longer reduce!
  // - For equal token ranges: if S := T, process T families before S families.
  //   Parsing T can't depend on an equal-length S, as the grammar is acyclic.
  //
  // This isn't quite enough: we don't know the token length of the reduction
  // until we walk up the stack to perform the pop.
  // So we perform the pop part upfront, and place the push specification on
  // priority queues such that we can retrieve a family at a time.

  // A reduction family is characterized by its token range and symbol produced.
  // It is used as a key in the priority queues to group pushes by family.
  struct Family {
    // The start of the token range of the reduce.
    Token::Index Start;
    SymbolID Symbol;
    // Rule must produce Symbol and can otherwise be arbitrary.
    // RuleIDs have the topological order based on the acyclic grammar.
    // FIXME: should SymbolIDs be so ordered instead?
    RuleID Rule;

    bool operator==(const Family &Other) const {
      return Start == Other.Start && Symbol == Other.Symbol;
    }
    // The larger Family is the one that should be processed first.
    bool operator<(const Family &Other) const {
      if (Start != Other.Start)
        return Start < Other.Start;
      if (Symbol != Other.Symbol)
        return Rule > Other.Rule;
      assert(*this == Other);
      return false;
    }
  };

  // A sequence is the ForestNode payloads of the GSS nodes we are reducing.
  using Sequence = llvm::SmallVector<const ForestNode *, Rule::MaxElements>;
  // Like ArrayRef<const ForestNode*>, but with the missing operator<.
  // (Sequences are big to move by value as the collections gets rearranged).
  struct SequenceRef {
    SequenceRef(const Sequence &S) : S(S) {}
    llvm::ArrayRef<const ForestNode *> S;
    friend bool operator==(SequenceRef A, SequenceRef B) { return A.S == B.S; }
    friend bool operator<(const SequenceRef &A, const SequenceRef &B) {
      return std::lexicographical_compare(A.S.begin(), A.S.end(), B.S.begin(),
                                          B.S.end());
    }
  };
  // Underlying storage for sequences pointed to by stored SequenceRefs.
  std::deque<Sequence> SequenceStorage;
  // We don't actually destroy the sequences between calls, to reuse storage.
  // Everything SequenceStorage[ >=SequenceStorageCount ] is reusable scratch.
  unsigned SequenceStorageCount;

  // Halfway through a reduction (after the pop, before the push), we have
  // collected nodes for the RHS of a rule, and reached a base node.
  // They specify a sequence ForestNode we may build (but we dedup first).
  // (The RuleID is not stored here, but rather in the Family).
  struct PushSpec {
    // The last node popped before pushing. Its parent is the reduction base(s).
    // (Base is more fundamental, but this is cheaper to store).
    const GSS::Node* LastPop = nullptr;
    Sequence *Seq = nullptr;
  };
  KeyedQueue<Family, PushSpec> Sequences; // FIXME: rename => PendingPushes?

  // We treat Heads as a queue of Pop operations still to be performed.
  // PoppedHeads is our position within it.
  std::vector<const GSS::Node *> *Heads;
  unsigned NextPopHead;
  SymbolID Lookahead;

  Sequence TempSequence;
public:
  GLRReduce(const ParseParams &Params) : Params(Params) {}

  void operator()(std::vector<const GSS::Node *> &Heads, SymbolID Lookahead) {
    assert(isToken(Lookahead));

    NextPopHead = 0;
    this->Heads = &Heads;
    this->Lookahead = Lookahead;
    assert(Sequences.empty());
    SequenceStorageCount = 0;

    popPending();
    while (!Sequences.empty()) {
      pushNext();
      popPending();
    }
  }

private:
  // pop walks up the parent chain(s) for a reduction from Head by to Rule.
  // Once we reach the end, record the bases and sequences.
  void pop(const GSS::Node *Head, RuleID RID) {
    LLVM_DEBUG(llvm::dbgs() << "  Pop " << Params.G.dumpRule(RID) << "\n");
    const auto &Rule = Params.G.lookupRule(RID);
    Family F{/*Start=*/0, /*Symbol=*/Rule.Target, /*Rule=*/RID};
    TempSequence.resize_for_overwrite(Rule.Size);
    auto DFS = [&](const GSS::Node *N, unsigned I, auto &DFS) {
      TempSequence[Rule.Size - 1 - I] = N->Payload;
      if (I + 1 == Rule.Size) {
        F.Start = TempSequence.front()->startTokenIndex();
        LLVM_DEBUG({
          for (const auto *B : N->parents())
            llvm::dbgs() << "    --> base at S" << B->State << "\n";
        });

        // Copy the chain to stable storage so it can be enqueued.
        if (SequenceStorageCount == SequenceStorage.size())
          SequenceStorage.emplace_back();
        SequenceStorage[SequenceStorageCount] = TempSequence;
        Sequence *Seq = &SequenceStorage[SequenceStorageCount++];

        Sequences.emplace(F, PushSpec{N, Seq});
        return;
      }
      for (const GSS::Node *Parent : N->parents())
        DFS(Parent, I + 1, DFS);
    };
    DFS(Head, 0, DFS);
  }

  // popPending pops every available reduction.
  void popPending() {
    for (; NextPopHead < Heads->size(); ++NextPopHead) {
      // In trivial cases, we perform the complete reduce here!
      if (popAndPushTrivial())
        continue;
      for (const auto &A :
           Params.Table.getActions((*Heads)[NextPopHead]->State, Lookahead)) {
        if (A.kind() != LRTable::Action::Reduce)
          continue;
        pop((*Heads)[NextPopHead], A.getReduceRule());
      }
    }
  }

  // Storage reused by each call to pushNext.
  std::vector<std::pair</*Goto*/ StateID, const GSS::Node *>> FamilyBases;
  std::vector<std::pair<RuleID, SequenceRef>> FamilySequences;
  std::vector<const GSS::Node *> Parents;
  std::vector<const ForestNode *> SequenceNodes;

  // Process one push family, forming a forest node.
  // This produces new GSS heads which may enable more pops.
  void pushNext() {
    assert(!Sequences.empty());
    Family F = Sequences.top().first;

    LLVM_DEBUG(llvm::dbgs() << "  Push " << Params.G.symbolName(F.Symbol)
                            << " from token " << F.Start << "\n");

    // Grab the sequences and bases for this family.
    // We don't care which rule yielded each base. If Family.Symbol is S, the
    // base includes an item X := ... • S ... and since the grammar is
    // context-free, *all* parses of S are valid here.
    FamilySequences.clear();
    FamilyBases.clear();
    do {
      const PushSpec &Push = Sequences.top().second;
      FamilySequences.emplace_back(Sequences.top().first.Rule, *Push.Seq);
      for (const GSS::Node *Base : Push.LastPop->parents())
        FamilyBases.emplace_back(
            Params.Table.getGoToState(Base->State, F.Symbol), Base);

      Sequences.pop();
    } while (!Sequences.empty() && Sequences.top().first == F);
    // Build a forest node for each unique sequence.
    sortAndUnique(FamilySequences);
    SequenceNodes.clear();
    for (const auto &SequenceSpec : FamilySequences)
      SequenceNodes.push_back(&Params.Forest.createSequence(
          F.Symbol, SequenceSpec.first, SequenceSpec.second.S));
    // Wrap in an ambiguous node if needed.
    const ForestNode *Parsed =
        SequenceNodes.size() == 1
            ? SequenceNodes.front()
            : &Params.Forest.createAmbiguous(F.Symbol, SequenceNodes);
    LLVM_DEBUG(llvm::dbgs() << "    --> " << Parsed->dump(Params.G) << "\n");

    // Bases for this family, deduplicate them, and group by the goTo State.
    sortAndUnique(FamilyBases);
    // Create a GSS node for each unique goto state.
    llvm::ArrayRef<decltype(FamilyBases)::value_type> BasesLeft = FamilyBases;
    while (!BasesLeft.empty()) {
      StateID NextState = BasesLeft.front().first;
      Parents.clear();
      for (const auto &Base : BasesLeft) {
        if (Base.first != NextState)
          break;
        Parents.push_back(Base.second);
      }
      BasesLeft = BasesLeft.drop_front(Parents.size());
      Heads->push_back(Params.GSStack.addNode(NextState, Parsed, Parents));
    }
  }

  // In general we split a reduce into a pop/push, so concurrently-available
  // reductions can run in the correct order. The data structures are expensive.
  //
  // When only one reduction is possible at a time, we can skip this:
  // we pop and immediately push, as an LR parser (as opposed to GLR) would.
  // This is valid whenever there's only one concurrent PushSpec.
  //
  // This function handles a trivial but common subset of these cases:
  //  - there must be no pending pushes, and only one poppable head
  //  - the head must have only one reduction rule
  //  - the reduction path must be a straight line (no multiple parents)
  // (Roughly this means there's no local ambiguity, so the LR algorithm works).
  bool popAndPushTrivial() {
    if (!Sequences.empty() || Heads->size() != NextPopHead + 1)
      return false;
    const GSS::Node *Head = Heads->back();
    llvm::Optional<RuleID> RID;
    for (auto &A : Params.Table.getActions(Head->State, Lookahead)) {
      if (A.kind() != LRTable::Action::Reduce)
        continue;
      if (RID)
        return false;
      RID = A.getReduceRule();
    }
    if (!RID)
      return true; // no reductions available, but we've processed the head!
    const auto &Rule = Params.G.lookupRule(*RID);
    const GSS::Node *Base = Head;
    TempSequence.resize_for_overwrite(Rule.Size);
    for (unsigned I = 0; I < Rule.Size; ++I) {
      if (Base->parents().size() != 1)
        return false;
      TempSequence[Rule.Size - 1 - I] = Base->Payload;
      Base = Base->parents().front();
    }
    const ForestNode *Parsed =
        &Params.Forest.createSequence(Rule.Target, *RID, TempSequence);
    StateID NextState = Params.Table.getGoToState(Base->State, Rule.Target);
    Heads->push_back(Params.GSStack.addNode(NextState, Parsed, {Base}));
    return true;
  }
};

} // namespace

const ForestNode &glrParse(const TokenStream &Tokens, const ParseParams &Params,
                           SymbolID StartSymbol) {
  GLRReduce Reduce(Params);
  assert(isNonterminal(StartSymbol) && "Start symbol must be a nonterminal");
  llvm::ArrayRef<ForestNode> Terminals = Params.Forest.createTerminals(Tokens);
  auto &G = Params.G;
  (void)G;
  auto &GSS = Params.GSStack;

  StateID StartState = Params.Table.getStartState(StartSymbol);
  // Heads correspond to the parse of tokens [0, I), NextHeads to [0, I+1).
  std::vector<const GSS::Node *> Heads = {GSS.addNode(/*State=*/StartState,
                                                      /*ForestNode=*/nullptr,
                                                      {})};
  std::vector<const GSS::Node *> NextHeads;
  auto MaybeGC = [&, Roots(std::vector<const GSS::Node *>{}), I(0u)]() mutable {
    assert(NextHeads.empty() && "Running GC at the wrong time!");
    if (++I != 20) // Run periodically to balance CPU and memory usage.
      return;
    I = 0;

    // We need to copy the list: Roots is consumed by the GC.
    Roots = Heads;
    GSS.gc(std::move(Roots));
  };
  // Each iteration fully processes a single token.
  for (unsigned I = 0; I < Terminals.size(); ++I) {
    LLVM_DEBUG(llvm::dbgs() << llvm::formatv(
                   "Next token {0} (id={1})\n",
                   G.symbolName(Terminals[I].symbol()), Terminals[I].symbol()));
    // Consume the token.
    glrShift(Heads, Terminals[I], Params, NextHeads);
    // Form nonterminals containing the token we just consumed.
    SymbolID Lookahead = I + 1 == Terminals.size() ? tokenSymbol(tok::eof)
                                                   : Terminals[I + 1].symbol();
    Reduce(NextHeads, Lookahead);
    // Prepare for the next token.
    std::swap(Heads, NextHeads);
    NextHeads.clear();
    MaybeGC();
  }
  LLVM_DEBUG(llvm::dbgs() << llvm::formatv("Reached eof\n"));

  StateID AcceptState = Params.Table.getGoToState(StartState, StartSymbol);
  const ForestNode *Result = nullptr;
  for (const auto *Head : Heads) {
    if (Head->State == AcceptState) {
      assert(Head->Payload->symbol() == StartSymbol);
      assert(Result == nullptr && "multiple results!");
      Result = Head->Payload;
    }
  }
  if (Result)
    return *Result;
  // We failed to parse the input, returning an opaque forest node for recovery.
  //
  // FIXME: We will need to invoke our generic error-recovery handlers when we
  // reach EOF without reaching accept state, and involving the eof
  // token in the above main for-loopmay be the best way to reuse the code).
  return Params.Forest.createOpaque(StartSymbol, /*Token::Index=*/0);
}

void glrReduce(std::vector<const GSS::Node *> &Heads, SymbolID Lookahead,
               const ParseParams &Params) {
  // Create a new GLRReduce each time for tests, performance doesn't matter.
  GLRReduce{Params}(Heads, Lookahead);
}

const GSS::Node *GSS::addNode(LRTable::StateID State, const ForestNode *Symbol,
                              llvm::ArrayRef<const Node *> Parents) {
  Node *Result = new (allocate(Parents.size()))
      Node({State, GCParity, static_cast<unsigned>(Parents.size())});
  Alive.push_back(Result);
  ++NodesCreated;
  Result->Payload = Symbol;
  if (!Parents.empty())
    llvm::copy(Parents, reinterpret_cast<const Node **>(Result + 1));
  return Result;
}

GSS::Node *GSS::allocate(unsigned Parents) {
  if (FreeList.size() <= Parents)
    FreeList.resize(Parents + 1);
  auto &SizedList = FreeList[Parents];
  if (!SizedList.empty()) {
    auto *Result = SizedList.back();
    SizedList.pop_back();
    return Result;
  }
  return static_cast<Node *>(
      Arena.Allocate(sizeof(Node) + Parents * sizeof(Node *), alignof(Node)));
}

void GSS::destroy(Node *N) {
  unsigned ParentCount = N->ParentCount;
  N->~Node();
  assert(FreeList.size() > ParentCount && "established on construction!");
  FreeList[ParentCount].push_back(N);
}

unsigned GSS::gc(std::vector<const Node *> &&Queue) {
#ifndef NDEBUG
  auto ParityMatches = [&](const Node *N) { return N->GCParity == GCParity; };
  assert("Before GC" && llvm::all_of(Alive, ParityMatches));
  auto Deferred = llvm::make_scope_exit(
      [&] { assert("After GC" && llvm::all_of(Alive, ParityMatches)); });
  assert(llvm::all_of(
      Queue, [&](const Node *R) { return llvm::is_contained(Alive, R); }));
#endif
  unsigned InitialCount = Alive.size();

  // Mark
  GCParity = !GCParity;
  while (!Queue.empty()) {
    Node *N = const_cast<Node *>(Queue.back()); // Safe: we created these nodes.
    Queue.pop_back();
    if (N->GCParity != GCParity) { // Not seen yet
      N->GCParity = GCParity;      // Mark as seen
      for (const Node *P : N->parents()) // And walk parents
        Queue.push_back(P);
    }
  }
  // Sweep
  llvm::erase_if(Alive, [&](Node *N) {
    if (N->GCParity == GCParity) // Walk reached this node.
      return false;
    destroy(N);
    return true;
  });

  LLVM_DEBUG(llvm::dbgs() << "GC pruned " << (InitialCount - Alive.size())
                          << "/" << InitialCount << " GSS nodes\n");
  return InitialCount - Alive.size();
}

} // namespace pseudo
} // namespace clang
