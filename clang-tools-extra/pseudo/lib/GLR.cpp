//===--- GLR.cpp   -----------------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang-pseudo/GLR.h"
#include "clang-pseudo/Language.h"
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
#include <optional>
#include <queue>

#define DEBUG_TYPE "GLR.cpp"

namespace clang {
namespace pseudo {
namespace {

Token::Index findRecoveryEndpoint(ExtensionID Strategy, Token::Index Begin,
                                  const TokenStream &Tokens,
                                  const Language &Lang) {
  assert(Strategy != 0);
  if (auto S = Lang.RecoveryStrategies.lookup(Strategy))
    return S(Begin, Tokens);
  return Token::Invalid;
}

} // namespace

void glrRecover(llvm::ArrayRef<const GSS::Node *> OldHeads,
                unsigned &TokenIndex, const ParseParams &Params,
                const Language &Lang,
                std::vector<const GSS::Node *> &NewHeads) {
  LLVM_DEBUG(llvm::dbgs() << "Recovery at token " << TokenIndex << "...\n");
  // Describes a possibility to recover by forcibly interpreting a range of
  // tokens around the cursor as a nonterminal that we expected to see.
  struct PlaceholderRecovery {
    // The token prior to the nonterminal which is being recovered.
    // This starts of the region we're skipping, so higher Position is better.
    Token::Index Position;
    // The nonterminal which will be created in order to recover.
    SymbolID Symbol;
    // The heuristic used to choose the bounds of the nonterminal to recover.
    ExtensionID Strategy;

    // The GSS head where we are expecting the recovered nonterminal.
    const GSS::Node *RecoveryNode;
    // Payload of nodes on the way back from the OldHead to the recovery node.
    // These represent the partial parse that is being discarded.
    // They should become the children of the opaque recovery node.
    // FIXME: internal structure of opaque nodes is not implemented.
    //
    // There may be multiple paths leading to the same recovery node, we choose
    // one arbitrarily.
    std::vector<const ForestNode *> DiscardedParse;
  };
  std::vector<PlaceholderRecovery> Options;

  // Find recovery options by walking up the stack.
  //
  // This is similar to exception handling: we walk up the "frames" of nested
  // rules being parsed until we find one that has a "handler" which allows us
  // to determine the node bounds without parsing it.
  //
  // Unfortunately there's a significant difference: the stack contains both
  // "upward" nodes (ancestor parses) and "leftward" ones.
  // e.g. when parsing `{ if (1) ? }` as compound-stmt, the stack contains:
  //   stmt := IF ( expr ) . stmt      - current state, we should recover here!
  //   stmt := IF ( expr . ) stmt      - (left, no recovery here)
  //   stmt := IF ( . expr ) stmt      - left, we should NOT recover here!
  //   stmt := IF . ( expr ) stmt      - (left, no recovery here)
  //   stmt-seq := . stmt              - up, we might recover here
  //   compound-stmt := { . stmt-seq } - up, we should recover here!
  //
  // It's not obvious how to avoid collecting "leftward" recovery options.
  // I think the distinction is ill-defined after merging items into states.
  // For now, we have to take this into account when defining recovery rules.
  // (e.g. in the expr recovery above, stay inside the parentheses).
  // FIXME: find a more satisfying way to avoid such false recovery.
  // FIXME: Add a test for spurious recovery once tests can define strategies.
  std::vector<const ForestNode *> Path;
  llvm::DenseSet<const GSS::Node *> Seen;
  auto WalkUp = [&](const GSS::Node *N, Token::Index NextTok, auto &WalkUp) {
    if (!Seen.insert(N).second)
      return;
    if (!N->Recovered) { // Don't recover the same way twice!
      for (auto Strategy : Lang.Table.getRecovery(N->State)) {
        Options.push_back(PlaceholderRecovery{
            NextTok,
            Strategy.Result,
            Strategy.Strategy,
            N,
            Path,
        });
        LLVM_DEBUG(llvm::dbgs()
                   << "Option: recover " << Lang.G.symbolName(Strategy.Result)
                   << " at token " << NextTok << "\n");
      }
    }
    Path.push_back(N->Payload);
    for (const GSS::Node *Parent : N->parents())
      WalkUp(Parent, N->Payload->startTokenIndex(), WalkUp);
    Path.pop_back();
  };
  for (auto *N : OldHeads)
    WalkUp(N, TokenIndex, WalkUp);

  // Now we select the option(s) we will use to recover.
  //
  // We prefer options starting further right, as these discard less code
  // (e.g. we prefer to recover inner scopes rather than outer ones).
  // The options also need to agree on an endpoint, so the parser has a
  // consistent position afterwards.
  //
  // So conceptually we're sorting by the tuple (start, end), though we avoid
  // computing `end` for options that can't be winners.

  // Consider options starting further right first.
  // Don't drop the others yet though, we may still use them if preferred fails.
  llvm::stable_sort(Options, [&](const auto &L, const auto &R) {
    return L.Position > R.Position;
  });

  // We may find multiple winners, but they will have the same range.
  std::optional<Token::Range> RecoveryRange;
  std::vector<const PlaceholderRecovery *> BestOptions;
  for (const PlaceholderRecovery &Option : Options) {
    // If this starts further left than options we've already found, then
    // we'll never find anything better. Skip computing End for the rest.
    if (RecoveryRange && Option.Position < RecoveryRange->Begin)
      break;

    auto End = findRecoveryEndpoint(Option.Strategy, Option.Position,
                                    Params.Code, Lang);
    // Recovery may not take the parse backwards.
    if (End == Token::Invalid || End < TokenIndex)
      continue;
    if (RecoveryRange) {
      // If this is worse than our previous options, ignore it.
      if (RecoveryRange->End < End)
        continue;
      // If this is an improvement over our previous options, then drop them.
      if (RecoveryRange->End > End)
        BestOptions.clear();
    }
    // Create recovery nodes and heads for them in the GSS. These may be
    // discarded if a better recovery is later found, but this path isn't hot.
    RecoveryRange = {Option.Position, End};
    BestOptions.push_back(&Option);
  }

  if (BestOptions.empty()) {
    LLVM_DEBUG(llvm::dbgs() << "Recovery failed after trying " << Options.size()
                            << " strategies\n");
    return;
  }

  // We've settled on a set of recovery options, so create their nodes and
  // advance the cursor.
  LLVM_DEBUG({
    llvm::dbgs() << "Recovered range=" << *RecoveryRange << ":";
    for (const auto *Option : BestOptions)
      llvm::dbgs() << " " << Lang.G.symbolName(Option->Symbol);
    llvm::dbgs() << "\n";
  });
  // FIXME: in general, we might have the same Option->Symbol multiple times,
  // and we risk creating redundant Forest and GSS nodes.
  // We also may inadvertently set up the next glrReduce to create a sequence
  // node duplicating an opaque node that we're creating here.
  // There are various options, including simply breaking ties between options.
  // For now it's obscure enough to ignore.
  for (const PlaceholderRecovery *Option : BestOptions) {
    Option->RecoveryNode->Recovered = true;
    const ForestNode &Placeholder =
        Params.Forest.createOpaque(Option->Symbol, RecoveryRange->Begin);
    LRTable::StateID OldState = Option->RecoveryNode->State;
    LRTable::StateID NewState =
        isToken(Option->Symbol)
            ? *Lang.Table.getShiftState(OldState, Option->Symbol)
            : *Lang.Table.getGoToState(OldState, Option->Symbol);
    const GSS::Node *NewHead =
        Params.GSStack.addNode(NewState, &Placeholder, {Option->RecoveryNode});
    NewHeads.push_back(NewHead);
  }
  TokenIndex = RecoveryRange->End;
}

using StateID = LRTable::StateID;

llvm::raw_ostream &operator<<(llvm::raw_ostream &OS, const GSS::Node &N) {
  std::vector<std::string> ParentStates;
  for (const auto *Parent : N.parents())
    ParentStates.push_back(llvm::formatv("{0}", Parent->State));
  OS << llvm::formatv("state {0}, parsed symbol {1}, parents {3}", N.State,
                      N.Payload ? N.Payload->symbol() : 0,
                      llvm::join(ParentStates, ", "));
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
              const Language &Lang, std::vector<const GSS::Node *> &NewHeads) {
  assert(NewTok.kind() == ForestNode::Terminal);
  LLVM_DEBUG(llvm::dbgs() << llvm::formatv("  Shift {0} ({1} active heads):\n",
                                           Lang.G.symbolName(NewTok.symbol()),
                                           OldHeads.size()));

  // We group pending shifts by their target state so we can merge them.
  llvm::SmallVector<std::pair<StateID, const GSS::Node *>, 8> Shifts;
  for (const auto *H : OldHeads)
    if (auto S = Lang.Table.getShiftState(H->State, NewTok.symbol()))
      Shifts.push_back({*S, H});
  llvm::stable_sort(Shifts, llvm::less_first{});

  auto Rest = llvm::ArrayRef(Shifts);
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
  const Language& Lang;
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
  GLRReduce(const ParseParams &Params, const Language &Lang)
      : Params(Params), Lang(Lang) {}

  // Reduce Heads, resulting in new nodes that are appended to Heads.
  // The "consumed" nodes are not removed!
  // Only reduce rules compatible with the Lookahead are applied, though
  // tokenSymbol(tok::unknown) will match any rule.
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
  bool canReduce(const Rule &R, RuleID RID,
                 llvm::ArrayRef<const ForestNode *> RHS) const {
    if (!R.Guarded)
      return true;
    if (auto Guard = Lang.Guards.lookup(RID))
      return Guard({RHS, Params.Code, Lookahead});
    LLVM_DEBUG(llvm::dbgs()
               << llvm::formatv("missing guard implementation for rule {0}\n",
                                Lang.G.dumpRule(RID)));
    return true;
  }
  // pop walks up the parent chain(s) for a reduction from Head by to Rule.
  // Once we reach the end, record the bases and sequences.
  void pop(const GSS::Node *Head, RuleID RID, const Rule &Rule) {
    LLVM_DEBUG(llvm::dbgs() << "  Pop " << Lang.G.dumpRule(RID) << "\n");
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
        if (!canReduce(Rule, RID, TempSequence))
          return;
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
      for (RuleID RID :
           Lang.Table.getReduceRules((*Heads)[NextPopHead]->State)) {
        const auto &Rule = Lang.G.lookupRule(RID);
        if (Lang.Table.canFollow(Rule.Target, Lookahead))
          pop((*Heads)[NextPopHead], RID, Rule);
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

    LLVM_DEBUG(llvm::dbgs() << "  Push " << Lang.G.symbolName(F.Symbol)
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
      for (const GSS::Node *Base : Push.LastPop->parents()) {
        auto NextState = Lang.Table.getGoToState(Base->State, F.Symbol);
        assert(NextState.has_value() && "goto must succeed after reduce!");
        FamilyBases.emplace_back(*NextState, Base);
      }

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
    LLVM_DEBUG(llvm::dbgs() << "    --> " << Parsed->dump(Lang.G) << "\n");

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
  //
  // Returns true if we successfully consumed the next unpopped head.
  bool popAndPushTrivial() {
    if (!Sequences.empty() || Heads->size() != NextPopHead + 1)
      return false;
    const GSS::Node *Head = Heads->back();
    std::optional<RuleID> RID;
    for (RuleID R : Lang.Table.getReduceRules(Head->State)) {
      if (RID.has_value())
        return false;
      RID = R;
    }
    if (!RID)
      return true; // no reductions available, but we've processed the head!
    const auto &Rule = Lang.G.lookupRule(*RID);
    if (!Lang.Table.canFollow(Rule.Target, Lookahead))
      return true; // reduction is not available
    const GSS::Node *Base = Head;
    TempSequence.resize_for_overwrite(Rule.Size);
    for (unsigned I = 0; I < Rule.Size; ++I) {
      if (Base->parents().size() != 1)
        return false;
      TempSequence[Rule.Size - 1 - I] = Base->Payload;
      Base = Base->parents().front();
    }
    if (!canReduce(Rule, *RID, TempSequence))
      return true; // reduction is not available
    const ForestNode *Parsed =
        &Params.Forest.createSequence(Rule.Target, *RID, TempSequence);
    auto NextState = Lang.Table.getGoToState(Base->State, Rule.Target);
    assert(NextState.has_value() && "goto must succeed after reduce!");
    Heads->push_back(Params.GSStack.addNode(*NextState, Parsed, {Base}));
    LLVM_DEBUG(llvm::dbgs()
               << "  Reduce (trivial) " << Lang.G.dumpRule(*RID) << "\n"
               << "    --> S" << Heads->back()->State << "\n");
    return true;
  }
};

} // namespace

ForestNode &glrParse(const ParseParams &Params, SymbolID StartSymbol,
                     const Language &Lang) {
  GLRReduce Reduce(Params, Lang);
  assert(isNonterminal(StartSymbol) && "Start symbol must be a nonterminal");
  llvm::ArrayRef<ForestNode> Terminals = Params.Forest.createTerminals(Params.Code);
  auto &GSS = Params.GSStack;

  StateID StartState = Lang.Table.getStartState(StartSymbol);
  // Heads correspond to the parse of tokens [0, I), NextHeads to [0, I+1).
  std::vector<const GSS::Node *> Heads = {GSS.addNode(/*State=*/StartState,
                                                      /*ForestNode=*/nullptr,
                                                      {})};
  // Invariant: Heads is partitioned by source: {shifted | reduced}.
  // HeadsPartition is the index of the first head formed by reduction.
  // We use this to discard and recreate the reduced heads during recovery.
  unsigned HeadsPartition = Heads.size();
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
  for (unsigned I = 0; I < Terminals.size();) {
    LLVM_DEBUG(llvm::dbgs() << llvm::formatv(
                   "Next token {0} (id={1})\n",
                  Lang.G.symbolName(Terminals[I].symbol()), Terminals[I].symbol()));
    // Consume the token.
    glrShift(Heads, Terminals[I], Params, Lang, NextHeads);

    // If we weren't able to consume the token, try to skip over some tokens
    // so we can keep parsing.
    if (NextHeads.empty()) {
      // The reduction in the previous round was constrained by lookahead.
      // On valid code this only rejects dead ends, but on broken code we should
      // consider all possibilities.
      //
      // We discard all heads formed by reduction, and recreate them without
      // this constraint. This may duplicate some nodes, but it's rare.
      LLVM_DEBUG(llvm::dbgs() << "Shift failed, will attempt recovery. "
                                 "Re-reducing without lookahead.\n");
      Heads.resize(HeadsPartition);
      Reduce(Heads, /*allow all reductions*/ tokenSymbol(tok::unknown));

      glrRecover(Heads, I, Params, Lang, NextHeads);
      if (NextHeads.empty())
        // FIXME: Ensure the `_ := start-symbol` rules have a fallback
        // error-recovery strategy attached. Then this condition can't happen.
        return Params.Forest.createOpaque(StartSymbol, /*Token::Index=*/0);
    } else
      ++I;

    // Form nonterminals containing the token we just consumed.
    SymbolID Lookahead =
        I == Terminals.size() ? tokenSymbol(tok::eof) : Terminals[I].symbol();
    HeadsPartition = NextHeads.size();
    Reduce(NextHeads, Lookahead);
    // Prepare for the next token.
    std::swap(Heads, NextHeads);
    NextHeads.clear();
    MaybeGC();
  }
  LLVM_DEBUG(llvm::dbgs() << llvm::formatv("Reached eof\n"));

  // The parse was successful if in state `_ := start-symbol EOF .`
  // The GSS parent has `_ := start-symbol . EOF`; its payload is the parse.
  auto AfterStart = Lang.Table.getGoToState(StartState, StartSymbol);
  assert(AfterStart.has_value() && "goto must succeed after start symbol!");
  auto Accept = Lang.Table.getShiftState(*AfterStart, tokenSymbol(tok::eof));
  assert(Accept.has_value() && "shift EOF must succeed!");
  auto SearchForAccept = [&](llvm::ArrayRef<const GSS::Node *> Heads) {
    const ForestNode *Result = nullptr;
    for (const auto *Head : Heads) {
      if (Head->State == *Accept) {
        assert(Head->Payload->symbol() == tokenSymbol(tok::eof));
        assert(Result == nullptr && "multiple results!");
        Result = Head->parents().front()->Payload;
        assert(Result->symbol() == StartSymbol);
      }
    }
    return Result;
  };
  if (auto *Result = SearchForAccept(Heads))
    return *const_cast<ForestNode *>(Result); // Safe: we created all nodes.
  // We failed to parse the input, returning an opaque forest node for recovery.
  // FIXME: as above, we can add fallback error handling so this is impossible.
  return Params.Forest.createOpaque(StartSymbol, /*Token::Index=*/0);
}

void glrReduce(std::vector<const GSS::Node *> &Heads, SymbolID Lookahead,
               const ParseParams &Params, const Language &Lang) {
  // Create a new GLRReduce each time for tests, performance doesn't matter.
  GLRReduce{Params, Lang}(Heads, Lookahead);
}

const GSS::Node *GSS::addNode(LRTable::StateID State, const ForestNode *Symbol,
                              llvm::ArrayRef<const Node *> Parents) {
  Node *Result = new (allocate(Parents.size())) Node();
  Result->State = State;
  Result->GCParity = GCParity;
  Result->ParentCount = Parents.size();
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
