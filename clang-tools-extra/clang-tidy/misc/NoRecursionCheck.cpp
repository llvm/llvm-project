//===--- NoRecursionCheck.cpp - clang-tidy --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "NoRecursionCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Analysis/CallGraph.h"
#include "llvm/ADT/SCCIterator.h"

using namespace clang::ast_matchers;

namespace clang::tidy::misc {

namespace {

/// Much like SmallSet, with two differences:
/// 1. It can *only* be constructed from an ArrayRef<>. If the element count
///    is small, there is no copy and said storage *must* outlive us.
/// 2. it is immutable, the way it was constructed it will stay.
template <typename T, unsigned SmallSize> class ImmutableSmallSet {
  ArrayRef<T> Vector;
  llvm::DenseSet<T> Set;

  static_assert(SmallSize <= 32, "N should be small");

  bool isSmall() const { return Set.empty(); }

public:
  using size_type = size_t;

  ImmutableSmallSet() = delete;
  ImmutableSmallSet(const ImmutableSmallSet &) = delete;
  ImmutableSmallSet(ImmutableSmallSet &&) = delete;
  T &operator=(const ImmutableSmallSet &) = delete;
  T &operator=(ImmutableSmallSet &&) = delete;

  // WARNING: Storage *must* outlive us if we decide that the size is small.
  ImmutableSmallSet(ArrayRef<T> Storage) {
    // Is size small-enough to just keep using the existing storage?
    if (Storage.size() <= SmallSize) {
      Vector = Storage;
      return;
    }

    // We've decided that it isn't performant to keep using vector.
    // Let's migrate the data into Set.
    Set.reserve(Storage.size());
    Set.insert_range(Storage);
  }

  /// count - Return 1 if the element is in the set, 0 otherwise.
  size_type count(const T &V) const {
    if (isSmall()) {
      // Since the collection is small, just do a linear search.
      return llvm::is_contained(Vector, V) ? 1 : 0;
    }

    return Set.count(V);
  }
};

/// Much like SmallSetVector, but with one difference:
/// when the size is \p SmallSize or less, when checking whether an element is
/// already in the set or not, we perform linear search over the vector,
/// but if the size is larger than \p SmallSize, we look in set.
/// FIXME: upstream this into SetVector/SmallSetVector itself.
template <typename T, unsigned SmallSize> class SmartSmallSetVector {
public:
  using size_type = size_t;

private:
  SmallVector<T, SmallSize> Vector;
  llvm::DenseSet<T> Set;

  static_assert(SmallSize <= 32, "N should be small");

  // Are we still using Vector for uniqness tracking?
  bool isSmall() const { return Set.empty(); }

  // Will one more entry cause Vector to switch away from small-size storage?
  bool entiretyOfVectorSmallSizeIsOccupied() const {
    assert(isSmall() && Vector.size() <= SmallSize &&
           "Shouldn't ask if we have already [should have] migrated into Set.");
    return Vector.size() == SmallSize;
  }

  void populateSet() {
    assert(Set.empty() && "Should not have already utilized the Set.");
    // Magical growth factor prediction - to how many elements do we expect to
    // sanely grow after switching away from small-size storage?
    const size_t NewMaxElts = 4 * Vector.size();
    Vector.reserve(NewMaxElts);
    Set.reserve(NewMaxElts);
    Set.insert_range(Vector);
  }

  /// count - Return 1 if the element is in the set, 0 otherwise.
  size_type count(const T &V) const {
    if (isSmall()) {
      // Since the collection is small, just do a linear search.
      return llvm::is_contained(Vector, V) ? 1 : 0;
    }
    // Look-up in the Set.
    return Set.count(V);
  }

  bool setInsert(const T &V) {
    if (count(V) != 0)
      return false; // Already exists.
    // Does not exist, Can/need to record it.
    if (isSmall()) { // Are we still using Vector for uniqness tracking?
      // Will one more entry fit within small-sized Vector?
      if (!entiretyOfVectorSmallSizeIsOccupied())
        return true; // We'll insert into vector right afterwards anyway.
      // Time to switch to Set.
      populateSet();
    }
    // Set time!
    // Note that this must be after `populateSet()` might have been called.
    bool SetInsertionSucceeded = Set.insert(V).second;
    (void)SetInsertionSucceeded;
    assert(SetInsertionSucceeded && "We did check that no such value existed");
    return true;
  }

public:
  /// Insert a new element into the SmartSmallSetVector.
  /// \returns true if the element was inserted into the SmartSmallSetVector.
  bool insert(const T &X) {
    bool Result = setInsert(X);
    if (Result)
      Vector.push_back(X);
    return Result;
  }

  /// Clear the SmartSmallSetVector and return the underlying vector.
  decltype(Vector) takeVector() {
    Set.clear();
    return std::move(Vector);
  }
};

constexpr unsigned SmallCallStackSize = 16;
constexpr unsigned SmallSCCSize = 32;

using CallStackTy =
    llvm::SmallVector<CallGraphNode::CallRecord, SmallCallStackSize>;

// In given SCC, find *some* call stack that will be cyclic.
// This will only find *one* such stack, it might not be the smallest one,
// and there may be other loops.
CallStackTy pathfindSomeCycle(ArrayRef<CallGraphNode *> SCC) {
  // We'll need to be able to performantly look up whether some CallGraphNode
  // is in SCC or not, so cache all the SCC elements in a set.
  const ImmutableSmallSet<CallGraphNode *, SmallSCCSize> SCCElts(SCC);

  // Is node N part if the current SCC?
  auto NodeIsPartOfSCC = [&SCCElts](CallGraphNode *N) {
    return SCCElts.count(N) != 0;
  };

  // Track the call stack that will cause a cycle.
  SmartSmallSetVector<CallGraphNode::CallRecord, SmallCallStackSize>
      CallStackSet;

  // Arbitrarily take the first element of SCC as entry point.
  CallGraphNode::CallRecord EntryNode(SCC.front(), /*CallExpr=*/nullptr);
  // Continue recursing into subsequent callees that are part of this SCC,
  // and are thus known to be part of the call graph loop, until loop forms.
  CallGraphNode::CallRecord *Node = &EntryNode;
  while (true) {
    // Did we see this node before?
    if (!CallStackSet.insert(*Node))
      break; // Cycle completed! Note that didn't insert the node into stack!
    // Else, perform depth-first traversal: out of all callees, pick first one
    // that is part of this SCC. This is not guaranteed to yield shortest cycle.
    Node = llvm::find_if(Node->Callee->callees(), NodeIsPartOfSCC);
  }

  // Note that we failed to insert the last node, that completes the cycle.
  // But we really want to have it. So insert it manually into stack only.
  CallStackTy CallStack = CallStackSet.takeVector();
  CallStack.emplace_back(*Node);

  return CallStack;
}

} // namespace

void NoRecursionCheck::registerMatchers(MatchFinder *Finder) {
  Finder->addMatcher(translationUnitDecl().bind("TUDecl"), this);
}

void NoRecursionCheck::handleSCC(ArrayRef<CallGraphNode *> SCC) {
  assert(!SCC.empty() && "Empty SCC does not make sense.");

  // First of all, call out every strongly connected function.
  for (CallGraphNode *N : SCC) {
    FunctionDecl *D = N->getDefinition();
    diag(D->getLocation(), "function %0 is within a recursive call chain") << D;
  }

  // Now, SCC only tells us about strongly connected function declarations in
  // the call graph. It doesn't *really* tell us about the cycles they form.
  // And there may be more than one cycle in SCC.
  // So let's form a call stack that eventually exposes *some* cycle.
  const CallStackTy EventuallyCyclicCallStack = pathfindSomeCycle(SCC);
  assert(!EventuallyCyclicCallStack.empty() && "We should've found the cycle");

  // While last node of the call stack does cause a loop, due to the way we
  // pathfind the cycle, the loop does not necessarily begin at the first node
  // of the call stack, so drop front nodes of the call stack until it does.
  const auto CyclicCallStack =
      ArrayRef<CallGraphNode::CallRecord>(EventuallyCyclicCallStack)
          .drop_until([LastNode = EventuallyCyclicCallStack.back()](
                          CallGraphNode::CallRecord FrontNode) {
            return FrontNode == LastNode;
          });
  assert(CyclicCallStack.size() >= 2 && "Cycle requires at least 2 frames");

  // Which function we decided to be the entry point that lead to the recursion?
  FunctionDecl *CycleEntryFn = CyclicCallStack.front().Callee->getDefinition();
  // And now, for ease of understanding, let's print the call sequence that
  // forms the cycle in question.
  diag(CycleEntryFn->getLocation(),
       "example recursive call chain, starting from function %0",
       DiagnosticIDs::Note)
      << CycleEntryFn;
  for (int CurFrame = 1, NumFrames = CyclicCallStack.size();
       CurFrame != NumFrames; ++CurFrame) {
    CallGraphNode::CallRecord PrevNode = CyclicCallStack[CurFrame - 1];
    CallGraphNode::CallRecord CurrNode = CyclicCallStack[CurFrame];

    Decl *PrevDecl = PrevNode.Callee->getDecl();
    Decl *CurrDecl = CurrNode.Callee->getDecl();

    diag(CurrNode.CallExpr->getBeginLoc(),
         "Frame #%0: function %1 calls function %2 here:", DiagnosticIDs::Note)
        << CurFrame << cast<NamedDecl>(PrevDecl) << cast<NamedDecl>(CurrDecl);
  }

  diag(CyclicCallStack.back().CallExpr->getBeginLoc(),
       "... which was the starting point of the recursive call chain; there "
       "may be other cycles",
       DiagnosticIDs::Note);
}

void NoRecursionCheck::check(const MatchFinder::MatchResult &Result) {
  // Build call graph for the entire translation unit.
  const auto *TU = Result.Nodes.getNodeAs<TranslationUnitDecl>("TUDecl");
  CallGraph CG;
  CG.addToCallGraph(const_cast<TranslationUnitDecl *>(TU));

  // Look for cycles in call graph,
  // by looking for Strongly Connected Components (SCC's)
  for (llvm::scc_iterator<CallGraph *> SCCI = llvm::scc_begin(&CG),
                                       SCCE = llvm::scc_end(&CG);
       SCCI != SCCE; ++SCCI) {
    if (!SCCI.hasCycle()) // We only care about cycles, not standalone nodes.
      continue;
    handleSCC(*SCCI);
  }
}

} // namespace clang::tidy::misc
