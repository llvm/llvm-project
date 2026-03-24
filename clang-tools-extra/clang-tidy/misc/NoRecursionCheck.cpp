//===----------------------------------------------------------------------===//
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

static constexpr unsigned SmallCallStackSize = 16;
static constexpr unsigned SmallSCCSize = 32;

using CallStackTy = SmallVector<CallGraphNode::CallRecord, SmallCallStackSize>;

// In given SCC, find *some* call stack that will be cyclic.
// This will only find *one* such stack, it might not be the smallest one,
// and there may be other loops.
static CallStackTy pathfindSomeCycle(ArrayRef<CallGraphNode *> SCC) {
  // We'll need to be able to performantly look up whether some CallGraphNode
  // is in SCC or not, so cache all the SCC elements in a set.
  const llvm::SmallPtrSet<CallGraphNode *, SmallSCCSize> SCCElts(
      llvm::from_range, SCC);

  // Is node N part if the current SCC?
  auto NodeIsPartOfSCC = [&SCCElts](CallGraphNode *N) {
    return SCCElts.contains(N);
  };

  // Track the call stack that will cause a cycle.
  llvm::SmallSetVector<CallGraphNode::CallRecord, SmallCallStackSize>
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

void NoRecursionCheck::registerMatchers(MatchFinder *Finder) {
  Finder->addMatcher(translationUnitDecl().bind("TUDecl"), this);
}

void NoRecursionCheck::handleSCC(ArrayRef<CallGraphNode *> SCC) {
  assert(!SCC.empty() && "Empty SCC does not make sense.");

  // First of all, call out every strongly connected function.
  for (const CallGraphNode *N : SCC) {
    const FunctionDecl *D = N->getDefinition();
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
  const FunctionDecl *CycleEntryFn =
      CyclicCallStack.front().Callee->getDefinition();
  // And now, for ease of understanding, let's print the call sequence that
  // forms the cycle in question.
  diag(CycleEntryFn->getLocation(),
       "example recursive call chain, starting from function %0",
       DiagnosticIDs::Note)
      << CycleEntryFn;
  for (int CurFrame = 1, NumFrames = CyclicCallStack.size();
       CurFrame != NumFrames; ++CurFrame) {
    const CallGraphNode::CallRecord PrevNode = CyclicCallStack[CurFrame - 1];
    const CallGraphNode::CallRecord CurrNode = CyclicCallStack[CurFrame];

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
