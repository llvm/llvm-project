//==- CoreEngine.cpp - Path-Sensitive Dataflow Engine ------------*- C++ -*-//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines a generic engine for intraprocedural, path-sensitive,
//  dataflow analysis via graph reachability engine.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "CoreEngine"

#include "clang/StaticAnalyzer/Core/PathSensitive/AnalysisManager.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CoreEngine.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/ExprEngine.h"
#include "clang/AST/Expr.h"
#include "clang/AST/StmtCXX.h"
#include "llvm/Support/Casting.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/Statistic.h"

using namespace clang;
using namespace ento;

STATISTIC(NumSteps,
            "The # of steps executed.");
STATISTIC(NumReachedMaxSteps,
            "The # of times we reached the max number of steps.");
STATISTIC(NumPathsExplored,
            "The # of paths explored by the analyzer.");

//===----------------------------------------------------------------------===//
// Worklist classes for exploration of reachable states.
//===----------------------------------------------------------------------===//

WorkList::Visitor::~Visitor() {}

namespace {
class DFS : public WorkList {
  SmallVector<WorkListUnit,20> Stack;
public:
  virtual bool hasWork() const {
    return !Stack.empty();
  }

  virtual void enqueue(const WorkListUnit& U) {
    Stack.push_back(U);
  }

  virtual WorkListUnit dequeue() {
    assert (!Stack.empty());
    const WorkListUnit& U = Stack.back();
    Stack.pop_back(); // This technically "invalidates" U, but we are fine.
    return U;
  }
  
  virtual bool visitItemsInWorkList(Visitor &V) {
    for (SmallVectorImpl<WorkListUnit>::iterator
         I = Stack.begin(), E = Stack.end(); I != E; ++I) {
      if (V.visit(*I))
        return true;
    }
    return false;
  }
};

class BFS : public WorkList {
  std::deque<WorkListUnit> Queue;
public:
  virtual bool hasWork() const {
    return !Queue.empty();
  }

  virtual void enqueue(const WorkListUnit& U) {
    Queue.push_back(U);
  }

  virtual WorkListUnit dequeue() {
    WorkListUnit U = Queue.front();
    Queue.pop_front();
    return U;
  }
  
  virtual bool visitItemsInWorkList(Visitor &V) {
    for (std::deque<WorkListUnit>::iterator
         I = Queue.begin(), E = Queue.end(); I != E; ++I) {
      if (V.visit(*I))
        return true;
    }
    return false;
  }
};

} // end anonymous namespace

// Place the dstor for WorkList here because it contains virtual member
// functions, and we the code for the dstor generated in one compilation unit.
WorkList::~WorkList() {}

WorkList *WorkList::makeDFS() { return new DFS(); }
WorkList *WorkList::makeBFS() { return new BFS(); }

namespace {
  class BFSBlockDFSContents : public WorkList {
    std::deque<WorkListUnit> Queue;
    SmallVector<WorkListUnit,20> Stack;
  public:
    virtual bool hasWork() const {
      return !Queue.empty() || !Stack.empty();
    }

    virtual void enqueue(const WorkListUnit& U) {
      if (isa<BlockEntrance>(U.getNode()->getLocation()))
        Queue.push_front(U);
      else
        Stack.push_back(U);
    }

    virtual WorkListUnit dequeue() {
      // Process all basic blocks to completion.
      if (!Stack.empty()) {
        const WorkListUnit& U = Stack.back();
        Stack.pop_back(); // This technically "invalidates" U, but we are fine.
        return U;
      }

      assert(!Queue.empty());
      // Don't use const reference.  The subsequent pop_back() might make it
      // unsafe.
      WorkListUnit U = Queue.front();
      Queue.pop_front();
      return U;
    }
    virtual bool visitItemsInWorkList(Visitor &V) {
      for (SmallVectorImpl<WorkListUnit>::iterator
           I = Stack.begin(), E = Stack.end(); I != E; ++I) {
        if (V.visit(*I))
          return true;
      }
      for (std::deque<WorkListUnit>::iterator
           I = Queue.begin(), E = Queue.end(); I != E; ++I) {
        if (V.visit(*I))
          return true;
      }
      return false;
    }

  };
} // end anonymous namespace

WorkList* WorkList::makeBFSBlockDFSContents() {
  return new BFSBlockDFSContents();
}

//===----------------------------------------------------------------------===//
// Core analysis engine.
//===----------------------------------------------------------------------===//

/// ExecuteWorkList - Run the worklist algorithm for a maximum number of steps.
bool CoreEngine::ExecuteWorkList(const LocationContext *L, unsigned Steps,
                                   ProgramStateRef InitState) {

  if (G->num_roots() == 0) { // Initialize the analysis by constructing
    // the root if none exists.

    const CFGBlock *Entry = &(L->getCFG()->getEntry());

    assert (Entry->empty() &&
            "Entry block must be empty.");

    assert (Entry->succ_size() == 1 &&
            "Entry block must have 1 successor.");

    // Mark the entry block as visited.
    FunctionSummaries->markVisitedBasicBlock(Entry->getBlockID(),
                                             L->getDecl(),
                                             L->getCFG()->getNumBlockIDs());

    // Get the solitary successor.
    const CFGBlock *Succ = *(Entry->succ_begin());

    // Construct an edge representing the
    // starting location in the function.
    BlockEdge StartLoc(Entry, Succ, L);

    // Set the current block counter to being empty.
    WList->setBlockCounter(BCounterFactory.GetEmptyCounter());

    if (!InitState)
      // Generate the root.
      generateNode(StartLoc, SubEng.getInitialState(L), 0);
    else
      generateNode(StartLoc, InitState, 0);
  }

  // Check if we have a steps limit
  bool UnlimitedSteps = Steps == 0;

  while (WList->hasWork()) {
    if (!UnlimitedSteps) {
      if (Steps == 0) {
        NumReachedMaxSteps++;
        break;
      }
      --Steps;
    }

    NumSteps++;

    const WorkListUnit& WU = WList->dequeue();

    // Set the current block counter.
    WList->setBlockCounter(WU.getBlockCounter());

    // Retrieve the node.
    ExplodedNode *Node = WU.getNode();

    dispatchWorkItem(Node, Node->getLocation(), WU);
  }
  SubEng.processEndWorklist(hasWorkRemaining());
  return WList->hasWork();
}

void CoreEngine::dispatchWorkItem(ExplodedNode* Pred, ProgramPoint Loc,
                                  const WorkListUnit& WU) {
  // Dispatch on the location type.
  switch (Loc.getKind()) {
    case ProgramPoint::BlockEdgeKind:
      HandleBlockEdge(cast<BlockEdge>(Loc), Pred);
      break;

    case ProgramPoint::BlockEntranceKind:
      HandleBlockEntrance(cast<BlockEntrance>(Loc), Pred);
      break;

    case ProgramPoint::BlockExitKind:
      assert (false && "BlockExit location never occur in forward analysis.");
      break;

    case ProgramPoint::CallEnterKind: {
      CallEnter CEnter = cast<CallEnter>(Loc);
      SubEng.processCallEnter(CEnter, Pred);
      break;
    }

    case ProgramPoint::CallExitBeginKind:
      SubEng.processCallExit(Pred);
      break;

    case ProgramPoint::EpsilonKind: {
      assert(Pred->hasSinglePred() &&
             "Assume epsilon has exactly one predecessor by construction");
      ExplodedNode *PNode = Pred->getFirstPred();
      dispatchWorkItem(Pred, PNode->getLocation(), WU);
      break;
    }
    default:
      assert(isa<PostStmt>(Loc) ||
             isa<PostInitializer>(Loc) ||
             isa<PostImplicitCall>(Loc) ||
             isa<CallExitEnd>(Loc));
      HandlePostStmt(WU.getBlock(), WU.getIndex(), Pred);
      break;
  }
}

bool CoreEngine::ExecuteWorkListWithInitialState(const LocationContext *L,
                                                 unsigned Steps,
                                                 ProgramStateRef InitState, 
                                                 ExplodedNodeSet &Dst) {
  bool DidNotFinish = ExecuteWorkList(L, Steps, InitState);
  for (ExplodedGraph::eop_iterator I = G->eop_begin(), 
                                   E = G->eop_end(); I != E; ++I) {
    Dst.Add(*I);
  }
  return DidNotFinish;
}

void CoreEngine::HandleBlockEdge(const BlockEdge &L, ExplodedNode *Pred) {

  const CFGBlock *Blk = L.getDst();
  NodeBuilderContext BuilderCtx(*this, Blk, Pred);

  // Mark this block as visited.
  const LocationContext *LC = Pred->getLocationContext();
  FunctionSummaries->markVisitedBasicBlock(Blk->getBlockID(),
                                           LC->getDecl(),
                                           LC->getCFG()->getNumBlockIDs());

  // Check if we are entering the EXIT block.
  if (Blk == &(L.getLocationContext()->getCFG()->getExit())) {

    assert (L.getLocationContext()->getCFG()->getExit().size() == 0
            && "EXIT block cannot contain Stmts.");

    // Process the final state transition.
    SubEng.processEndOfFunction(BuilderCtx, Pred);

    // This path is done. Don't enqueue any more nodes.
    return;
  }

  // Call into the SubEngine to process entering the CFGBlock.
  ExplodedNodeSet dstNodes;
  BlockEntrance BE(Blk, Pred->getLocationContext());
  NodeBuilderWithSinks nodeBuilder(Pred, dstNodes, BuilderCtx, BE);
  SubEng.processCFGBlockEntrance(L, nodeBuilder, Pred);

  // Auto-generate a node.
  if (!nodeBuilder.hasGeneratedNodes()) {
    nodeBuilder.generateNode(Pred->State, Pred);
  }

  // Enqueue nodes onto the worklist.
  enqueue(dstNodes);
}

void CoreEngine::HandleBlockEntrance(const BlockEntrance &L,
                                       ExplodedNode *Pred) {

  // Increment the block counter.
  const LocationContext *LC = Pred->getLocationContext();
  unsigned BlockId = L.getBlock()->getBlockID();
  BlockCounter Counter = WList->getBlockCounter();
  Counter = BCounterFactory.IncrementCount(Counter, LC->getCurrentStackFrame(),
                                           BlockId);
  WList->setBlockCounter(Counter);

  // Process the entrance of the block.
  if (CFGElement E = L.getFirstElement()) {
    NodeBuilderContext Ctx(*this, L.getBlock(), Pred);
    SubEng.processCFGElement(E, Pred, 0, &Ctx);
  }
  else
    HandleBlockExit(L.getBlock(), Pred);
}

void CoreEngine::HandleBlockExit(const CFGBlock * B, ExplodedNode *Pred) {

  if (const Stmt *Term = B->getTerminator()) {
    switch (Term->getStmtClass()) {
      default:
        llvm_unreachable("Analysis for this terminator not implemented.");

      case Stmt::BinaryOperatorClass: // '&&' and '||'
        HandleBranch(cast<BinaryOperator>(Term)->getLHS(), Term, B, Pred);
        return;

      case Stmt::BinaryConditionalOperatorClass:
      case Stmt::ConditionalOperatorClass:
        HandleBranch(cast<AbstractConditionalOperator>(Term)->getCond(),
                     Term, B, Pred);
        return;

        // FIXME: Use constant-folding in CFG construction to simplify this
        // case.

      case Stmt::ChooseExprClass:
        HandleBranch(cast<ChooseExpr>(Term)->getCond(), Term, B, Pred);
        return;

      case Stmt::CXXTryStmtClass: {
        // Generate a node for each of the successors.
        // Our logic for EH analysis can certainly be improved.
        for (CFGBlock::const_succ_iterator it = B->succ_begin(),
             et = B->succ_end(); it != et; ++it) {
          if (const CFGBlock *succ = *it) {
            generateNode(BlockEdge(B, succ, Pred->getLocationContext()),
                         Pred->State, Pred);
          }
        }
        return;
      }
        
      case Stmt::DoStmtClass:
        HandleBranch(cast<DoStmt>(Term)->getCond(), Term, B, Pred);
        return;

      case Stmt::CXXForRangeStmtClass:
        HandleBranch(cast<CXXForRangeStmt>(Term)->getCond(), Term, B, Pred);
        return;

      case Stmt::ForStmtClass:
        HandleBranch(cast<ForStmt>(Term)->getCond(), Term, B, Pred);
        return;

      case Stmt::ContinueStmtClass:
      case Stmt::BreakStmtClass:
      case Stmt::GotoStmtClass:
        break;

      case Stmt::IfStmtClass:
        HandleBranch(cast<IfStmt>(Term)->getCond(), Term, B, Pred);
        return;

      case Stmt::IndirectGotoStmtClass: {
        // Only 1 successor: the indirect goto dispatch block.
        assert (B->succ_size() == 1);

        IndirectGotoNodeBuilder
           builder(Pred, B, cast<IndirectGotoStmt>(Term)->getTarget(),
                   *(B->succ_begin()), this);

        SubEng.processIndirectGoto(builder);
        return;
      }

      case Stmt::ObjCForCollectionStmtClass: {
        // In the case of ObjCForCollectionStmt, it appears twice in a CFG:
        //
        //  (1) inside a basic block, which represents the binding of the
        //      'element' variable to a value.
        //  (2) in a terminator, which represents the branch.
        //
        // For (1), subengines will bind a value (i.e., 0 or 1) indicating
        // whether or not collection contains any more elements.  We cannot
        // just test to see if the element is nil because a container can
        // contain nil elements.
        HandleBranch(Term, Term, B, Pred);
        return;
      }

      case Stmt::SwitchStmtClass: {
        SwitchNodeBuilder builder(Pred, B, cast<SwitchStmt>(Term)->getCond(),
                                    this);

        SubEng.processSwitch(builder);
        return;
      }

      case Stmt::WhileStmtClass:
        HandleBranch(cast<WhileStmt>(Term)->getCond(), Term, B, Pred);
        return;
    }
  }

  assert (B->succ_size() == 1 &&
          "Blocks with no terminator should have at most 1 successor.");

  generateNode(BlockEdge(B, *(B->succ_begin()), Pred->getLocationContext()),
               Pred->State, Pred);
}

void CoreEngine::HandleBranch(const Stmt *Cond, const Stmt *Term, 
                                const CFGBlock * B, ExplodedNode *Pred) {
  assert(B->succ_size() == 2);
  NodeBuilderContext Ctx(*this, B, Pred);
  ExplodedNodeSet Dst;
  SubEng.processBranch(Cond, Term, Ctx, Pred, Dst,
                       *(B->succ_begin()), *(B->succ_begin()+1));
  // Enqueue the new frontier onto the worklist.
  enqueue(Dst);
}

void CoreEngine::HandlePostStmt(const CFGBlock *B, unsigned StmtIdx, 
                                  ExplodedNode *Pred) {
  assert(B);
  assert(!B->empty());

  if (StmtIdx == B->size())
    HandleBlockExit(B, Pred);
  else {
    NodeBuilderContext Ctx(*this, B, Pred);
    SubEng.processCFGElement((*B)[StmtIdx], Pred, StmtIdx, &Ctx);
  }
}

/// generateNode - Utility method to generate nodes, hook up successors,
///  and add nodes to the worklist.
void CoreEngine::generateNode(const ProgramPoint &Loc,
                              ProgramStateRef State,
                              ExplodedNode *Pred) {

  bool IsNew;
  ExplodedNode *Node = G->getNode(Loc, State, false, &IsNew);

  if (Pred)
    Node->addPredecessor(Pred, *G);  // Link 'Node' with its predecessor.
  else {
    assert (IsNew);
    G->addRoot(Node);  // 'Node' has no predecessor.  Make it a root.
  }

  // Only add 'Node' to the worklist if it was freshly generated.
  if (IsNew) WList->enqueue(Node);
}

void CoreEngine::enqueueStmtNode(ExplodedNode *N,
                                 const CFGBlock *Block, unsigned Idx) {
  assert(Block);
  assert (!N->isSink());

  // Check if this node entered a callee.
  if (isa<CallEnter>(N->getLocation())) {
    // Still use the index of the CallExpr. It's needed to create the callee
    // StackFrameContext.
    WList->enqueue(N, Block, Idx);
    return;
  }

  // Do not create extra nodes. Move to the next CFG element.
  if (isa<PostInitializer>(N->getLocation()) ||
      isa<PostImplicitCall>(N->getLocation())) {
    WList->enqueue(N, Block, Idx+1);
    return;
  }

  if (isa<EpsilonPoint>(N->getLocation())) {
    WList->enqueue(N, Block, Idx);
    return;
  }

  // At this point, we know we're processing a normal statement.
  CFGStmt CS = cast<CFGStmt>((*Block)[Idx]);
  PostStmt Loc(CS.getStmt(), N->getLocationContext());

  if (Loc == N->getLocation()) {
    // Note: 'N' should be a fresh node because otherwise it shouldn't be
    // a member of Deferred.
    WList->enqueue(N, Block, Idx+1);
    return;
  }

  bool IsNew;
  ExplodedNode *Succ = G->getNode(Loc, N->getState(), false, &IsNew);
  Succ->addPredecessor(N, *G);

  if (IsNew)
    WList->enqueue(Succ, Block, Idx+1);
}

ExplodedNode *CoreEngine::generateCallExitBeginNode(ExplodedNode *N) {
  // Create a CallExitBegin node and enqueue it.
  const StackFrameContext *LocCtx
                         = cast<StackFrameContext>(N->getLocationContext());

  // Use the callee location context.
  CallExitBegin Loc(LocCtx);

  bool isNew;
  ExplodedNode *Node = G->getNode(Loc, N->getState(), false, &isNew);
  Node->addPredecessor(N, *G);
  return isNew ? Node : 0;
}


void CoreEngine::enqueue(ExplodedNodeSet &Set) {
  for (ExplodedNodeSet::iterator I = Set.begin(),
                                 E = Set.end(); I != E; ++I) {
    WList->enqueue(*I);
  }
}

void CoreEngine::enqueue(ExplodedNodeSet &Set,
                         const CFGBlock *Block, unsigned Idx) {
  for (ExplodedNodeSet::iterator I = Set.begin(),
                                 E = Set.end(); I != E; ++I) {
    enqueueStmtNode(*I, Block, Idx);
  }
}

void CoreEngine::enqueueEndOfFunction(ExplodedNodeSet &Set) {
  for (ExplodedNodeSet::iterator I = Set.begin(), E = Set.end(); I != E; ++I) {
    ExplodedNode *N = *I;
    // If we are in an inlined call, generate CallExitBegin node.
    if (N->getLocationContext()->getParent()) {
      N = generateCallExitBeginNode(N);
      if (N)
        WList->enqueue(N);
    } else {
      // TODO: We should run remove dead bindings here.
      G->addEndOfPath(N);
      NumPathsExplored++;
    }
  }
}


void NodeBuilder::anchor() { }

ExplodedNode* NodeBuilder::generateNodeImpl(const ProgramPoint &Loc,
                                            ProgramStateRef State,
                                            ExplodedNode *FromN,
                                            bool MarkAsSink) {
  HasGeneratedNodes = true;
  bool IsNew;
  ExplodedNode *N = C.Eng.G->getNode(Loc, State, MarkAsSink, &IsNew);
  N->addPredecessor(FromN, *C.Eng.G);
  Frontier.erase(FromN);

  if (!IsNew)
    return 0;

  if (!MarkAsSink)
    Frontier.Add(N);

  return N;
}

void NodeBuilderWithSinks::anchor() { }

StmtNodeBuilder::~StmtNodeBuilder() {
  if (EnclosingBldr)
    for (ExplodedNodeSet::iterator I = Frontier.begin(),
                                   E = Frontier.end(); I != E; ++I )
      EnclosingBldr->addNodes(*I);
}

void BranchNodeBuilder::anchor() { }

ExplodedNode *BranchNodeBuilder::generateNode(ProgramStateRef State,
                                              bool branch,
                                              ExplodedNode *NodePred) {
  // If the branch has been marked infeasible we should not generate a node.
  if (!isFeasible(branch))
    return NULL;

  ProgramPoint Loc = BlockEdge(C.Block, branch ? DstT:DstF,
                               NodePred->getLocationContext());
  ExplodedNode *Succ = generateNodeImpl(Loc, State, NodePred);
  return Succ;
}

ExplodedNode*
IndirectGotoNodeBuilder::generateNode(const iterator &I,
                                      ProgramStateRef St,
                                      bool IsSink) {
  bool IsNew;
  ExplodedNode *Succ = Eng.G->getNode(BlockEdge(Src, I.getBlock(),
                                      Pred->getLocationContext()), St,
                                      IsSink, &IsNew);
  Succ->addPredecessor(Pred, *Eng.G);

  if (!IsNew)
    return 0;

  if (!IsSink)
    Eng.WList->enqueue(Succ);

  return Succ;
}


ExplodedNode*
SwitchNodeBuilder::generateCaseStmtNode(const iterator &I,
                                        ProgramStateRef St) {

  bool IsNew;
  ExplodedNode *Succ = Eng.G->getNode(BlockEdge(Src, I.getBlock(),
                                      Pred->getLocationContext()), St,
                                      false, &IsNew);
  Succ->addPredecessor(Pred, *Eng.G);
  if (!IsNew)
    return 0;

  Eng.WList->enqueue(Succ);
  return Succ;
}


ExplodedNode*
SwitchNodeBuilder::generateDefaultCaseNode(ProgramStateRef St,
                                           bool IsSink) {
  // Get the block for the default case.
  assert(Src->succ_rbegin() != Src->succ_rend());
  CFGBlock *DefaultBlock = *Src->succ_rbegin();

  // Sanity check for default blocks that are unreachable and not caught
  // by earlier stages.
  if (!DefaultBlock)
    return NULL;
  
  bool IsNew;
  ExplodedNode *Succ = Eng.G->getNode(BlockEdge(Src, DefaultBlock,
                                      Pred->getLocationContext()), St,
                                      IsSink, &IsNew);
  Succ->addPredecessor(Pred, *Eng.G);

  if (!IsNew)
    return 0;

  if (!IsSink)
    Eng.WList->enqueue(Succ);

  return Succ;
}
