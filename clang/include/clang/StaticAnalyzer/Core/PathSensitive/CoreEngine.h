//===- CoreEngine.h - Path-Sensitive Dataflow Engine ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  This file defines a generic engine for intraprocedural, path-sensitive,
//  dataflow analysis via graph reachability.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_STATICANALYZER_CORE_PATHSENSITIVE_COREENGINE_H
#define LLVM_CLANG_STATICANALYZER_CORE_PATHSENSITIVE_COREENGINE_H

#include "clang/AST/Stmt.h"
#include "clang/Analysis/AnalysisDeclContext.h"
#include "clang/Analysis/CFG.h"
#include "clang/Analysis/ProgramPoint.h"
#include "clang/Basic/LLVM.h"
#include "clang/StaticAnalyzer/Core/BugReporter/BugReporter.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/BlockCounter.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/ExplodedGraph.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/ProgramState_Fwd.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/WorkList.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/Support/Casting.h"
#include <cassert>
#include <memory>
#include <utility>
#include <vector>

namespace clang {

class AnalyzerOptions;
class CXXBindTemporaryExpr;
class Expr;
class LabelDecl;

namespace ento {

class FunctionSummariesTy;
class ExprEngine;

//===----------------------------------------------------------------------===//
/// CoreEngine - Implements the core logic of the graph-reachability analysis.
/// It traverses the CFG and generates the ExplodedGraph.
class CoreEngine {
  friend class ExprEngine;
  friend class IndirectGotoNodeBuilder;
  friend class NodeBuilder;
  friend class NodeBuilderContext;
  friend class SwitchNodeBuilder;

public:
  using BlocksExhausted =
      std::vector<std::pair<BlockEdge, const ExplodedNode *>>;

  using BlocksAborted =
      std::vector<std::pair<const CFGBlock *, const ExplodedNode *>>;

private:
  ExprEngine &ExprEng;

  /// G - The simulation graph.  Each node is a (location,state) pair.
  mutable ExplodedGraph G;

  /// WList - A set of queued nodes that need to be processed by the
  ///  worklist algorithm.  It is up to the implementation of WList to decide
  ///  the order that nodes are processed.
  std::unique_ptr<WorkList> WList;
  std::unique_ptr<WorkList> CTUWList;

  /// BCounterFactory - A factory object for created BlockCounter objects.
  ///   These are used to record for key nodes in the ExplodedGraph the
  ///   number of times different CFGBlocks have been visited along a path.
  BlockCounter::Factory BCounterFactory;

  /// The locations where we stopped doing work because we visited a location
  ///  too many times.
  BlocksExhausted blocksExhausted;

  /// The locations where we stopped because the engine aborted analysis,
  /// usually because it could not reason about something.
  BlocksAborted blocksAborted;

  /// The information about functions shared by the whole translation unit.
  /// (This data is owned by AnalysisConsumer.)
  FunctionSummariesTy *FunctionSummaries;

  /// Add path tags with some useful data along the path when we see that
  /// something interesting is happening. This field is the allocator for such
  /// tags.
  DataTag::Factory DataTags;

  void setBlockCounter(BlockCounter C);

  void generateNode(const ProgramPoint &Loc,
                    ProgramStateRef State,
                    ExplodedNode *Pred);

  void HandleBlockEdge(const BlockEdge &E, ExplodedNode *Pred);
  void HandleBlockEntrance(const BlockEntrance &E, ExplodedNode *Pred);
  void HandleBlockExit(const CFGBlock *B, ExplodedNode *Pred);

  void HandleCallEnter(const CallEnter &CE, ExplodedNode *Pred);

  void HandlePostStmt(const CFGBlock *B, unsigned StmtIdx, ExplodedNode *Pred);

  void HandleBranch(const Stmt *Cond, const Stmt *Term, const CFGBlock *B,
                    ExplodedNode *Pred);
  void HandleCleanupTemporaryBranch(const CXXBindTemporaryExpr *BTE,
                                    const CFGBlock *B, ExplodedNode *Pred);

  /// Handle conditional logic for running static initializers.
  void HandleStaticInit(const DeclStmt *DS, const CFGBlock *B,
                        ExplodedNode *Pred);

  void HandleVirtualBaseBranch(const CFGBlock *B, ExplodedNode *Pred);

private:
  ExplodedNode *generateCallExitBeginNode(ExplodedNode *N,
                                          const ReturnStmt *RS);

  /// Helper function called by `HandleBranch()`. If the currently handled
  /// branch corresponds to a loop, this returns the number of already
  /// completed iterations in that loop, otherwise the return value is
  /// `std::nullopt`. Note that this counts _all_ earlier iterations, including
  /// ones that were performed within an earlier iteration of an outer loop.
  std::optional<unsigned> getCompletedIterationCount(const CFGBlock *B,
                                                     ExplodedNode *Pred) const;

public:
  /// Construct a CoreEngine object to analyze the provided CFG.
  CoreEngine(ExprEngine &exprengine,
             FunctionSummariesTy *FS,
             AnalyzerOptions &Opts);

  CoreEngine(const CoreEngine &) = delete;
  CoreEngine &operator=(const CoreEngine &) = delete;

  /// getGraph - Returns the exploded graph.
  ExplodedGraph &getGraph() { return G; }

  /// ExecuteWorkList - Run the worklist algorithm for a maximum number of
  ///  steps.  Returns true if there is still simulation state on the worklist.
  bool ExecuteWorkList(const LocationContext *L, unsigned Steps,
                       ProgramStateRef InitState);

  /// Dispatch the work list item based on the given location information.
  /// Use Pred parameter as the predecessor state.
  void dispatchWorkItem(ExplodedNode* Pred, ProgramPoint Loc,
                        const WorkListUnit& WU);

  // Functions for external checking of whether we have unfinished work.
  bool wasBlockAborted() const { return !blocksAborted.empty(); }
  bool wasBlocksExhausted() const { return !blocksExhausted.empty(); }
  bool hasWorkRemaining() const { return wasBlocksExhausted() ||
                                         WList->hasWork() ||
                                         wasBlockAborted(); }

  /// Inform the CoreEngine that a basic block was aborted because
  /// it could not be completely analyzed.
  void addAbortedBlock(const ExplodedNode *node, const CFGBlock *block) {
    blocksAborted.push_back(std::make_pair(block, node));
  }

  WorkList *getWorkList() const { return WList.get(); }
  WorkList *getCTUWorkList() const { return CTUWList.get(); }

  auto exhausted_blocks() const {
    return llvm::iterator_range(blocksExhausted);
  }

  auto aborted_blocks() const { return llvm::iterator_range(blocksAborted); }

  /// Enqueue the given set of nodes onto the work list.
  void enqueue(ExplodedNodeSet &Set);

  /// Enqueue nodes that were created as a result of processing
  /// a statement onto the work list.
  void enqueue(ExplodedNodeSet &Set, const CFGBlock *Block, unsigned Idx);

  /// enqueue the nodes corresponding to the end of function onto the
  /// end of path / work list.
  void enqueueEndOfFunction(ExplodedNodeSet &Set, const ReturnStmt *RS);

  /// Enqueue a single node created as a result of statement processing.
  void enqueueStmtNode(ExplodedNode *N, const CFGBlock *Block, unsigned Idx);

  DataTag::Factory &getDataTags() { return DataTags; }
};

class NodeBuilderContext {
  const CoreEngine &Eng;
  const CFGBlock *Block;
  const LocationContext *LC;

public:
  NodeBuilderContext(const CoreEngine &E, const CFGBlock *B,
                     const LocationContext *L)
      : Eng(E), Block(B), LC(L) {
    assert(B);
  }

  NodeBuilderContext(const CoreEngine &E, const CFGBlock *B, ExplodedNode *N)
      : NodeBuilderContext(E, B, N->getLocationContext()) {}

  /// Return the CoreEngine associated with this builder.
  const CoreEngine &getEngine() const { return Eng; }

  /// Return the CFGBlock associated with this builder.
  const CFGBlock *getBlock() const { return Block; }

  /// Return the location context associated with this builder.
  const LocationContext *getLocationContext() const { return LC; }

  /// Returns the number of times the current basic block has been
  /// visited on the exploded graph path.
  unsigned blockCount() const {
    return Eng.WList->getBlockCounter().getNumVisited(
                    LC->getStackFrame(),
                    Block->getBlockID());
  }
};

/// \class NodeBuilder
/// This is the simplest builder which generates nodes in the
/// ExplodedGraph.
///
/// The main benefit of the builder is that it automatically tracks the
/// frontier nodes (or destination set). This is the set of nodes which should
/// be propagated to the next step / builder. They are the nodes which have been
/// added to the builder (either as the input node set or as the newly
/// constructed nodes) but did not have any outgoing transitions added.
class NodeBuilder {
  virtual void anchor();

protected:
  const NodeBuilderContext &C;

  bool HasGeneratedNodes = false;

  /// The frontier set - a set of nodes which need to be propagated after
  /// the builder dies.
  ExplodedNodeSet &Frontier;

  bool hasNoSinksInFrontier() {
    for (const auto  I : Frontier)
      if (I->isSink())
        return false;
    return true;
  }

  ExplodedNode *generateNodeImpl(const ProgramPoint &PP,
                                 ProgramStateRef State,
                                 ExplodedNode *Pred,
                                 bool MarkAsSink = false);

public:
  NodeBuilder(ExplodedNode *SrcNode, ExplodedNodeSet &DstSet,
              const NodeBuilderContext &Ctx)
      : C(Ctx), Frontier(DstSet) {
    Frontier.Add(SrcNode);
  }

  NodeBuilder(const ExplodedNodeSet &SrcSet, ExplodedNodeSet &DstSet,
              const NodeBuilderContext &Ctx)
      : C(Ctx), Frontier(DstSet) {
    Frontier.insert(SrcSet);
    assert(hasNoSinksInFrontier());
  }

  virtual ~NodeBuilder() = default;

  /// Generates a node in the ExplodedGraph.
  ExplodedNode *generateNode(const ProgramPoint &PP,
                             ProgramStateRef State,
                             ExplodedNode *Pred) {
    return generateNodeImpl(
        PP, State, Pred,
        /*MarkAsSink=*/State->isPosteriorlyOverconstrained());
  }

  /// Generates a sink in the ExplodedGraph.
  ///
  /// When a node is marked as sink, the exploration from the node is stopped -
  /// the node becomes the last node on the path and certain kinds of bugs are
  /// suppressed.
  ExplodedNode *generateSink(const ProgramPoint &PP,
                             ProgramStateRef State,
                             ExplodedNode *Pred) {
    return generateNodeImpl(PP, State, Pred, true);
  }

  const ExplodedNodeSet &getResults() { return Frontier; }

  using iterator = ExplodedNodeSet::iterator;

  /// Iterators through the results frontier.
  iterator begin() { return Frontier.begin(); }

  iterator end() { return Frontier.end(); }

  const NodeBuilderContext &getContext() { return C; }
  bool hasGeneratedNodes() { return HasGeneratedNodes; }

  void takeNodes(const ExplodedNodeSet &S) {
    for (const auto I : S)
      Frontier.erase(I);
  }

  void takeNodes(ExplodedNode *N) { Frontier.erase(N); }
  void addNodes(const ExplodedNodeSet &S) { Frontier.insert(S); }
  void addNodes(ExplodedNode *N) { Frontier.Add(N); }
};

/// \class StmtNodeBuilder
/// This builder class is useful for generating nodes that resulted from
/// visiting a statement. The main difference from its parent NodeBuilder is
/// that it creates a statement specific ProgramPoint.
class StmtNodeBuilder: public NodeBuilder {
  NodeBuilder *EnclosingBldr;

public:
  /// Constructs a StmtNodeBuilder. If the builder is going to process
  /// nodes currently owned by another builder(with larger scope), use
  /// Enclosing builder to transfer ownership.
  StmtNodeBuilder(ExplodedNode *SrcNode, ExplodedNodeSet &DstSet,
                  const NodeBuilderContext &Ctx,
                  NodeBuilder *Enclosing = nullptr)
      : NodeBuilder(SrcNode, DstSet, Ctx), EnclosingBldr(Enclosing) {
    if (EnclosingBldr)
      EnclosingBldr->takeNodes(SrcNode);
  }

  StmtNodeBuilder(ExplodedNodeSet &SrcSet, ExplodedNodeSet &DstSet,
                  const NodeBuilderContext &Ctx,
                  NodeBuilder *Enclosing = nullptr)
      : NodeBuilder(SrcSet, DstSet, Ctx), EnclosingBldr(Enclosing) {
    if (EnclosingBldr)
      EnclosingBldr->takeNodes(SrcSet);
  }

  ~StmtNodeBuilder() override;

  using NodeBuilder::generateNode;
  using NodeBuilder::generateSink;

  ExplodedNode *generateNode(const Stmt *S,
                             ExplodedNode *Pred,
                             ProgramStateRef St,
                             const ProgramPointTag *tag = nullptr,
                             ProgramPoint::Kind K = ProgramPoint::PostStmtKind){
    const ProgramPoint &L = ProgramPoint::getProgramPoint(S, K,
                                  Pred->getLocationContext(), tag);
    return NodeBuilder::generateNode(L, St, Pred);
  }

  ExplodedNode *generateSink(const Stmt *S,
                             ExplodedNode *Pred,
                             ProgramStateRef St,
                             const ProgramPointTag *tag = nullptr,
                             ProgramPoint::Kind K = ProgramPoint::PostStmtKind){
    const ProgramPoint &L = ProgramPoint::getProgramPoint(S, K,
                                  Pred->getLocationContext(), tag);
    return NodeBuilder::generateSink(L, St, Pred);
  }
};

/// BranchNodeBuilder is responsible for constructing the nodes
/// corresponding to the two branches of the if statement - true and false.
class BranchNodeBuilder: public NodeBuilder {
  const CFGBlock *DstT;
  const CFGBlock *DstF;

  void anchor() override;

public:
  BranchNodeBuilder(ExplodedNode *SrcNode, ExplodedNodeSet &DstSet,
                    const NodeBuilderContext &C, const CFGBlock *DT,
                    const CFGBlock *DF)
      : NodeBuilder(SrcNode, DstSet, C), DstT(DT), DstF(DF) {
    // The branch node builder does not generate autotransitions.
    // If there are no successors it means that both branches are infeasible.
    takeNodes(SrcNode);
  }

  BranchNodeBuilder(const ExplodedNodeSet &SrcSet, ExplodedNodeSet &DstSet,
                    const NodeBuilderContext &C, const CFGBlock *DT,
                    const CFGBlock *DF)
      : NodeBuilder(SrcSet, DstSet, C), DstT(DT), DstF(DF) {
    takeNodes(SrcSet);
  }

  ExplodedNode *generateNode(ProgramStateRef State, bool branch,
                             ExplodedNode *Pred);
};

class IndirectGotoNodeBuilder {
  const CoreEngine &Eng;
  const CFGBlock *Src;
  const CFGBlock &DispatchBlock;
  const Expr *E;
  ExplodedNode *Pred;

public:
  IndirectGotoNodeBuilder(ExplodedNode *Pred, const CFGBlock *Src,
                          const Expr *E, const CFGBlock *Dispatch,
                          const CoreEngine *Eng)
      : Eng(*Eng), Src(Src), DispatchBlock(*Dispatch), E(E), Pred(Pred) {}

  using iterator = CFGBlock::const_succ_iterator;

  iterator begin() { return DispatchBlock.succ_begin(); }
  iterator end() { return DispatchBlock.succ_end(); }

  ExplodedNode *generateNode(const CFGBlock *Block, ProgramStateRef State,
                             bool IsSink = false);

  const Expr *getTarget() const { return E; }

  ProgramStateRef getState() const { return Pred->State; }

  const LocationContext *getLocationContext() const {
    return Pred->getLocationContext();
  }
};

class SwitchNodeBuilder {
  const CoreEngine &Eng;
  const CFGBlock *Src;
  ExplodedNode *Pred;

public:
  SwitchNodeBuilder(ExplodedNode *P, const CFGBlock *S, CoreEngine &E)
      : Eng(E), Src(S), Pred(P) {}

  using iterator = CFGBlock::const_succ_reverse_iterator;

  iterator begin() { return Src->succ_rbegin() + 1; }
  iterator end() { return Src->succ_rend(); }

  ExplodedNode *generateCaseStmtNode(const CFGBlock *Block,
                                     ProgramStateRef State);

  ExplodedNode *generateDefaultCaseNode(ProgramStateRef State,
                                        bool IsSink = false);
};

} // namespace ento

} // namespace clang

#endif // LLVM_CLANG_STATICANALYZER_CORE_PATHSENSITIVE_COREENGINE_H
