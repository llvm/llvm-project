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

public:
  using BlocksExhausted =
      std::vector<std::pair<BlockEntrance, const ExplodedNode *>>;

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

  /// Whether the single-TU phase completed the exploration of all paths
  /// within its budget limit.
  /// The CTU phase replaces \c WList, so this has to be remembered separately.
  bool ExploredAllSTUPaths = false;

  /// The information about functions shared by the whole translation unit.
  /// (This data is owned by AnalysisConsumer.)
  FunctionSummariesTy *FunctionSummaries;

  /// Add path tags with some useful data along the path when we see that
  /// something interesting is happening. This field is the allocator for such
  /// tags.
  DataTag::Factory DataTags;

  void setBlockCounter(BlockCounter C);

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
  bool ExecuteWorkList(const StackFrame *SF, unsigned Steps,
                       ProgramStateRef InitState);

  /// Dispatch the work list item based on the given location information.
  /// Use Pred parameter as the predecessor state.
  void dispatchWorkItem(ExplodedNode* Pred, ProgramPoint Loc,
                        const WorkListUnit& WU);

  // Functions for external checking of whether we have unfinished work.
  bool wasBlockAborted() const { return !blocksAborted.empty(); }
  bool wasBlocksExhausted() const { return !blocksExhausted.empty(); }
  bool hasExploredAllPaths() const {
    return !wasBlocksExhausted() && !WList->hasWork() && ExploredAllSTUPaths &&
           !wasBlockAborted();
  }

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

  ExplodedNode *makeNode(const ProgramPoint &Loc, ProgramStateRef State,
                         ExplodedNode *Pred, bool MarkAsSink = false) const;

  ExplodedNode *makePostStmtNode(const Stmt *S, ProgramStateRef State,
                                 ExplodedNode *Pred,
                                 bool MarkAsSink = false) const {
    PostStmt Loc(S, Pred->getStackFrame(), /*tag=*/nullptr);
    return makeNode(Loc, State, Pred, MarkAsSink);
  }

  ExplodedNode *
  makeNodeWithBinding(ExplodedNode *Pred, const Expr *E, SVal V,
                      ProgramStateRef State,
                      ProgramPoint::Kind K = ProgramPoint::PostStmtKind) const {
    const StackFrame *SF = Pred->getStackFrame();
    State = State->BindExpr(E, SF, V);
    const auto &L = ProgramPoint::getProgramPoint(E, K, SF, /*tag=*/nullptr);
    return makeNode(L, State, Pred);
  }

  ExplodedNode *
  makeNodeWithBinding(ExplodedNode *Pred, const Expr *E, SVal V,
                      ProgramPoint::Kind K = ProgramPoint::PostStmtKind) const {
    return makeNodeWithBinding(Pred, E, V, Pred->getState(), K);
  }

  /// Enqueue the given set of nodes onto the work list.
  void enqueue(ExplodedNodeSet &Set);

  /// Enqueue nodes that were created as a result of processing
  /// a statement onto the work list.
  void enqueueStmtNodes(ExplodedNodeSet &Set, const CFGBlock *Block,
                        unsigned Idx);

  /// enqueue the nodes corresponding to the end of function onto the
  /// end of path / work list.
  void enqueueEndOfFunction(ExplodedNodeSet &Set, const ReturnStmt *RS);

  /// Enqueue a single node created as a result of statement processing.
  void enqueueStmtNode(ExplodedNode *N, const CFGBlock *Block, unsigned Idx);

  DataTag::Factory &getDataTags() { return DataTags; }
};

} // namespace ento

} // namespace clang

#endif // LLVM_CLANG_STATICANALYZER_CORE_PATHSENSITIVE_COREENGINE_H
