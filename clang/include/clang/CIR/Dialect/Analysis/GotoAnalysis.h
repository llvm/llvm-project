//===- GotoAnalysis.h - Reachability through jumps in CIR -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares an analysis of which operations control can reach in
// structured CIR, counting the jumps that are not CFG edges yet.
//
//===----------------------------------------------------------------------===//

#ifndef CLANG_CIR_DIALECT_ANALYSIS_GOTOANALYSIS_H
#define CLANG_CIR_DIALECT_ANALYSIS_GOTOANALYSIS_H

#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"

namespace cir {

/// Answers whether control can reach an operation in structured CIR.
///
/// Until cir-goto-solver runs, a `cir.goto` enters its `cir.label`, and a
/// `cir.switch` enters its `cir.case` operations, without a CFG edge. Code that
/// only such a jump enters has no CFG predecessor, so reachability that follows
/// CFG edges alone, such as MLIR's rewrite drivers use, misses it.
///
/// In each region, control enters at the entry block. It also enters at each
/// operation that is or holds a jump target: a `cir.label`, or an operation
/// whose regions contain one, short of an operation isolated from above; and a
/// `cir.case`, or an operation whose regions contain one, short of the case's
/// `cir.switch`. Control reaches the rest of the block from that operation on.
/// From each block it reaches, control follows the successor edges and enters
/// each successor at its start. An operation is reachable if control reaches it
/// in its block and reaches every enclosing operation below the root.
///
/// The analysis over-approximates. Control may enter an operation that holds a
/// jump target only through that target, yet every region of the operation
/// counts as entered. And an operation that held a jump target keeps counting
/// as a holder after a rewrite moves or erases that target.
///
/// As the listener of a rewriter, the analysis stays valid while the rewriter
/// changes the IR. Its notifications only record what changed, and the next
/// query brings the results up to date.
class GotoAnalysis : public mlir::RewriterBase::Listener {
public:
  /// Analyze the operations nested in `root`.
  explicit GotoAnalysis(mlir::Operation *root);

  /// Return whether control can reach `op`, which must be nested in the root.
  /// Call this only while the IR is in a consistent state.
  bool isReachable(mlir::Operation *op);

  void notifyOperationInserted(mlir::Operation *op,
                               mlir::OpBuilder::InsertPoint previous) override;
  void notifyOperationModified(mlir::Operation *op) override;
  void notifyOperationErased(mlir::Operation *op) override;
  void notifyBlockInserted(mlir::Block *block, mlir::Region *previous,
                           mlir::Region::iterator previousIt) override;
  void notifyBlockErased(mlir::Block *block) override;

private:
  /// A block that control reaches.
  struct ReachedBlock {
    /// The first holder of a jump target in the block, where control enters
    /// it, or null if an edge reaches the whole block.
    mlir::Operation *entry = nullptr;
    /// The successors of the block in its region when it was last updated.
    llvm::SmallVector<mlir::Block *, 2> successors;
  };

  /// What control reaches in a region, and how the region changed since.
  struct RegionState {
    /// Whether the results must be computed from scratch.
    bool stale = true;
    /// The entry block when the results were computed.
    mlir::Block *entry = nullptr;
    /// The blocks that control reaches.
    llvm::DenseMap<mlir::Block *, ReachedBlock> reached;
    /// Blocks whose terminator changed since the last update.
    llvm::SmallSetVector<mlir::Block *, 4> changed;
    /// Blocks that left the region since the last update. They are compared,
    /// never dereferenced.
    llvm::SmallDenseSet<mlir::Block *, 4> removed;
    /// Holders that arrived in the region since the last update.
    llvm::SmallVector<mlir::Operation *, 2> gained;
    /// Holders that left a block of the region since the last update, with
    /// that block. They are compared, never dereferenced.
    llvm::SmallVector<std::pair<mlir::Block *, mlir::Operation *>, 2> lost;
  };

  /// Mark `target` and the operations that hold it.
  void addHolders(mlir::Operation *target);
  /// Mark the jump targets nested in `op`, `op` included, and their holders.
  void addNestedHolders(mlir::Operation *op);
  /// Return the state of `region`, brought up to date.
  RegionState &getState(mlir::Region *region);
  /// Compute the state of `region` from scratch.
  void compute(mlir::Region *region, RegionState &state);
  /// Bring the state of `region` up to date with the recorded changes. Fail if
  /// a change may have disconnected a block, which only a full computation can
  /// tell.
  mlir::LogicalResult update(mlir::Region *region, RegionState &state);
  /// Record that control reaches `block` from `entry` on, and what it reaches
  /// from there.
  void reach(RegionState &state, mlir::Block *block, mlir::Operation *entry);
  /// Record a change in the state of a region, if it has one.
  void markChanged(mlir::Block *block);
  void markRemoved(mlir::Region *region, mlir::Block *block);
  void markGained(mlir::Operation *holder);
  void markLost(mlir::Block *block, mlir::Operation *holder);
  void markStale(mlir::Region *region);

  mlir::Operation *root;
  /// The jump targets and the operations that hold them.
  llvm::DenseSet<mlir::Operation *> holders;
  /// The state of each region queried so far.
  llvm::DenseMap<mlir::Region *, RegionState> regions;
};

} // namespace cir

#endif // CLANG_CIR_DIALECT_ANALYSIS_GOTOANALYSIS_H
