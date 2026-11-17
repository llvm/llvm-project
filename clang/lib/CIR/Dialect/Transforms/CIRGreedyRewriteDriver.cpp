//===- CIRGreedyRewriteDriver.cpp - Greedy rewriting across jumps ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements cir::applyOpPatternsGreedily. It is the worklist loop of
// mlir::applyOpPatternsGreedily under its default configuration, except that it
// asks cir::GotoAnalysis instead of the CFG whether control reaches an
// operation, and that it bounds how often an operation folds in place.
//
//===----------------------------------------------------------------------===//

#include "clang/CIR/Dialect/Transforms/CIRTransformUtils.h"

#include "mlir/Config/mlir-config.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Rewrite/PatternApplicator.h"
#include "clang/CIR/Dialect/Analysis/GotoAnalysis.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;

namespace {

/// Operations to visit, popped from the back. Pushing an operation that is
/// already on the list does nothing, and removing one leaves a null entry that
/// pops skip, as in MLIR's greedy driver.
class Worklist {
public:
  bool empty() const { return positions.empty(); }

  void push(Operation *op) {
    if (positions.try_emplace(op, list.size()).second)
      list.push_back(op);
  }

  Operation *pop() {
    while (!list.back())
      list.pop_back();
    Operation *op = list.pop_back_val();
    positions.erase(op);
    return op;
  }

  void remove(Operation *op) {
    auto it = positions.find(op);
    if (it == positions.end())
      return;
    list[it->second] = nullptr;
    positions.erase(it);
  }

private:
  SmallVector<Operation *> list;
  DenseMap<Operation *, unsigned> positions;
};

/// The most times an operation folds in place in one run. On a cycle of SSA
/// values, a fold can return the operation's own result, which counts as an
/// in-place change, forever. Code that only a jump enters may hold such a
/// cycle, because the verifier checks dominance only in blocks with a CFG path
/// from the entry. `OpBuilder::tryFold` bounds in-place folds the same way.
constexpr unsigned kMaxInPlaceFolds = 64;

/// The worklist loop of `mlir::applyOpPatternsGreedily` under its default
/// configuration: any operation may be enqueued, folding is on, and the number
/// of rewrites is not limited.
class GreedyRewriteDriver final : public RewriterBase::Listener {
public:
  GreedyRewriteDriver(MLIRContext *context,
                      const FrozenRewritePatternSet &patterns, Region &scope)
      : rewriter(context), matcher(patterns), scope(scope),
        reachability(scope.getParentOp()) {
    matcher.applyDefaultCostModel();
    rewriter.setListener(this);
  }

  /// Rewrite `ops` and every operation the rewrites touch. With
  /// `eraseOnlyNewlyDead`, dead code elimination spares the operations of
  /// `ops` that are unused now. Return whether the IR changed.
  bool run(ArrayRef<Operation *> ops, bool eraseOnlyNewlyDead);

private:
  bool isReachable(Operation *op);
  bool fold(Operation *op);
  void addToWorklist(Operation *op);
  void addOperandsToWorklist(Operation *op);

  void notifyBlockInserted(Block *block, Region *previous,
                           Region::iterator previousIt) override;
  void notifyBlockErased(Block *block) override;
  void notifyOperationInserted(Operation *op,
                               OpBuilder::InsertPoint previous) override;
  void notifyOperationModified(Operation *op) override;
  void notifyOperationErased(Operation *op) override;

  PatternRewriter rewriter;
  PatternApplicator matcher;
  Region &scope;
  Worklist worklist;
  DeadnessCache deadness;
  cir::GotoAnalysis reachability;
  /// How many times each operation has folded in place.
  DenseMap<Operation *, unsigned> inPlaceFolds;
  /// Operations to keep even when they are trivially dead. With
  /// `eraseOnlyNewlyDead`, the given operations that start out unused.
  DenseSet<Operation *> keptIfDead;
};

} // namespace

bool GreedyRewriteDriver::run(ArrayRef<Operation *> ops,
                              bool eraseOnlyNewlyDead) {
  for (Operation *op : ops) {
    worklist.push(op);
    if (eraseOnlyNewlyDead && op->use_empty())
      keptIfDead.insert(op);
  }

  bool changed = false;
  while (!worklist.empty()) {
    Operation *op = worklist.pop();
    if (!isReachable(op))
      continue;

    if (!keptIfDead.contains(op) && deadness.isOpTriviallyDead(op)) {
      rewriter.eraseOp(op);
      changed = true;
      continue;
    }

    // A constant would fold to its attribute and be materialized again,
    // forever.
    if (!op->hasTrait<OpTrait::ConstantLike>() && fold(op)) {
      changed = true;
      continue;
    }

    changed |= succeeded(matcher.matchAndRewrite(op, rewriter));
  }
  return changed;
}

bool GreedyRewriteDriver::isReachable(Operation *op) {
  bool reachable = reachability.isReachable(op);
#if MLIR_ENABLE_EXPENSIVE_PATTERN_API_CHECKS
  // The analysis may over-approximate after a jump target moves, but it must
  // reach whatever a fresh analysis reaches.
  assert(
      (reachable || !cir::GotoAnalysis(scope.getParentOp()).isReachable(op)) &&
      "cached reachability misses an operation that control reaches");
#endif // MLIR_ENABLE_EXPENSIVE_PATTERN_API_CHECKS
  return reachable;
}

bool GreedyRewriteDriver::fold(Operation *op) {
  if (inPlaceFolds.lookup(op) == kMaxInPlaceFolds)
    return false;
  SmallVector<OpFoldResult> foldResults;
  if (failed(op->fold(foldResults)))
    return false;

  // The fold changed `op` in place, without notifying the rewriter.
  if (foldResults.empty()) {
    ++inPlaceFolds[op];
    notifyOperationModified(op);
    return true;
  }

  // Replace the results, materializing attributes as constants before `op`.
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(op);
  SmallVector<Value> replacements;
  for (auto [foldResult, type] :
       llvm::zip_equal(foldResults, op->getResultTypes())) {
    if (auto value = dyn_cast<Value>(foldResult)) {
      replacements.push_back(value);
      continue;
    }
    Operation *constant = op->getDialect()->materializeConstant(
        rewriter, cast<Attribute>(foldResult), type, op->getLoc());
    if (!constant) {
      // Erase what was materialized for the earlier results. Like MLIR's
      // driver, this expects the folder not to have mixed existing values
      // with attributes.
      llvm::SmallDenseSet<Operation *> materialized;
      for (Value replacement : replacements) {
        assert(replacement.use_empty() &&
               "folder reused an existing value for one result but "
               "materialization failed for another");
        materialized.insert(replacement.getDefiningOp());
      }
      for (Operation *materializedOp : materialized)
        rewriter.eraseOp(materializedOp);
      return false;
    }
    assert(constant->hasTrait<OpTrait::ConstantLike>() &&
           "materializeConstant produced an op that is not ConstantLike");
    assert(constant->getResultTypes()[0] == type &&
           "materializeConstant produced a result of the wrong type");
    replacements.push_back(constant->getResult(0));
  }
  rewriter.replaceOp(op, replacements);
  return true;
}

void GreedyRewriteDriver::addToWorklist(Operation *op) {
  // Push `op` and then its ancestors within the scope, as MLIR's driver does.
  // Push nothing for an operation outside the scope.
  SmallVector<Operation *, 8> ancestors;
  do {
    ancestors.push_back(op);
    Region *region = op->getParentRegion();
    if (region == &scope) {
      for (Operation *ancestor : ancestors)
        worklist.push(ancestor);
      return;
    }
    if (!region)
      return;
    op = region->getParentOp();
  } while (op);
}

void GreedyRewriteDriver::addOperandsToWorklist(Operation *op) {
  // Once `op` is gone, the definer of an operand with at most one other user
  // may be dead or simplify further.
  for (Value operand : op->getOperands()) {
    if (!operand)
      continue;
    Operation *definer = operand.getDefiningOp();
    if (!definer)
      continue;
    Operation *otherUser = nullptr;
    bool hasMoreThanTwoUsers = false;
    for (Operation *user : operand.getUsers()) {
      if (user == op || user == otherUser)
        continue;
      if (!otherUser) {
        otherUser = user;
        continue;
      }
      hasMoreThanTwoUsers = true;
      break;
    }
    if (!hasMoreThanTwoUsers)
      addToWorklist(definer);
  }
}

// The notifications first invalidate the deadness cache as MLIR's greedy driver
// does, then the reachability analysis, then update the worklist.

void GreedyRewriteDriver::notifyBlockInserted(Block *block, Region *previous,
                                              Region::iterator previousIt) {
  // Moving a block can change the context of every operation it contains.
  for (Operation &op : *block)
    deadness.invalidate(&op);
  if (previous)
    if (Operation *parent = previous->getParentOp())
      deadness.invalidateAncestors(parent);
  if (Operation *parent = block->getParentOp())
    deadness.invalidateAncestors(parent);
  reachability.notifyBlockInserted(block, previous);
}

void GreedyRewriteDriver::notifyBlockErased(Block *block) {
  if (Operation *parent = block->getParentOp())
    deadness.invalidateAncestors(parent);
  reachability.notifyBlockErased(block);
}

void GreedyRewriteDriver::notifyOperationInserted(
    Operation *op, OpBuilder::InsertPoint previous) {
  // The old parent loses a child, but its remaining descendants are unchanged.
  if (previous.isSet())
    if (Operation *parent = previous.getBlock()->getParentOp())
      deadness.invalidateAncestors(parent);
  deadness.invalidate(op);
  reachability.notifyOperationInserted(op, previous);
  addToWorklist(op);
}

void GreedyRewriteDriver::notifyOperationModified(Operation *op) {
  deadness.invalidate(op);
  reachability.notifyOperationModified(op);
  addToWorklist(op);
}

void GreedyRewriteDriver::notifyOperationErased(Operation *op) {
  assert(op != scope.getParentOp() &&
         "the operation that holds the scope must not be erased");
  deadness.invalidate(op);
  reachability.notifyOperationErased(op);
  inPlaceFolds.erase(op);
  keptIfDead.erase(op);
  addOperandsToWorklist(op);
  worklist.remove(op);
}

bool cir::applyOpPatternsGreedily(ArrayRef<Operation *> ops,
                                  const FrozenRewritePatternSet &patterns,
                                  Region &scope, bool eraseOnlyNewlyDead) {
  assert(llvm::all_of(ops,
                      [&](Operation *op) {
                        return scope.findAncestorOpInRegion(*op) != nullptr;
                      }) &&
         "operations must be within the scope");
  if (ops.empty())
    return false;
  return GreedyRewriteDriver(ops.front()->getContext(), patterns, scope)
      .run(ops, eraseOnlyNewlyDead);
}
