//===- MarkUnreachableTargets.cpp ----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass marks OpenMP target operations that are in unreachable code
// with an attribute. This allows device compilation to skip generating code
// for such ops.
//
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/OpenMP/Passes.h"

#include "flang/Optimizer/Dialect/FIRDialect.h"
#include "flang/Optimizer/Dialect/FIROps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/SmallSet.h"

namespace flangomp {
#define GEN_PASS_DEF_MARKUNREACHABLETARGETSPASS
#include "flang/Optimizer/OpenMP/Passes.h.inc"
} // namespace flangomp

using namespace mlir;

namespace {

/// Check if an operation is nested inside a fir.if with a constant false
/// condition.
static bool isInUnreachableIfBlock(Operation *op) {
  Operation *current = op;

  // Walk up through parent operations
  while (current) {
    Operation *parentOp = current->getParentOp();
    if (!parentOp)
      break;

    // Check for fir.if with constant false condition
    if (auto firIf = dyn_cast<fir::IfOp>(parentOp)) {
      if (auto constOp =
              firIf.getCondition().getDefiningOp<arith::ConstantOp>()) {
        if (auto intAttr = dyn_cast<IntegerAttr>(constOp.getValue())) {
          // If condition is false (0) and op is in the "then" region
          if (intAttr.getInt() == 0 &&
              current->getParentRegion() == &firIf.getThenRegion())
            return true;
          // If condition is true (non-zero) and op is in the "else" region
          if (intAttr.getInt() != 0 && !firIf.getElseRegion().empty() &&
              current->getParentRegion() == &firIf.getElseRegion())
            return true;
        }
      }
    }

    current = parentOp;
  }

  return false;
}

/// Check if a block is unreachable due to constant condition branches.
/// A block is unreachable only if ALL predecessors lead to it through
/// unreachable paths (i.e., constant false conditions).
/// This handles patterns like:
///   %false = arith.constant false
///   cf.cond_br %false, ^bb1, ^bb2
/// where ^bb1 is unreachable.
static bool isBlockUnreachable(Block *block) {
  // Entry blocks and blocks with no predecessors are reachable
  if (block->hasNoPredecessors())
    return false;

  // Check all predecessors - block is unreachable only if ALL paths are
  // provably unreachable via constant conditions
  for (Block *pred : block->getPredecessors()) {
    Operation *terminator = pred->getTerminator();

    // Check if this is a cf.cond_br with constant condition
    if (auto condBr = dyn_cast<cf::CondBranchOp>(terminator)) {
      // Try to get the constant value of the condition
      if (auto constOp =
              condBr.getCondition().getDefiningOp<arith::ConstantOp>()) {
        if (auto intAttr = dyn_cast<IntegerAttr>(constOp.getValue())) {
          bool condIsTrue = intAttr.getInt() != 0;

          // If condition is false and block is the true destination,
          // this path is unreachable - continue checking other predecessors
          if (!condIsTrue && block == condBr.getTrueDest())
            continue;
          // If condition is true and block is the false destination,
          // this path is unreachable - continue checking other predecessors
          if (condIsTrue && block == condBr.getFalseDest())
            continue;
          // Otherwise, this path IS reachable (condition matches destination)
          return false;
        }
      }
    }

    // If we reach here, this predecessor either:
    // - is not a CondBranchOp, OR
    // - doesn't have a constant condition
    // Either way, this path could be taken, so block is reachable
    return false;
  }

  // All predecessors lead to this block through unreachable paths
  return true;
}

/// Recursively check if an operation is in an unreachable block.
/// This walks up the block hierarchy to check if any containing block
/// is unreachable, handling both fir.if and cf.cond_br patterns.
static bool isOperationUnreachable(Operation *op) {
  // First check for fir.if patterns (before SCF lowering)
  if (isInUnreachableIfBlock(op))
    return true;

  // Then check for cf.cond_br patterns (after SCF lowering)
  Block *currentBlock = op->getBlock();

  // Walk up through nested regions checking each block
  while (currentBlock) {
    if (isBlockUnreachable(currentBlock))
      return true;

    // Move to parent operation's block
    Operation *parentOp = currentBlock->getParentOp();
    if (!parentOp || isa<ModuleOp>(parentOp) || isa<func::FuncOp>(parentOp))
      break;

    currentBlock = parentOp->getBlock();
  }

  return false;
}

class MarkUnreachableTargetsPass
    : public flangomp::impl::MarkUnreachableTargetsPassBase<
          MarkUnreachableTargetsPass> {
public:
  MarkUnreachableTargetsPass() = default;

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    auto module = getOperation();

    // Walk all target operations and mark those that are unreachable
    module.walk([&](omp::TargetOp targetOp) {
      if (isOperationUnreachable(targetOp.getOperation()))
        targetOp->setAttr("omp.target_unreachable", UnitAttr::get(context));
    });
  }
};

} // namespace
