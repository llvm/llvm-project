//===- DeleteUnreachableTargets.cpp --------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass removes OpenMP target operations that are in unreachable code.
// This ensures host and device compilation have consistent target regions.
//
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/Dialect/FIRDialect.h"
#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/OpenMP/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/SmallSet.h"

namespace flangomp {
#define GEN_PASS_DEF_DELETEUNREACHABLETARGETSPASS
#include "flang/Optimizer/OpenMP/Passes.h.inc"
} // namespace flangomp

using namespace mlir;

namespace {

/// Check if an operation is unreachable from the entry block of its function.
/// This function detects unreachability in two ways:
/// 1. Constant conditions in structured control flow (e.g., fir.if with
///    constant false condition)
/// 2. Block-level unreachability (using DominanceInfo)
static bool isOperationUnreachable(Operation *op, DominanceInfo &domInfo) {
  // Walk up through parent operations to check for constant conditions in
  // structured control flow (fir.if)
  Operation *current = op;
  while (current) {
    Operation *parentOp = current->getParentOp();
    if (!parentOp)
      break;

    // Check for fir.if with constant condition.
    // This catches Fortran constructs like "if (.false.) then ... end if".
    if (auto firIf = dyn_cast<fir::IfOp>(parentOp)) {
      IntegerAttr constAttr;
      if (matchPattern(firIf.getCondition(), m_Constant(&constAttr))) {
        // If condition is false (0) and op is in the "then" region
        if (constAttr.getInt() == 0 &&
            current->getParentRegion() == &firIf.getThenRegion())
          return true;
        // If condition is true (non-zero) and op is in the "else" region
        if (constAttr.getInt() != 0 && !firIf.getElseRegion().empty() &&
            current->getParentRegion() == &firIf.getElseRegion())
          return true;
      }
    }

    // Stop at function boundary
    if (isa<func::FuncOp>(parentOp))
      break;

    current = parentOp;
  }

  // Check block-level reachability using DominanceInfo.
  // This detects blocks that are unreachable from the function entry due to
  // control flow (e.g., blocks with no predecessors, disconnected blocks).

  // Find the ancestor block that is in the function's main region.
  // We need to check if that block is reachable, not just the immediate
  // block containing the operation (which might be inside a fir.if region).
  auto funcOp = op->getParentOfType<func::FuncOp>();
  if (!funcOp)
    return false;

  Region *funcRegion = &funcOp.getRegion();
  Operation *ancestor = op;
  Block *blockInFuncRegion = nullptr;

  // Walk up to find the block in the function's main region
  while (ancestor) {
    Block *block = ancestor->getBlock();
    if (block && block->getParent() == funcRegion) {
      blockInFuncRegion = block;
      break;
    }
    ancestor = ancestor->getParentOp();
    if (!ancestor || ancestor == funcOp.getOperation())
      break;
  }

  // Check if we found a block in the function's main region and if it's
  // reachable
  if (blockInFuncRegion) {
    Block *entryBlock = &funcRegion->front();
    if (blockInFuncRegion == entryBlock)
      return false;
    return !domInfo.isReachableFromEntry(blockInFuncRegion);
  }

  return false;
}

class DeleteUnreachableTargetsPass
    : public flangomp::impl::DeleteUnreachableTargetsPassBase<
          DeleteUnreachableTargetsPass> {
public:
  DeleteUnreachableTargetsPass() = default;

  void runOnOperation() override {
    auto module = getOperation();

    module.walk([&](func::FuncOp funcOp) {
      // Create dominance info for this function
      DominanceInfo domInfo(funcOp);

      // Collect unreachable target operations in this function
      SmallVector<omp::TargetOp> unreachableTargets;
      funcOp.walk([&](omp::TargetOp targetOp) {
        if (isOperationUnreachable(targetOp.getOperation(), domInfo))
          unreachableTargets.push_back(targetOp);
      });

      // Delete unreachable target operations
      for (omp::TargetOp targetOp : unreachableTargets)
        targetOp->erase();
    });
  }
};

} // namespace
