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
#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"
#include "mlir/Analysis/DataFlow/Utils.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/SmallVector.h"

namespace flangomp {
#define GEN_PASS_DEF_DELETEUNREACHABLETARGETSPASS
#include "flang/Optimizer/OpenMP/Passes.h.inc"
} // namespace flangomp

using namespace mlir;

namespace {

/// Check if an operation is unreachable using DeadCodeAnalysis.
static bool isOperationUnreachable(Operation *op, DataFlowSolver &solver) {
  Block *block = op->getBlock();
  if (!block)
    return false;

  // Query DeadCodeAnalysis to check if the block is live (reachable).
  ProgramPoint *point = solver.getProgramPointBefore(block);
  const dataflow::Executable *executable =
      solver.lookupState<dataflow::Executable>(point);

  return (executable && !executable->isLive());
}

class DeleteUnreachableTargetsPass
    : public flangomp::impl::DeleteUnreachableTargetsPassBase<
          DeleteUnreachableTargetsPass> {
public:
  DeleteUnreachableTargetsPass() = default;

  void runOnOperation() override {
    auto module = getOperation();
    DataFlowSolver solver;
    dataflow::loadBaselineAnalyses(solver);

    if (failed(solver.initializeAndRun(module))) {
      signalPassFailure();
      return;
    }

    // Collect unreachable target operations
    SmallVector<omp::TargetOp> unreachableTargets;
    module.walk([&](omp::TargetOp targetOp) {
      if (isOperationUnreachable(targetOp.getOperation(), solver))
        unreachableTargets.push_back(targetOp);
    });

    // Delete unreachable target operations
    for (omp::TargetOp targetOp : unreachableTargets)
      targetOp->erase();
  }
};

} // namespace
