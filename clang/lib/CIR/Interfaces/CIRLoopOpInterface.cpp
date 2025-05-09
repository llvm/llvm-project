//===---------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===---------------------------------------------------------------------===//

#include "clang/CIR/Interfaces/CIRLoopOpInterface.h"

#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/Interfaces/CIRLoopOpInterface.cpp.inc"
#include "llvm/Support/ErrorHandling.h"

namespace cir {

void LoopOpInterface::getLoopOpSuccessorRegions(
    LoopOpInterface op, mlir::RegionBranchPoint point,
    llvm::SmallVectorImpl<mlir::RegionSuccessor> &regions) {
  assert(point.isParent() || point.getRegionOrNull());

  // Branching to first region: go to condition or body (do-while).
  if (point.isParent()) {
    regions.emplace_back(&op.getEntry(), op.getEntry().getArguments());
    return;
  }

  // Branching from condition: go to body or exit.
  if (&op.getCond() == point.getRegionOrNull()) {
    regions.emplace_back(mlir::RegionSuccessor(op->getResults()));
    regions.emplace_back(&op.getBody(), op.getBody().getArguments());
    return;
  }

  // Branching from body: go to step (for) or condition.
  if (&op.getBody() == point.getRegionOrNull()) {
    // FIXME(cir): Should we consider break/continue statements here?
    mlir::Region *afterBody =
        (op.maybeGetStep() ? op.maybeGetStep() : &op.getCond());
    regions.emplace_back(afterBody, afterBody->getArguments());
    return;
  }

  // Branching from step: go to condition.
  if (op.maybeGetStep() == point.getRegionOrNull()) {
    regions.emplace_back(&op.getCond(), op.getCond().getArguments());
    return;
  }

  llvm_unreachable("unexpected branch origin");
}

/// Verify invariants of the LoopOpInterface.
llvm::LogicalResult detail::verifyLoopOpInterface(mlir::Operation *op) {
  // FIXME: fix this so the conditionop isn't requiring MLIRCIR
  // auto loopOp = mlir::cast<LoopOpInterface>(op);
  // if (!mlir::isa<ConditionOp>(loopOp.getCond().back().getTerminator()))
  //   return op->emitOpError(
  //       "expected condition region to terminate with 'cir.condition'");
  return llvm::success();
}

} // namespace cir
