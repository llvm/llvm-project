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
  assert(point.isParent() || point.getTerminatorPredecessorOrNull());

  // Branching to first region: go to condition or body (do-while).
  if (point.isParent()) {
    regions.emplace_back(&op.getEntry());
    return;
  }

  mlir::Region *parentRegion =
      point.getTerminatorPredecessorOrNull()->getParentRegion();

  // Branching from condition: go to body or exit.
  if (&op.getCond() == parentRegion) {
    regions.emplace_back(mlir::RegionSuccessor::parent());
    regions.emplace_back(&op.getBody());
    return;
  }

  // Branching from body: go to step (for) or condition.
  if (&op.getBody() == parentRegion) {
    // FIXME(cir): Should we consider break/continue statements here?
    mlir::Region *afterBody =
        (op.maybeGetStep() ? op.maybeGetStep() : &op.getCond());
    regions.emplace_back(afterBody);
    return;
  }

  // Branching from step: go to condition.
  if (op.maybeGetStep() == parentRegion) {
    regions.emplace_back(&op.getCond());
    return;
  }

  llvm_unreachable("unexpected branch origin");
}

mlir::ValueRange
LoopOpInterface::getLoopOpSuccessorInputs(LoopOpInterface op,
                                          mlir::RegionSuccessor successor) {
  if (successor.isParent())
    return op->getResults();
  if (successor == &op.getEntry())
    return op.getEntry().getArguments();
  if (successor == &op.getBody())
    return op.getBody().getArguments();
  mlir::Region *afterBody =
      (op.maybeGetStep() ? op.maybeGetStep() : &op.getCond());
  if (successor == afterBody)
    return afterBody->getArguments();
  if (successor == &op.getCond())
    return op.getCond().getArguments();
  llvm_unreachable("invalid region successor");
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
