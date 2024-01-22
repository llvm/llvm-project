//===- CIRLoopOpInterface.cpp - Interface for CIR loop-like ops *- C++ -*-===//
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

namespace mlir {
namespace cir {

void LoopOpInterface::getLoopOpSuccessorRegions(
    LoopOpInterface op, RegionBranchPoint point,
    SmallVectorImpl<RegionSuccessor> &regions) {
  assert(point.isParent() || point.getRegionOrNull());

  // Branching to first region: go to condition or body (do-while).
  if (point.isParent()) {
    regions.emplace_back(&op.getEntry(), op.getEntry().getArguments());
  }
  // Branching from condition: go to body or exit.
  else if (&op.getCond() == point.getRegionOrNull()) {
    regions.emplace_back(RegionSuccessor(op->getResults()));
    regions.emplace_back(&op.getBody(), op.getBody().getArguments());
  }
  // Branching from body: go to step (for) or condition.
  else if (&op.getBody() == point.getRegionOrNull()) {
    // FIXME(cir): Should we consider break/continue statements here?
    auto *afterBody = (op.maybeGetStep() ? op.maybeGetStep() : &op.getCond());
    regions.emplace_back(afterBody, afterBody->getArguments());
  }
  // Branching from step: go to condition.
  else if (op.maybeGetStep() == point.getRegionOrNull()) {
    regions.emplace_back(&op.getCond(), op.getCond().getArguments());
  } else {
    llvm_unreachable("unexpected branch origin");
  }
}

/// Verify invariants of the LoopOpInterface.
LogicalResult detail::verifyLoopOpInterface(Operation *op) {
  auto loopOp = cast<LoopOpInterface>(op);
  if (!isa<ConditionOp>(loopOp.getCond().back().getTerminator()))
    return op->emitOpError(
        "expected condition region to terminate with 'cir.condition'");
  return success();
}

} // namespace cir
} // namespace mlir
