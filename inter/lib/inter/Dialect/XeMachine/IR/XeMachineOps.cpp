#include "inter/Dialect/XeMachine/IR/XeMachine.h"

#include "mlir/IR/OpImplementation.h"

using namespace mlir;
using namespace inter::xemachine;

#define GET_OP_CLASSES
#include "inter/Dialect/XeMachine/IR/XeMachineOps.cpp.inc"

#define GET_OP_INTERFACE_CLASSES
#include "inter/Dialect/XeMachine/IR/XeMachineInterfaces.cpp.inc"

//===----------------------------------------------------------------------===//
// Structured control flow region modeling.
//
// exec_if/uniform_if: parent -> both regions; regions -> parent (results).
// uniform_loop: parent -> body; body -> body (back-edge) or parent (exit).
//===----------------------------------------------------------------------===//

void ExecIfOp::getSuccessorRegions(RegionBranchPoint point,
                                   SmallVectorImpl<RegionSuccessor> &regions) {
  if (point.isParent()) {
    regions.emplace_back(&getThenRegion());
    if (!getElseRegion().empty())
      regions.emplace_back(&getElseRegion());
    return;
  }
  regions.emplace_back(getOperation());
}

void UniformIfOp::getSuccessorRegions(RegionBranchPoint point,
                                      SmallVectorImpl<RegionSuccessor> &regions) {
  if (point.isParent()) {
    regions.emplace_back(&getThenRegion());
    if (!getElseRegion().empty())
      regions.emplace_back(&getElseRegion());
    return;
  }
  regions.emplace_back(getOperation());
}

void UniformLoopOp::getSuccessorRegions(
    RegionBranchPoint point, SmallVectorImpl<RegionSuccessor> &regions) {
  if (point.isParent()) {
    regions.emplace_back(&getBody());
    return;
  }
  regions.emplace_back(&getBody());
  regions.emplace_back(getOperation());
}

//===----------------------------------------------------------------------===//
// Operand -> successor-input mapping.
//===----------------------------------------------------------------------===//

ValueRange ExecIfOp::getSuccessorInputs(RegionSuccessor successor) {
  return successor.isOperation() ? ValueRange(getResults()) : ValueRange();
}

ValueRange UniformIfOp::getSuccessorInputs(RegionSuccessor successor) {
  return successor.isOperation() ? ValueRange(getResults()) : ValueRange();
}

ValueRange UniformLoopOp::getSuccessorInputs(RegionSuccessor successor) {
  return successor.isOperation() ? ValueRange(getResults())
                                 : ValueRange(getInits());
}

OperandRange UniformLoopOp::getEntrySuccessorOperands(RegionSuccessor successor) {
  return getInits();
}

MutableOperandRange
ContinueIfOp::getMutableSuccessorOperands(RegionSuccessor point) {
  // Operand 0 is the condition; only carried values flow to successors.
  return MutableOperandRange(getOperation(), /*start=*/1,
                             getNumOperands() - 1);
}
