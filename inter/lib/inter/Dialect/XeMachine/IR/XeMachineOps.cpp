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
// exec_if: parent -> arms/fallthrough; then -> else/parent; else -> parent.
// uniform_if: parent -> either arm/fallthrough; regions -> parent.
// uniform_loop: parent -> body; body -> body (back-edge) or parent (exit).
//===----------------------------------------------------------------------===//

void ExecIfOp::getSuccessorRegions(RegionBranchPoint point,
                                   SmallVectorImpl<RegionSuccessor> &regions) {
  bool hasElse = !getElseRegion().empty();
  if (point.isParent()) {
    regions.emplace_back(&getThenRegion());
    if (hasElse)
      regions.emplace_back(&getElseRegion());
    else if (getNumResults() == 0)
      regions.emplace_back(getOperation());
    return;
  }

  Region *source =
      point.getTerminatorPredecessorOrNull()->getParentRegion();
  if (source == &getThenRegion() && hasElse)
    regions.emplace_back(&getElseRegion());
  regions.emplace_back(getOperation());
}

void UniformIfOp::getSuccessorRegions(RegionBranchPoint point,
                                      SmallVectorImpl<RegionSuccessor> &regions) {
  if (point.isParent()) {
    regions.emplace_back(&getThenRegion());
    if (getElseRegion().empty())
      regions.emplace_back(getOperation());
    else
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
                                  : getBody().getArguments();
}

OperandRange
UniformLoopOp::getEntrySuccessorOperands(RegionSuccessor successor) {
  return getInits();
}

MutableOperandRange
YieldOp::getMutableSuccessorOperands(RegionSuccessor successor) {
  MutableOperandRange values = getValuesMutable();
  if (successor.isOperation())
    return values;
  return values.slice(0, 0);
}

MutableOperandRange
ContinueIfOp::getMutableSuccessorOperands(RegionSuccessor point) {
  // Operand 0 is the condition; only carried values flow to successors.
  return MutableOperandRange(getOperation(), /*start=*/1,
                             getNumOperands() - 1);
}
