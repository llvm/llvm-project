// Select the Xe hardware model and run the generic machine scheduler.

#include "Xe2ScheduleModel.h"

#include "inter/Dialect/XeMachine/IR/XeMachine.h"
#include "inter/Transforms/MachineScheduler.h"
#include "inter/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace inter {
#define GEN_PASS_DEF_MACHINESCHEDULE
#include "inter/Transforms/Passes.h.inc"
} // namespace inter

using namespace mlir;
using namespace inter::xemachine;

namespace {

class MachineSchedulePass
    : public inter::impl::MachineScheduleBase<MachineSchedulePass> {
public:
  void runOnOperation() override {
    func::FuncOp function = getOperation();
    if (function.isExternal())
      return;

    WalkResult hasMachineOperation = function.walk(
        [](InstructionIssueOpInterface) { return WalkResult::interrupt(); });
    if (!hasMachineOperation.wasInterrupted())
      return;

    TargetAttr target = function->getAttrOfType<TargetAttr>(kTargetAttrName);
    if (!target) {
      function.emitError("machine scheduling requires a target attribute");
      return signalPassFailure();
    }
    if (target.getChip().getValue() != "bmg") {
      function.emitError("machine scheduling does not support target '")
          << target.getChip().getValue() << "'";
      return signalPassFailure();
    }

    FailureOr<std::unique_ptr<inter::MachineScheduleModel>> model =
        inter::createXe2ScheduleModel(function);
    if (failed(model) ||
        failed(inter::scheduleMachineRegion(function.getBody(), **model)))
      return signalPassFailure();
  }
};

} // namespace
