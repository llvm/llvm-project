// Normalize XeMachine register aliases before scheduling and allocation.

#include "inter/Dialect/XeMachine/IR/XeMachine.h"
#include "inter/Dialect/XeMachine/IR/XeMachineRegAllocPreparation.h"
#include "inter/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace inter {
#define GEN_PASS_DEF_PREPAREREGALLOC
#include "inter/Transforms/Passes.h.inc"
} // namespace inter

using namespace mlir;

namespace {

class PrepareRegAllocPass
    : public inter::impl::PrepareRegAllocBase<PrepareRegAllocPass> {
public:
  void runOnOperation() override {
    func::FuncOp function = getOperation();
    if (function.isExternal())
      return;
    if (failed(inter::xemachine::prepareRegisterAllocation(function)))
      return signalPassFailure();
  }
};

} // namespace
