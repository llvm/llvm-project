// inter-convert-calls: llvm.call builtins -> xw special ops.
// Token wiring is not done here; inter-convert-memory owns ordering.

#include "inter/Dialect/Inter/IR/XW.h"
#include "inter/Support/Builtins.h"
#include "inter/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"

namespace inter {
#define GEN_PASS_DEF_CONVERTCALLS
#include "inter/Transforms/Passes.h.inc"
} // namespace inter

using namespace mlir;

namespace {

struct ConvertCalls : public inter::impl::ConvertCallsBase<ConvertCalls> {
  void runOnOperation() override {
    func::FuncOp func = getOperation();
    if (!func->hasAttr("xemachine.kernel"))
      return;
    SmallVector<LLVM::CallOp> calls;
    func.walk([&](LLVM::CallOp call) { calls.push_back(call); });
    for (LLVM::CallOp call : calls) {
      if (failed(convertCall(call)))
        return signalPassFailure();
    }
  }

  int constOperand(Value v) {
    auto cst = v.getDefiningOp<LLVM::ConstantOp>();
    if (!cst)
      return -1;
    return cast<IntegerAttr>(cst.getValue()).getInt();
  }

  LogicalResult convertCall(LLVM::CallOp call) {
    auto callee = call.getCallee();
    if (!callee)
      return call.emitOpError("indirect function calls are not supported"),
             failure();

    OpBuilder b(call);
    Location loc = call.getLoc();
    Type i64 = b.getI64Type();

    if (*callee == inter::builtins::kGetGlobalId) {
      int dim =
          call.getNumOperands() ? constOperand(call.getArgOperands()[0]) : -1;
      if (dim < 0)
        return emitError(call.getLoc(), "non-constant id dimension"), failure();
      auto op = xw::GlobalIdOp::create(b, loc, i64, b.getI32IntegerAttr(dim));
      call->replaceAllUsesWith(ValueRange{op.getId()});
    } else if (*callee == inter::builtins::kGetLocalId) {
      int dim =
          call.getNumOperands() ? constOperand(call.getArgOperands()[0]) : -1;
      if (dim < 0)
        return emitError(call.getLoc(), "non-constant id dimension"), failure();
      auto op = xw::LocalIdOp::create(b, loc, i64, b.getI32IntegerAttr(dim));
      call->replaceAllUsesWith(ValueRange{op.getId()});
    } else if (*callee == inter::builtins::kBarrier) {
      xw::BarrierOp::create(
          b, loc, inter::xemachine::MemTokenType::get(call.getContext()),
          /*dependency=*/Value());
    } else if (*callee == inter::builtins::kAtomicAdd) {
      auto op = xw::AtomicAddOp::create(
          b, loc, b.getI32Type(),
          inter::xemachine::MemTokenType::get(call.getContext()),
          call.getArgOperands()[0], call.getArgOperands()[1],
          /*dependency=*/Value());
      call->replaceAllUsesWith(ValueRange{op.getOld()});
    } else {
      call.emitOpError("function calls are not supported; '")
          << *callee << "' is not a recognized builtin";
      return failure();
    }
    call->erase();
    return success();
  }
};

} // namespace
