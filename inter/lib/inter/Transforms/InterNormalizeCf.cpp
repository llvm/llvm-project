// inter-normalize-cf: llvm.func -> func.func, llvm branches -> cf branches.
// Prepares imported kernels for the upstream lift-cf-to-scf pass.

#include "inter/Transforms/Passes.h"

#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"

namespace inter {
#define GEN_PASS_DEF_NORMALIZECF
#include "inter/Transforms/Passes.h.inc"
} // namespace inter

using namespace mlir;

namespace {

struct NormalizeCf : public inter::impl::NormalizeCfBase<NormalizeCf> {
  void runOnOperation() override {
    SmallVector<LLVM::LLVMFuncOp> funcs;
    getOperation().walk([&](LLVM::LLVMFuncOp f) {
      if (!f.getBody().empty())
        funcs.push_back(f);
    });
    for (LLVM::LLVMFuncOp f : funcs)
      if (failed(convertFunc(f)))
        return signalPassFailure();
  }

  LogicalResult convertFunc(LLVM::LLVMFuncOp llvmFunc) {
    OpBuilder b(llvmFunc);
    auto func = b.create<func::FuncOp>(
        llvmFunc.getLoc(), llvmFunc.getName(),
        b.getFunctionType(llvmFunc.getArgumentTypes(),
                          llvmFunc.getResultTypes()));
    func->setAttr("xemachine.kernel", b.getUnitAttr());
    Region &body = llvmFunc.getBody();
    func.getBody().takeBody(body);

    SmallVector<Operation *> toConvert;
    func.walk([&](Operation *op) {
      if (isa<LLVM::BrOp, LLVM::CondBrOp, LLVM::ReturnOp>(op))
        toConvert.push_back(op);
    });
    for (Operation *op : toConvert) {
      OpBuilder ob(op);
      if (auto br = dyn_cast<LLVM::BrOp>(op)) {
        cf::BranchOp::create(ob, br.getLoc(), br.getDestOperands(), br.getDest());
      } else if (auto cbr = dyn_cast<LLVM::CondBrOp>(op)) {
        cf::CondBranchOp::create(ob, cbr.getLoc(), cbr.getCondition(),
                                 cbr.getTrueDest(), cbr.getTrueDestOperands(),
                                 cbr.getFalseDest(),
                                 cbr.getFalseDestOperands());
      } else if (auto ret = dyn_cast<LLVM::ReturnOp>(op)) {
        func::ReturnOp::create(ob, ret.getLoc());
      }
      op->erase();
    }
    llvmFunc.erase();
    return success();
  }
};

} // namespace
