// inter-normalize-cf: llvm.func -> func.func, llvm branches -> cf branches.
// Prepares imported kernels for the upstream lift-cf-to-scf pass.

#include "inter/Transforms/Passes.h"

#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Interfaces/FunctionInterfaces.h"

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

    for (LLVM::LLVMFuncOp f : funcs) {
      if (f.getCConv() != LLVM::CConv::SPIR_KERNEL) {
        f.emitOpError("defined helper functions are not supported; inline "
                      "calls before compiling with Inter");
        return signalPassFailure();
      }
      if (f.isVarArg()) {
        f.emitOpError("variadic kernels are not supported");
        return signalPassFailure();
      }
      if (f.getNumResults() != 0) {
        f.emitOpError("kernel return values are not supported");
        return signalPassFailure();
      }
    }

    for (LLVM::LLVMFuncOp f : funcs)
      if (failed(convertFunc(f)))
        return signalPassFailure();
  }

  LogicalResult convertFunc(LLVM::LLVMFuncOp llvmFunc) {
    OpBuilder b(llvmFunc);
    auto func =
        func::FuncOp::create(b, llvmFunc.getLoc(), llvmFunc.getName(),
                             b.getFunctionType(llvmFunc.getArgumentTypes(),
                                               llvmFunc.getResultTypes()));
    cast<FunctionOpInterface>(func.getOperation())
        .setVisibility(llvmFunc.getVisibility());
    if (ArrayAttr attrs = llvmFunc.getAllArgAttrs())
      func.setAllArgAttrs(attrs);
    if (ArrayAttr attrs = llvmFunc.getAllResultAttrs())
      func.setAllResultAttrs(attrs);
    for (NamedAttribute attr : llvmFunc->getDiscardableAttrs())
      func->setDiscardableAttr(attr.getName(), attr.getValue());

    // Keep LLVM-only function properties inspectable until the semantic import
    // gives each supported property an Inter representation.
    func->setAttr("xemachine.llvm_func_properties",
                  llvmFunc->getPropertiesAsAttribute());
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
        cf::BranchOp newBranch = cf::BranchOp::create(
            ob, br.getLoc(), br.getDestOperands(), br.getDest());
        if (Attribute annotation = br.getLoopAnnotationAttr())
          newBranch->setAttr("llvm.loop_annotation", annotation);
      } else if (auto cbr = dyn_cast<LLVM::CondBrOp>(op)) {
        cf::CondBranchOp newBranch = cf::CondBranchOp::create(
            ob, cbr.getLoc(), cbr.getCondition(), cbr.getTrueDest(),
            cbr.getTrueDestOperands(), cbr.getFalseDest(),
            cbr.getFalseDestOperands());
        if (DenseI32ArrayAttr weights = cbr.getBranchWeightsAttr())
          newBranch.setBranchWeightsAttr(weights);
        if (Attribute annotation = cbr.getLoopAnnotationAttr())
          newBranch->setAttr("llvm.loop_annotation", annotation);
      } else if (auto ret = dyn_cast<LLVM::ReturnOp>(op)) {
        func::ReturnOp::create(ob, ret.getLoc(), ret.getOperands());
      }
      op->erase();
    }
    llvmFunc.erase();
    return success();
  }
};

} // namespace
