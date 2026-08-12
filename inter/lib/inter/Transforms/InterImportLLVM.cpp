#include "inter/Dialect/Inter/IR/XW.h"

#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "inter/Transforms/Passes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "llvm/Support/MathExtras.h"

namespace inter {
#define GEN_PASS_DEF_IMPORTLLVM
#define GEN_PASS_DEF_VERIFYSTRUCTURED
#include "inter/Transforms/Passes.h.inc"
} // namespace inter.

using namespace mlir;

namespace {

struct ImportLLVM final : inter::impl::ImportLLVMBase<ImportLLVM> {
  LogicalResult importFunction(LLVM::LLVMFuncOp llvmFunction) {
    if (llvmFunction.getCConv() != LLVM::CConv::SPIR_KERNEL)
      return llvmFunction.emitOpError(
                 "defined helpers must be inlined before LLVM import"),
             failure();
    if (llvmFunction.isVarArg())
      return llvmFunction.emitOpError("variadic kernels are unsupported"),
             failure();

    OpBuilder builder(llvmFunction);
    func::FuncOp function = func::FuncOp::create(
        builder, llvmFunction.getLoc(), llvmFunction.getName(),
        builder.getFunctionType(llvmFunction.getArgumentTypes(),
                                llvmFunction.getResultTypes()));
    cast<FunctionOpInterface>(function.getOperation())
        .setVisibility(llvmFunction.getVisibility());
    if (ArrayAttr attrs = llvmFunction.getAllArgAttrs())
      function.setAllArgAttrs(attrs);
    if (ArrayAttr attrs = llvmFunction.getAllResultAttrs())
      function.setAllResultAttrs(attrs);

    SmallVector<Attribute> descriptors;
    for (auto [index, type] : llvm::enumerate(llvmFunction.getArgumentTypes())) {
      NamedAttrList descriptor;
      descriptor.set("offset", builder.getI64IntegerAttr(24 + index * 8));
      if (auto pointer = dyn_cast<LLVM::LLVMPointerType>(type)) {
        descriptor.set("kind", builder.getStringAttr("pointer"));
        descriptor.set("address_space",
                       builder.getI32IntegerAttr(pointer.getAddressSpace()));
        descriptor.set("size", builder.getI64IntegerAttr(8));
      } else {
        descriptor.set("kind", builder.getStringAttr("value"));
        descriptor.set("size", builder.getI64IntegerAttr(8));
      }
      descriptors.push_back(builder.getDictionaryAttr(descriptor));
    }
    function->setAttr("xw.kernel", builder.getUnitAttr());
    function->setAttr(xw::XWDialect::getSimdWidthAttrName(),
                      builder.getI32IntegerAttr(16));
    function->setAttr("xw.kernel_args", builder.getArrayAttr(descriptors));
    NamedAttrList imported;
    for (NamedAttribute attr : llvmFunction->getDiscardableAttrs())
      imported.set(attr.getName(), attr.getValue());
    if (!imported.empty())
      function->setAttr("xw.imported_llvm_metadata",
                        builder.getDictionaryAttr(imported));
    function.getBody().takeBody(llvmFunction.getBody());

    SmallVector<Operation *> terminators;
    function.walk([&](Operation *op) {
      if (isa<LLVM::BrOp, LLVM::CondBrOp, LLVM::ReturnOp>(op))
        terminators.push_back(op);
    });
    for (Operation *op : terminators) {
      OpBuilder nested(op);
      if (auto branch = dyn_cast<LLVM::BrOp>(op)) {
        cf::BranchOp::create(nested, branch.getLoc(), branch.getDestOperands(),
                             branch.getDest());
      } else if (auto branch = dyn_cast<LLVM::CondBrOp>(op)) {
        cf::CondBranchOp converted = cf::CondBranchOp::create(
            nested, branch.getLoc(), branch.getCondition(),
            branch.getTrueDest(), branch.getTrueDestOperands(),
            branch.getFalseDest(), branch.getFalseDestOperands());
        if (DenseI32ArrayAttr weights = branch.getBranchWeightsAttr())
          converted.setBranchWeightsAttr(weights);
      } else {
        auto result = cast<LLVM::ReturnOp>(op);
        func::ReturnOp::create(nested, result.getLoc(), result.getOperands());
      }
      op->erase();
    }
    llvmFunction.erase();
    return success();
  }

  void runOnOperation() override {
    SmallVector<LLVM::LLVMFuncOp> definitions;
    getOperation().walk([&](LLVM::LLVMFuncOp function) {
      if (!function.isExternal())
        definitions.push_back(function);
    });
    for (LLVM::LLVMFuncOp function : definitions)
      if (failed(importFunction(function)))
        return signalPassFailure();
  }
};

struct VerifyStructured final
    : inter::impl::VerifyStructuredBase<VerifyStructured> {
  void runOnOperation() override {
    Operation *illegal = nullptr;
    getOperation().walk([&](Operation *op) {
      if (isa<cf::BranchOp, cf::CondBranchOp, LLVM::BrOp, LLVM::CondBrOp>(op)) {
        illegal = op;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (!illegal)
      return;
    illegal->emitOpError(
        "unstructured control flow remains after lift-cf-to-scf");
    signalPassFailure();
  }
};

} // namespace.
