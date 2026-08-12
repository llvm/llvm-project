#include "inter/Dialect/Inter/IR/XW.h"

#include "inter/Transforms/Passes.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
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
  using ImportLLVMBase::ImportLLVMBase;

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
    DataLayout layout = DataLayout::closest(llvmFunction);
    uint64_t offset = 24;
    SmallVector<Attribute> descriptors;
    for (auto [index, type] :
         llvm::enumerate(llvmFunction.getArgumentTypes())) {
      if (!isa<IntegerType, FloatType, LLVM::LLVMPointerType>(type))
        return llvmFunction.emitOpError()
                   << "kernel argument " << index
                   << " must be an integer, floating-point, or pointer type",
               failure();
      llvm::TypeSize typeSize = layout.getTypeSize(type);
      if (typeSize.isScalable())
        return llvmFunction.emitOpError()
                   << "kernel argument " << index << " has scalable size",
               failure();
      uint64_t alignment = layout.getTypeABIAlignment(type);
      uint64_t size = typeSize.getFixedValue();
      offset = llvm::alignTo(offset, alignment);
      NamedAttrList descriptor;
      descriptor.set("offset", builder.getI64IntegerAttr(offset));
      descriptor.set("alignment", builder.getI64IntegerAttr(alignment));
      descriptor.set("size", builder.getI64IntegerAttr(size));
      if (auto pointer = dyn_cast<LLVM::LLVMPointerType>(type)) {
        descriptor.set("kind", builder.getStringAttr("pointer"));
        descriptor.set("address_space",
                       builder.getI32IntegerAttr(pointer.getAddressSpace()));
        DictionaryAttr argAttrs = llvmFunction.getArgAttrDict(index);
        StringRef access =
            argAttrs &&
                    argAttrs.contains(LLVM::LLVMDialect::getReadonlyAttrName())
                ? "read_only"
            : argAttrs &&
                    argAttrs.contains(LLVM::LLVMDialect::getWriteOnlyAttrName())
                ? "write_only"
                : "read_write";
        descriptor.set("access", builder.getStringAttr(access));
      } else {
        descriptor.set("kind", builder.getStringAttr("value"));
      }
      descriptors.push_back(builder.getDictionaryAttr(descriptor));
      offset += size;
    }
    function->setAttr("xw.kernel", builder.getUnitAttr());
    function->setAttr(xw::XWDialect::getSimdWidthAttrName(),
                      builder.getI32IntegerAttr(simdWidth));
    function->setAttr("xw.kernel_args", builder.getArrayAttr(descriptors));
    NamedAttrList imported;
    for (NamedAttribute attr : llvmFunction->getDiscardableAttrs())
      if (!attr.getName().strref().starts_with("llvm."))
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
    if (simdWidth != 8 && simdWidth != 16 && simdWidth != 32) {
      getOperation().emitError("--simd-width must be 8, 16, or 32");
      return signalPassFailure();
    }
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
