#include "inter/Dialect/Inter/IR/XW.h"
#include "inter/Dialect/XeMachine/IR/XeMachineABI.h"
#include "inter/Dialect/XeMachine/IR/XeMachineTarget.h"

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
using namespace inter::xemachine;

namespace {

LogicalResult validateModule(ModuleOp moduleOp) {
  const KernelABI &abi = KernelABI::get();
  if (StringAttr triple = moduleOp->getAttrOfType<StringAttr>(
          LLVM::LLVMDialect::getTargetTripleAttrName())) {
    if (!triple.getValue().empty() &&
        triple.getValue() != "spir64-unknown-unknown")
      return moduleOp.emitError("unsupported LLVM target triple '")
             << triple.getValue() << "'";
  }

  if (ArrayAttr assembly =
          moduleOp->getAttrOfType<ArrayAttr>("llvm.module_asm"))
    if (!assembly.empty())
      return moduleOp.emitError("LLVM module assembly is unsupported");

  DataLayout layout(moduleOp);
  for (uint32_t value = 0; value <= 4; ++value) {
    KernelAddressSpace addressSpace = *abi.decodeAddressSpace(value);
    Type pointer = LLVM::LLVMPointerType::get(moduleOp.getContext(), value);
    llvm::TypeSize size = layout.getTypeSize(pointer);
    uint64_t alignment = layout.getTypeABIAlignment(pointer);
    uint32_t expectedBits = abi.getSourcePointerBitWidth(addressSpace);
    if (size.isScalable() || size.getFixedValue() * 8 != expectedBits ||
        alignment != abi.getPointerArgumentAlignment())
      return moduleOp.emitError()
             << "LLVM pointer layout for address space " << value << " must be "
             << expectedBits << " bits with "
             << abi.getPointerArgumentAlignment() << "-byte ABI alignment";
    std::optional<uint64_t> indexWidth = layout.getTypeIndexBitwidth(pointer);
    uint32_t expectedIndex = abi.getSourcePointerIndexBitWidth(addressSpace);
    if (!indexWidth || *indexWidth != expectedIndex)
      return moduleOp.emitError()
             << "LLVM pointer index width for address space " << value
             << " must be " << expectedIndex << " bits";
  }
  if (auto endianness = dyn_cast_or_null<StringAttr>(layout.getEndianness()))
    if (endianness.getValue() != "little")
      return moduleOp.emitError("LLVM data layout must be little-endian");
  return success();
}

struct ImportLLVM final : inter::impl::ImportLLVMBase<ImportLLVM> {
  using ImportLLVMBase::ImportLLVMBase;

  LogicalResult importFunction(LLVM::LLVMFuncOp llvmFunction) {
    const KernelABI &abi = KernelABI::get();
    if (static_cast<unsigned>(llvmFunction.getCConv()) !=
        abi.getCallingConvention())
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
    uint64_t offset = abi.getFirstExplicitArgumentOffset();
    SmallVector<Attribute> descriptors;
    for (auto [index, type] :
         llvm::enumerate(llvmFunction.getArgumentTypes())) {
      if (!isa<IntegerType, FloatType, LLVM::LLVMPointerType>(type))
        return llvmFunction.emitOpError()
                   << "kernel argument " << index
                   << " must be an integer, floating-point, or pointer type",
               failure();
      uint64_t alignment = 0;
      uint64_t size = 0;
      if (auto pointer = dyn_cast<LLVM::LLVMPointerType>(type)) {
        if (!abi.decodeAddressSpace(pointer.getAddressSpace()))
          return llvmFunction.emitOpError()
                     << "kernel argument " << index << " has address space "
                     << pointer.getAddressSpace() << " outside the kernel ABI",
                 failure();
        alignment = abi.getPointerArgumentAlignment();
        size = abi.getPointerArgumentSize();
      } else {
        bool isFloat = isa<FloatType>(type);
        uint32_t bitWidth = type.getIntOrFloatBitWidth();
        std::optional<uint32_t> scalarAlignment =
            abi.getScalarArgumentAlignment(bitWidth, isFloat);
        if (!scalarAlignment)
          return llvmFunction.emitOpError()
                     << "kernel argument " << index
                     << " has unsupported scalar width " << bitWidth,
                 failure();
        alignment = *scalarAlignment;
        size = (bitWidth + 7) / 8;
      }
      offset = llvm::alignTo(offset, alignment);
      if (abi.crossesPayloadBoundary(offset, size))
        offset = abi.getNextPayloadBoundary(offset);
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
                      builder.getI32IntegerAttr(effectiveSimdWidth));
    function->setAttr("xw.kernel_args", builder.getArrayAttr(descriptors));
    for (NamedAttribute attr : llvmFunction->getDiscardableAttrs())
      if (!attr.getName().strref().starts_with("llvm."))
        function->setAttr(attr.getName(), attr.getValue());
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
    if (failed(validateModule(getOperation())))
      return signalPassFailure();
    effectiveSimdWidth = simdWidth;
    if (IntegerAttr requested = getOperation()->getAttrOfType<IntegerAttr>(
            kCompilationSimdWidthAttrName))
      effectiveSimdWidth = requested.getInt();
    if (effectiveSimdWidth != 8 && effectiveSimdWidth != 16 &&
        effectiveSimdWidth != 32) {
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

private:
  unsigned effectiveSimdWidth = 16;
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
