#include "inter/Dialect/XeMachine/IR/XeMachine.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/SmallVector.h"

using namespace inter::xemachine;

#include "inter/Dialect/XeMachine/IR/XeMachineDialect.cpp.inc"
#include "inter/Dialect/XeMachine/IR/XeMachineEnums.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "inter/Dialect/XeMachine/IR/XeMachineTypes.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "inter/Dialect/XeMachine/IR/XeMachineAttrs.cpp.inc"

void XeMachineDialect::initialize() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "inter/Dialect/XeMachine/IR/XeMachineTypes.cpp.inc"
      >();
  addAttributes<
#define GET_ATTRDEF_LIST
#include "inter/Dialect/XeMachine/IR/XeMachineAttrs.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "inter/Dialect/XeMachine/IR/XeMachineOps.cpp.inc"
      ,
#define GET_OP_LIST
#include "inter/Dialect/XeMachine/IR/XeMachineTransformOps.cpp.inc"
      >();
}

mlir::LogicalResult
inter::xemachine::verifyKernelArgLayout(mlir::FunctionType functionType,
                                        mlir::ArrayAttr arguments,
                                        mlir::Operation *owner) {
  constexpr uint64_t firstExplicitArgument = 24;
  constexpr uint64_t loadedPayloadBytes = 64;
  if (!arguments || arguments.size() != functionType.getNumInputs())
    return owner->emitOpError("missing or invalid kernel argument layout");

  llvm::SmallVector<std::pair<uint64_t, uint64_t>> ranges;
  for (auto [index, type] : llvm::enumerate(functionType.getInputs())) {
    auto descriptor = mlir::dyn_cast<KernelArgAttr>(arguments[index]);
    if (!descriptor)
      return owner->emitOpError("invalid kernel argument descriptor");

    KernelArgKind expectedKind;
    uint64_t expectedSize;
    if (auto pointer = mlir::dyn_cast<mlir::LLVM::LLVMPointerType>(type)) {
      if (pointer.getAddressSpace() != 1)
        return owner->emitOpError(
            "only global pointer kernel arguments are supported");
      expectedKind = KernelArgKind::by_pointer;
      expectedSize = 8;
    } else if (auto integer = mlir::dyn_cast<mlir::IntegerType>(type)) {
      if (integer.getWidth() == 0 || integer.getWidth() > 64)
        return owner->emitOpError("unsupported kernel integer argument");
      expectedKind = KernelArgKind::by_value;
      expectedSize = llvm::divideCeil(integer.getWidth(), 8u);
    } else {
      return owner->emitOpError("unsupported kernel argument type");
    }
    if (descriptor.getKind() != expectedKind ||
        descriptor.getSize() != expectedSize)
      return owner->emitOpError(
          "kernel argument descriptor does not match type");

    uint64_t offset = descriptor.getOffset();
    uint64_t size = descriptor.getSize();
    if (offset < firstExplicitArgument)
      return owner->emitOpError(
          "kernel argument overlaps the implicit payload");
    if (offset % size != 0)
      return owner->emitOpError("kernel argument payload is misaligned");
    if (offset > loadedPayloadBytes || size > loadedPayloadBytes - offset)
      return owner->emitOpError(
          "kernel argument is outside the loaded payload");
    for (auto [begin, end] : ranges)
      if (offset < end && begin < offset + size)
        return owner->emitOpError("kernel argument payloads overlap");
    ranges.emplace_back(offset, offset + size);
  }
  return mlir::success();
}
