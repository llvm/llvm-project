#include "inter/Dialect/XeMachine/IR/XeMachine.h"

#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/MathExtras.h"

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
TargetAttr::verify(llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
                   mlir::StringAttr chip) {
  if (!chip)
    return emitError() << "target chip cannot be null";
  llvm::Expected<TargetConfig> target = TargetConfig::resolve(chip.getValue());
  if (!target)
    return emitError() << llvm::toString(target.takeError());
  return mlir::success();
}

mlir::LogicalResult
inter::xemachine::verifyKernelArgLayout(mlir::ArrayAttr arguments,
                                        mlir::Operation *owner) {
  const KernelABI &abi = KernelABI::get();
  if (!arguments)
    return owner->emitOpError("missing or invalid kernel argument layout");

  llvm::SmallVector<std::pair<uint64_t, uint64_t>> ranges;
  for (mlir::Attribute argument : arguments) {
    auto descriptor = mlir::dyn_cast<KernelArgAttr>(argument);
    if (!descriptor)
      return owner->emitOpError("invalid kernel argument descriptor");

    uint64_t offset = descriptor.getOffset();
    uint64_t size = descriptor.getSize();
    uint64_t alignment = descriptor.getAlignment();
    llvm::StringRef addressSpace = descriptor.getAddressSpace().getValue();
    llvm::StringRef access = descriptor.getAccess().getValue();
    if (size == 0 || alignment == 0 || !llvm::isPowerOf2_64(alignment))
      return owner->emitOpError("invalid kernel argument size or alignment");
    if (descriptor.getKind() == KernelArgKind::by_pointer) {
      if (size != abi.getPointerArgumentSize() || addressSpace == "none")
        return owner->emitOpError("invalid pointer kernel argument descriptor");
      if (access != "read_only" && access != "write_only" &&
          access != "read_write")
        return owner->emitOpError("invalid pointer kernel argument access");
    } else if (addressSpace != "none" || access != "none") {
      return owner->emitOpError("by-value argument has pointer ABI properties");
    }
    if (offset < abi.getFirstExplicitArgumentOffset())
      return owner->emitOpError(
          "kernel argument overlaps the implicit payload");
    if (offset % alignment != 0)
      return owner->emitOpError("kernel argument payload is misaligned");
    if (offset > abi.getCrossThreadPayloadLimit() ||
        size > abi.getCrossThreadPayloadLimit() - offset)
      return owner->emitOpError(
          "kernel argument is outside the loaded payload");
    if (abi.crossesPayloadBoundary(offset, size))
      return owner->emitOpError("kernel argument crosses a payload boundary");
    for (auto [begin, end] : ranges)
      if (offset < end && begin < offset + size)
        return owner->emitOpError("kernel argument payloads overlap");
    ranges.emplace_back(offset, offset + size);
  }
  return mlir::success();
}
