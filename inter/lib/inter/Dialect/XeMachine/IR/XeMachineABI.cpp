#include "inter/Dialect/XeMachine/IR/XeMachineABI.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MathExtras.h"

#include <array>
#include <cassert>

using namespace inter::xemachine;

const KernelABI &KernelABI::get() {
  static const KernelABI abi;
  return abi;
}

llvm::CallingConv::ID KernelABI::getCallingConvention() const {
  return llvm::CallingConv::SPIR_KERNEL;
}

std::optional<KernelAddressSpace>
KernelABI::decodeAddressSpace(uint32_t addressSpace) const {
  if (addressSpace > static_cast<uint32_t>(KernelAddressSpace::generic))
    return std::nullopt;
  return static_cast<KernelAddressSpace>(addressSpace);
}

llvm::StringRef
KernelABI::getAddressSpaceName(KernelAddressSpace addressSpace) const {
  switch (addressSpace) {
  case KernelAddressSpace::privateSpace:
    return "private";
  case KernelAddressSpace::global:
    return "global";
  case KernelAddressSpace::constant:
    return "constant";
  case KernelAddressSpace::local:
    return "local";
  case KernelAddressSpace::generic:
    return "generic";
  }
  llvm_unreachable("unknown kernel address space");
}

uint32_t
KernelABI::getSourcePointerBitWidth(KernelAddressSpace addressSpace) const {
  return 64;
}

uint32_t KernelABI::getSourcePointerIndexBitWidth(
    KernelAddressSpace addressSpace) const {
  return 64;
}

uint32_t
KernelABI::getMachinePointerBitWidth(KernelAddressSpace addressSpace) const {
  return addressSpace == KernelAddressSpace::local ? 32 : 64;
}

uint32_t KernelABI::getMachinePointerIndexBitWidth(
    KernelAddressSpace addressSpace) const {
  return getMachinePointerBitWidth(addressSpace);
}

std::optional<uint32_t>
KernelABI::getScalarArgumentAlignment(uint32_t bitWidth, bool isFloat) const {
  if (isFloat) {
    if (bitWidth == 32)
      return 4;
    return std::nullopt;
  }
  if (bitWidth == 1 || bitWidth == 8)
    return 1;
  if (bitWidth == 16)
    return 2;
  if (bitWidth == 32 || bitWidth == 64)
    return 4;
  return std::nullopt;
}

uint32_t KernelABI::getPointerArgumentSize() const { return 8; }
uint32_t KernelABI::getPointerArgumentAlignment() const { return 8; }
uint32_t KernelABI::getFirstExplicitArgumentOffset() const { return 24; }
uint32_t KernelABI::getCrossThreadPayloadLimit() const { return 192; }
uint32_t KernelABI::getInlinePayloadSize() const { return 32; }
uint32_t KernelABI::getPayloadChunkSize() const { return 64; }

bool KernelABI::crossesPayloadBoundary(uint64_t offset, uint64_t size) const {
  if (offset < getInlinePayloadSize())
    return offset + size > getInlinePayloadSize();
  uint64_t tailOffset = offset - getInlinePayloadSize();
  return tailOffset % getPayloadChunkSize() + size > getPayloadChunkSize();
}

uint64_t KernelABI::getNextPayloadBoundary(uint64_t offset) const {
  if (offset < getInlinePayloadSize())
    return getInlinePayloadSize();
  uint64_t tailOffset = offset - getInlinePayloadSize();
  return getInlinePayloadSize() +
         llvm::alignTo(tailOffset, uint64_t{getPayloadChunkSize()});
}

llvm::ArrayRef<ImplicitKernelArgumentLayout>
KernelABI::getImplicitArguments() const {
  static constexpr std::array<ImplicitKernelArgumentLayout, 2> arguments = {{
      {ImplicitKernelArgument::globalIdOffset, "global_id_offset", 0, 12},
      {ImplicitKernelArgument::enqueuedLocalSize, "enqueued_local_size", 12,
       12},
  }};
  return arguments;
}

uint32_t KernelABI::getImplicitArgumentDword(ImplicitKernelArgument argument,
                                             uint32_t axis) const {
  assert(axis < 3 && "implicit kernel argument axis must be 0, 1, or 2");
  for (const ImplicitKernelArgumentLayout &layout : getImplicitArguments())
    if (layout.argument == argument)
      return layout.offset / 4 + axis;
  llvm_unreachable("unknown implicit kernel argument");
}

uint32_t KernelABI::getGroupIdSubregister(uint32_t axis) const {
  static constexpr std::array<uint32_t, 3> subregisters = {1, 6, 7};
  assert(axis < subregisters.size() && "group ID axis must be 0, 1, or 2");
  return subregisters[axis];
}

uint32_t KernelABI::getLocalIdBlobOffset() const { return 32; }
uint32_t KernelABI::getLocalIdAxisStride() const { return 64; }

uint32_t KernelABI::getPerThreadPayloadSize(uint32_t highestUsedAxis) const {
  assert(highestUsedAxis < 3 && "local ID axis must be 0, 1, or 2");
  return (highestUsedAxis + 1) * getLocalIdAxisStride();
}

uint32_t KernelABI::getInlineDataRegister(uint32_t perThreadPayloadSize) const {
  assert(perThreadPayloadSize % getLocalIdAxisStride() == 0 &&
         "per-thread payload must contain whole local ID axes");
  return 1 + perThreadPayloadSize / getLocalIdAxisStride();
}

uint32_t KernelABI::getReservedPayloadGrfCount() const { return 5; }
uint32_t KernelABI::getScratchSlotAlignment() const { return 64; }
int64_t KernelABI::getScratchAddressBias() const { return 0x10000; }
uint32_t KernelABI::getScratchSurfaceMask() const { return 0xFFFFFC00; }
uint32_t KernelABI::getScratchSurfaceShift() const { return 4; }
uint32_t KernelABI::getScratchSurfaceSourceSubregister() const { return 5; }
uint32_t KernelABI::getScratchExdescBias() const { return 0x80000000u; }
