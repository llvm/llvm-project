#ifndef INTER_DIALECT_XEMACHINE_IR_XEMACHINEABI_H
#define INTER_DIALECT_XEMACHINE_IR_XEMACHINEABI_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/CallingConv.h"

#include <cstdint>
#include <optional>

namespace inter::xemachine {

enum class KernelAddressSpace : uint32_t {
  privateSpace = 0,
  global = 1,
  constant = 2,
  local = 3,
  generic = 4,
};

enum class ImplicitKernelArgument {
  globalIdOffset,
  enqueuedLocalSize,
};

struct ImplicitKernelArgumentLayout {
  ImplicitKernelArgument argument;
  llvm::StringLiteral name;
  uint32_t offset;
  uint32_t size;
};

class KernelABI {
public:
  static const KernelABI &get();

  llvm::CallingConv::ID getCallingConvention() const;
  std::optional<KernelAddressSpace>
  decodeAddressSpace(uint32_t addressSpace) const;
  llvm::StringRef getAddressSpaceName(KernelAddressSpace addressSpace) const;
  uint32_t getSourcePointerBitWidth(KernelAddressSpace addressSpace) const;
  uint32_t getSourcePointerIndexBitWidth(KernelAddressSpace addressSpace) const;
  uint32_t getMachinePointerBitWidth(KernelAddressSpace addressSpace) const;
  uint32_t
  getMachinePointerIndexBitWidth(KernelAddressSpace addressSpace) const;
  std::optional<uint32_t> getScalarArgumentAlignment(uint32_t bitWidth,
                                                     bool isFloat) const;
  uint32_t getPointerArgumentSize() const;
  uint32_t getPointerArgumentAlignment() const;

  uint32_t getFirstExplicitArgumentOffset() const;
  uint32_t getCrossThreadPayloadLimit() const;
  uint32_t getInlinePayloadSize() const;
  uint32_t getPayloadChunkSize() const;
  bool crossesPayloadBoundary(uint64_t offset, uint64_t size) const;
  uint64_t getNextPayloadBoundary(uint64_t offset) const;
  llvm::ArrayRef<ImplicitKernelArgumentLayout> getImplicitArguments() const;
  uint32_t getImplicitArgumentDword(ImplicitKernelArgument argument,
                                    uint32_t axis) const;
  uint32_t getGroupIdSubregister(uint32_t axis) const;

  uint32_t getLocalIdBlobOffset() const;
  uint32_t getLocalIdAxisStride() const;
  uint32_t getPerThreadPayloadSize(uint32_t highestUsedAxis) const;
  uint32_t getInlineDataRegister(uint32_t perThreadPayloadSize) const;
  uint32_t getReservedPayloadGrfCount() const;

  uint32_t getScratchSlotAlignment() const;
  int64_t getScratchAddressBias() const;
  uint32_t getScratchSurfaceMask() const;
  uint32_t getScratchSurfaceShift() const;
  uint32_t getScratchSurfaceSourceSubregister() const;
  uint32_t getScratchExdescBias() const;

private:
  KernelABI() = default;
};

} // namespace inter::xemachine

#endif // INTER_DIALECT_XEMACHINE_IR_XEMACHINEABI_H
