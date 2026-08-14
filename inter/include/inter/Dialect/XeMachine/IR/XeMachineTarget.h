#ifndef INTER_DIALECT_XEMACHINE_IR_XEMACHINETARGET_H
#define INTER_DIALECT_XEMACHINE_IR_XEMACHINETARGET_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>

namespace mlir {
class MLIRContext;
}

namespace inter::xemachine {

class TargetAttr;

enum class TargetChip { bmg };
enum class TargetArchitecture { xe2 };

struct ZebinCompatibilityIdentity {
  uint32_t productFamily;
  uint32_t graphicsCore;
  uint32_t targetMetadata;
  uint32_t productConfig;
  llvm::StringLiteral version;
};

class TargetConfig {
public:
  static llvm::Expected<TargetConfig>
  resolve(llvm::StringRef chip, llvm::ArrayRef<llvm::StringRef> features = {});

  TargetChip getChip() const { return chip; }
  llvm::StringRef getChipName() const;
  TargetArchitecture getArchitecture() const;
  llvm::StringRef getArchitectureName() const;
  uint32_t getGrfByteSize() const;
  uint32_t getGrfCount() const;
  uint32_t getReservedGrfCount() const;
  llvm::ArrayRef<uint32_t> getSupportedSimdWidths() const;
  bool supportsSimdWidth(uint32_t width) const;
  ZebinCompatibilityIdentity getZebinCompatibilityIdentity() const;
  TargetAttr getAttr(mlir::MLIRContext *context) const;

private:
  explicit TargetConfig(TargetChip chip) : chip(chip) {}

  TargetChip chip;
};

} // namespace inter::xemachine

#endif // INTER_DIALECT_XEMACHINE_IR_XEMACHINETARGET_H
