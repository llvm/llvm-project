#include "inter/Dialect/XeMachine/IR/XeMachine.h"

#include "mlir/IR/MLIRContext.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Twine.h"

#include <array>

using namespace inter::xemachine;

llvm::Expected<TargetConfig>
TargetConfig::resolve(llvm::StringRef chip,
                      llvm::ArrayRef<llvm::StringRef> features) {
  if (chip != "bmg")
    return llvm::createStringError("unsupported Intel GPU target chip '" +
                                   chip + "'");
  if (!features.empty())
    return llvm::createStringError("unsupported Intel GPU target feature '" +
                                   features.front() + "' for chip '" + chip +
                                   "'");
  return TargetConfig(TargetChip::bmg);
}

llvm::Expected<TargetConfig> TargetConfig::resolve(TargetAttr target) {
  if (!target)
    return llvm::createStringError("missing Intel GPU target attribute");
  return resolve(target.getChip().getValue());
}

llvm::StringRef TargetConfig::getChipName() const { return "bmg"; }

TargetArchitecture TargetConfig::getArchitecture() const {
  return TargetArchitecture::xe2;
}

llvm::StringRef TargetConfig::getArchitectureName() const { return "xe2"; }

uint32_t TargetConfig::getGrfByteSize() const { return 64; }

uint32_t TargetConfig::getGrfCount() const { return 128; }

uint32_t TargetConfig::getSbidCount(uint32_t grfCount) const {
  return grfCount > getGrfCount() ? 32 : 16;
}

llvm::ArrayRef<uint32_t> TargetConfig::getSupportedSimdWidths() const {
  static constexpr std::array<uint32_t, 3> widths = {8, 16, 32};
  return widths;
}

bool TargetConfig::supportsSimdWidth(uint32_t width) const {
  return llvm::is_contained(getSupportedSimdWidths(), width);
}

ZebinCompatibilityIdentity TargetConfig::getZebinCompatibilityIdentity() const {
  return {0x4FA, 0xC09, 0, 0x05004000, "1.64"};
}

TargetAttr TargetConfig::getAttr(mlir::MLIRContext *context) const {
  return TargetAttr::get(context, chip);
}
