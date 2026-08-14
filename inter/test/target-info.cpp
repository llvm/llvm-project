#include "inter/Dialect/XeMachine/IR/XeMachineTarget.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

int main(int argc, char **argv) {
  if (argc < 2) {
    llvm::errs() << "usage: inter-target-info <chip> [feature ...]\n";
    return 1;
  }

  llvm::SmallVector<llvm::StringRef> features;
  for (int index = 2; index < argc; ++index)
    features.push_back(argv[index]);
  llvm::Expected<inter::xemachine::TargetConfig> target =
      inter::xemachine::TargetConfig::resolve(argv[1], features);
  if (!target) {
    llvm::errs() << llvm::toString(target.takeError()) << '\n';
    return 1;
  }

  llvm::outs() << "chip: " << target->getChipName() << '\n';
  llvm::outs() << "architecture: " << target->getArchitectureName() << '\n';
  llvm::outs() << "grf-byte-size: " << target->getGrfByteSize() << '\n';
  llvm::outs() << "grf-count: " << target->getGrfCount() << '\n';
  llvm::outs() << "reserved-grf-count: " << target->getReservedGrfCount()
               << '\n';
  llvm::outs() << "simd-widths:";
  for (uint32_t width : target->getSupportedSimdWidths())
    llvm::outs() << ' ' << width;
  llvm::outs() << '\n';
  inter::xemachine::ZebinCompatibilityIdentity zebin =
      target->getZebinCompatibilityIdentity();
  llvm::outs() << "zebin-product-family: " << zebin.productFamily << '\n';
  llvm::outs() << "zebin-graphics-core: " << zebin.graphicsCore << '\n';
  llvm::outs() << "zebin-target-metadata: " << zebin.targetMetadata << '\n';
  llvm::outs() << "zebin-product-config: " << zebin.productConfig << '\n';
  llvm::outs() << "zebin-version: " << zebin.version << '\n';
  return 0;
}
