#include "ABIInfo.h"
#include "LowerModule.h"
#include "LowerTypes.h"
#include "TargetInfo.h"
#include "clang/CIR/MissingFeatures.h"
#include <memory>

namespace mlir {
namespace cir {

class X86_64ABIInfo : public ABIInfo {

public:
  X86_64ABIInfo(LowerTypes &CGT, X86AVXABILevel AVXLevel) : ABIInfo(CGT) {}
};

class X86_64TargetLoweringInfo : public TargetLoweringInfo {
public:
  X86_64TargetLoweringInfo(LowerTypes &LM, X86AVXABILevel AVXLevel)
      : TargetLoweringInfo(std::make_unique<X86_64ABIInfo>(LM, AVXLevel)) {
    assert(::cir::MissingFeatures::swift());
  }
};

std::unique_ptr<TargetLoweringInfo>
createX86_64TargetLoweringInfo(LowerModule &LM, X86AVXABILevel AVXLevel) {
  return std::make_unique<X86_64TargetLoweringInfo>(LM.getTypes(), AVXLevel);
}

} // namespace cir
} // namespace mlir
