#include "TargetLoweringInfo.h"

namespace cir {

TargetLoweringInfo::TargetLoweringInfo(std::unique_ptr<ABIInfo> Info)
    : Info(std::move(Info)) {}

TargetLoweringInfo::~TargetLoweringInfo() = default;

} // namespace cir
