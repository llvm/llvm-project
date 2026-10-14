//===- SPIRV.cpp - Emit CIR for SPIR/SPIR-V -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../TargetLoweringInfo.h"
#include "clang/CIR/Dialect/IR/CIROpsEnums.h"
#include "clang/CIR/MissingFeatures.h"

namespace cir {

namespace {

// SPIR-V OpenCL storage classes, indexed by cir::LangAddressSpace.
constexpr unsigned SPIRVAddrSpaceMap[] = {
    0, // Function
    0, // Function
    3, // Workgroup
    1, // CrossWorkgroup
    2, // UniformConstant
    4, // Generic
    5, // GlobalDevice
    6, // GlobalHost
};

class SPIRVTargetLoweringInfo : public TargetLoweringInfo {
  // SPIR keeps the default system scope, like CommonSPIRTargetCodeGenInfo.
  bool isSPIRV;

public:
  explicit SPIRVTargetLoweringInfo(bool isSPIRV) : isSPIRV(isSPIRV) {}

  unsigned getTargetAddrSpaceFromCIRAddrSpace(
      cir::LangAddressSpace addrSpace) const override {
    // TODO(cir): SYCL and CUDA/HIP device code map Default to Generic
    // (SPIRDefIsGenMap).
    assert(!cir::MissingFeatures::spirvDefaultIsGenericAddrSpace());
    auto idx = static_cast<unsigned>(addrSpace);
    assert(idx < std::size(SPIRVAddrSpaceMap) &&
           "Unknown CIR address space for SPIR-V target");
    return SPIRVAddrSpaceMap[idx];
  }

  cir::SyncScopeKind
  convertSyncScope(cir::SyncScopeKind syncScope) const override {
    return isSPIRV ? syncScope
                   : TargetLoweringInfo::convertSyncScope(syncScope);
  }
};

} // namespace

std::unique_ptr<TargetLoweringInfo>
createSPIRVTargetLoweringInfo(bool isSPIRV) {
  return std::make_unique<SPIRVTargetLoweringInfo>(isSPIRV);
}

} // namespace cir
