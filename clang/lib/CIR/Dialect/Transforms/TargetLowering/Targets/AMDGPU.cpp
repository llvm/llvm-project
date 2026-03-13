//===- AMDGPU.cpp - Emit CIR for AMDGPU -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../TargetLoweringInfo.h"
#include "clang/CIR/Dialect/IR/CIROpsEnums.h"
#include "llvm/Support/ErrorHandling.h"

namespace cir {

namespace {

class AMDGPUTargetLoweringInfo : public TargetLoweringInfo {
public:
  // Address space mapping from:
  // https://llvm.org/docs/AMDGPUUsage.html#address-spaces
  unsigned getTargetAddrSpaceFromCIRAddrSpace(
      cir::LangAddressSpace addrSpace) const override {
    switch (addrSpace) {
    case cir::LangAddressSpace::Default:
      return 0;
    case cir::LangAddressSpace::OffloadPrivate:
      return 5;
    case cir::LangAddressSpace::OffloadLocal:
      return 3;
    case cir::LangAddressSpace::OffloadGlobal:
      return 1;
    case cir::LangAddressSpace::OffloadConstant:
      return 4;
    case cir::LangAddressSpace::OffloadGeneric:
      return 0;
    }
    llvm_unreachable("Unknown CIR address space for AMDGPU target");
  }
};

} // namespace

std::unique_ptr<TargetLoweringInfo> createAMDGPUTargetLoweringInfo() {
  return std::make_unique<AMDGPUTargetLoweringInfo>();
}

} // namespace cir
