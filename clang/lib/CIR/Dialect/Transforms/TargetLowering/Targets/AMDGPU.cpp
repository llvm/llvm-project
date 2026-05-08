//===- AMDGPU.cpp - Emit CIR for AMDGPU -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../TargetLoweringInfo.h"
#include "clang/CIR/Dialect/IR/CIROpsEnums.h"
#include "llvm/Support/AMDGPUAddrSpace.h"

namespace cir {

namespace {

// Address space mapping from:
// https://llvm.org/docs/AMDGPUUsage.html#address-spaces
//
// Indexed by cir::LangAddressSpace enum values.
constexpr unsigned AMDGPUAddrSpaceMap[] = {
    llvm::AMDGPUAS::FLAT_ADDRESS,     // Default
    llvm::AMDGPUAS::PRIVATE_ADDRESS,  // OffloadPrivate
    llvm::AMDGPUAS::LOCAL_ADDRESS,    // OffloadLocal
    llvm::AMDGPUAS::GLOBAL_ADDRESS,   // OffloadGlobal
    llvm::AMDGPUAS::CONSTANT_ADDRESS, // OffloadConstant
    llvm::AMDGPUAS::FLAT_ADDRESS,     // OffloadGeneric
};

class AMDGPUTargetLoweringInfo : public TargetLoweringInfo {
public:
  unsigned getTargetAddrSpaceFromCIRAddrSpace(
      cir::LangAddressSpace addrSpace) const override {
    auto idx = static_cast<unsigned>(addrSpace);
    assert(idx < std::size(AMDGPUAddrSpaceMap) &&
           "Unknown CIR address space for AMDGPU target");
    return AMDGPUAddrSpaceMap[idx];
  }
};

} // namespace

std::unique_ptr<TargetLoweringInfo> createAMDGPUTargetLoweringInfo() {
  return std::make_unique<AMDGPUTargetLoweringInfo>();
}

} // namespace cir
