//===- SPIRV.cpp - Emit CIR for SPIR/SPIR-V -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../TargetLoweringInfo.h"
#include "clang/CIR/Dialect/IR/CIROpsEnums.h"

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
};

class SPIRVTargetLoweringInfo : public TargetLoweringInfo {
public:
  unsigned getTargetAddrSpaceFromCIRAddrSpace(
      cir::LangAddressSpace addrSpace) const override {
    auto idx = static_cast<unsigned>(addrSpace);
    assert(idx < std::size(SPIRVAddrSpaceMap) &&
           "Unknown CIR address space for SPIR-V target");
    return SPIRVAddrSpaceMap[idx];
  }
};

} // namespace

std::unique_ptr<TargetLoweringInfo> createSPIRVTargetLoweringInfo() {
  return std::make_unique<SPIRVTargetLoweringInfo>();
}

} // namespace cir
