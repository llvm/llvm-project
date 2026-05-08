//===- NVPTX.cpp ----------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "../TargetLoweringInfo.h"
#include "clang/CIR/Dialect/IR/CIROpsEnums.h"
#include "llvm/Support/NVPTXAddrSpace.h"

namespace cir {

namespace {

constexpr unsigned NVPTXAddrSpaceMap[] = {
    llvm::NVPTXAS::ADDRESS_SPACE_GENERIC, llvm::NVPTXAS::ADDRESS_SPACE_GENERIC,
    llvm::NVPTXAS::ADDRESS_SPACE_SHARED,  llvm::NVPTXAS::ADDRESS_SPACE_GLOBAL,
    llvm::NVPTXAS::ADDRESS_SPACE_CONST,   llvm::NVPTXAS::ADDRESS_SPACE_GENERIC,
};

class NVPTXTargetLoweringInfo : public TargetLoweringInfo {
public:
  unsigned getTargetAddrSpaceFromCIRAddrSpace(
      cir::LangAddressSpace addrSpace) const override {

    auto idx = static_cast<unsigned>(addrSpace);
    assert(idx < std::size(NVPTXAddrSpaceMap) &&
           "Unknown CIR address space for NVPTX target");
    return NVPTXAddrSpaceMap[idx];
  }
};

} // namespace

std::unique_ptr<TargetLoweringInfo> createNVPTXTargetLoweringInfo() {
  return std::make_unique<NVPTXTargetLoweringInfo>();
}
} // namespace cir
