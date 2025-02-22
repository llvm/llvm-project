//===- NVPTX.cpp - TargetInfo for NVPTX -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ABIInfoImpl.h"
#include "LowerFunctionInfo.h"
#include "LowerTypes.h"
#include "TargetInfo.h"
#include "TargetLoweringInfo.h"
#include "clang/CIR/ABIArgInfo.h"
#include "clang/CIR/MissingFeatures.h"
#include "llvm/Support/ErrorHandling.h"

using ABIArgInfo = cir::ABIArgInfo;
using MissingFeature = cir::MissingFeatures;

namespace cir {

//===----------------------------------------------------------------------===//
// NVPTX ABI Implementation
//===----------------------------------------------------------------------===//

namespace {

class NVPTXABIInfo : public ABIInfo {
public:
  NVPTXABIInfo(LowerTypes &lt) : ABIInfo(lt) {}

private:
  void computeInfo(LowerFunctionInfo &fi) const override {
    llvm_unreachable("NYI");
  }
};

class NVPTXTargetLoweringInfo : public TargetLoweringInfo {
public:
  NVPTXTargetLoweringInfo(LowerTypes &lt)
      : TargetLoweringInfo(std::make_unique<NVPTXABIInfo>(lt)) {}

  unsigned getTargetAddrSpaceFromCIRAddrSpace(
      cir::AddressSpaceAttr addressSpaceAttr) const override {
    using Kind = cir::AddressSpaceAttr::Kind;
    switch (addressSpaceAttr.getValue()) {
    case Kind::offload_private:
      return 0;
    case Kind::offload_local:
      return 3;
    case Kind::offload_global:
      return 1;
    case Kind::offload_constant:
      return 2;
    case Kind::offload_generic:
      return 4;
    default:
      cir_cconv_unreachable("Unknown CIR address space for this target");
    }
  }
};

} // namespace

std::unique_ptr<TargetLoweringInfo>
createNVPTXTargetLoweringInfo(LowerModule &lowerModule) {
  return std::make_unique<NVPTXTargetLoweringInfo>(lowerModule.getTypes());
}

} // namespace cir
