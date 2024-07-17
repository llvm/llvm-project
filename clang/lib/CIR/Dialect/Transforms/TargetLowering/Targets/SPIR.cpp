//===- SPIR.cpp - TargetInfo for SPIR and SPIR-V --------------------------===//
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

using ABIArgInfo = ::cir::ABIArgInfo;
using MissingFeature = ::cir::MissingFeatures;

namespace mlir {
namespace cir {

//===----------------------------------------------------------------------===//
// SPIR-V ABI Implementation
//===----------------------------------------------------------------------===//

namespace {

class SPIRVABIInfo : public ABIInfo {
public:
  SPIRVABIInfo(LowerTypes &LT) : ABIInfo(LT) {}

private:
  void computeInfo(LowerFunctionInfo &FI) const override {
    llvm_unreachable("ABI NYI");
  }
};

class SPIRVTargetLoweringInfo : public TargetLoweringInfo {
public:
  SPIRVTargetLoweringInfo(LowerTypes &LT)
      : TargetLoweringInfo(std::make_unique<SPIRVABIInfo>(LT)) {}
};

} // namespace

std::unique_ptr<TargetLoweringInfo>
createSPIRVTargetLoweringInfo(LowerModule &lowerModule) {
  return std::make_unique<SPIRVTargetLoweringInfo>(lowerModule.getTypes());
}

} // namespace cir
} // namespace mlir
