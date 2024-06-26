//===- AArch64.cpp --------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/CIR/Target/AArch64.h"
#include "ABIInfoImpl.h"
#include "LowerFunctionInfo.h"
#include "LowerTypes.h"
#include "TargetInfo.h"
#include "TargetLoweringInfo.h"
#include "clang/CIR/ABIArgInfo.h"
#include "clang/CIR/Dialect/IR/CIRTypes.h"
#include "clang/CIR/MissingFeatures.h"
#include "llvm/Support/ErrorHandling.h"

using AArch64ABIKind = ::cir::AArch64ABIKind;
using ABIArgInfo = ::cir::ABIArgInfo;
using MissingFeature = ::cir::MissingFeatures;

namespace mlir {
namespace cir {

//===----------------------------------------------------------------------===//
// AArch64 ABI Implementation
//===----------------------------------------------------------------------===//

namespace {

class AArch64ABIInfo : public ABIInfo {
  AArch64ABIKind Kind;

public:
  AArch64ABIInfo(LowerTypes &CGT, AArch64ABIKind Kind)
      : ABIInfo(CGT), Kind(Kind) {}

private:
  AArch64ABIKind getABIKind() const { return Kind; }

  ABIArgInfo classifyReturnType(Type RetTy, bool IsVariadic) const;

  void computeInfo(LowerFunctionInfo &FI) const override {
    if (!::mlir::cir::classifyReturnType(getCXXABI(), FI, *this))
      FI.getReturnInfo() =
          classifyReturnType(FI.getReturnType(), FI.isVariadic());

    for (auto &_ : FI.arguments())
      llvm_unreachable("NYI");
  }
};

class AArch64TargetLoweringInfo : public TargetLoweringInfo {
public:
  AArch64TargetLoweringInfo(LowerTypes &LT, AArch64ABIKind Kind)
      : TargetLoweringInfo(std::make_unique<AArch64ABIInfo>(LT, Kind)) {
    assert(!MissingFeature::swift());
  }
};

} // namespace

ABIArgInfo AArch64ABIInfo::classifyReturnType(Type RetTy,
                                              bool IsVariadic) const {
  if (isa<VoidType>(RetTy))
    return ABIArgInfo::getIgnore();

  llvm_unreachable("NYI");
}

std::unique_ptr<TargetLoweringInfo>
createAArch64TargetLoweringInfo(LowerModule &CGM, AArch64ABIKind Kind) {
  return std::make_unique<AArch64TargetLoweringInfo>(CGM.getTypes(), Kind);
}

} // namespace cir
} // namespace mlir
