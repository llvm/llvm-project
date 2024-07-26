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
  bool isDarwinPCS() const { return Kind == AArch64ABIKind::DarwinPCS; }

  ABIArgInfo classifyReturnType(Type RetTy, bool IsVariadic) const;
  ABIArgInfo classifyArgumentType(Type RetTy, bool IsVariadic,
                                  unsigned CallingConvention) const;

  void computeInfo(LowerFunctionInfo &FI) const override {
    if (!::mlir::cir::classifyReturnType(getCXXABI(), FI, *this))
      FI.getReturnInfo() =
          classifyReturnType(FI.getReturnType(), FI.isVariadic());

    for (auto &it : FI.arguments())
      it.info = classifyArgumentType(it.type, FI.isVariadic(),
                                     FI.getCallingConvention());
  }
};

class AArch64TargetLoweringInfo : public TargetLoweringInfo {
public:
  AArch64TargetLoweringInfo(LowerTypes &LT, AArch64ABIKind Kind)
      : TargetLoweringInfo(std::make_unique<AArch64ABIInfo>(LT, Kind)) {
    assert(!MissingFeature::swift());
  }

  unsigned getTargetAddrSpaceFromCIRAddrSpace(
      mlir::cir::AddressSpaceAttr addressSpaceAttr) const override {
    using Kind = mlir::cir::AddressSpaceAttr::Kind;
    switch (addressSpaceAttr.getValue()) {
    case Kind::offload_private:
    case Kind::offload_local:
    case Kind::offload_global:
    case Kind::offload_constant:
    case Kind::offload_generic:
      return 0;
    default:
      llvm_unreachable("Unknown CIR address space for this target");
    }
  }
};

} // namespace

ABIArgInfo AArch64ABIInfo::classifyReturnType(Type RetTy,
                                              bool IsVariadic) const {
  if (isa<VoidType>(RetTy))
    return ABIArgInfo::getIgnore();

  if (const auto _ = dyn_cast<VectorType>(RetTy)) {
    llvm_unreachable("NYI");
  }

  // Large vector types should be returned via memory.
  if (isa<VectorType>(RetTy) && getContext().getTypeSize(RetTy) > 128)
    llvm_unreachable("NYI");

  if (!isAggregateTypeForABI(RetTy)) {
    // NOTE(cir): Skip enum handling.

    if (MissingFeature::fixedSizeIntType())
      llvm_unreachable("NYI");

    return (isPromotableIntegerTypeForABI(RetTy) && isDarwinPCS()
                ? ABIArgInfo::getExtend(RetTy)
                : ABIArgInfo::getDirect());
  }

  llvm_unreachable("NYI");
}

ABIArgInfo
AArch64ABIInfo::classifyArgumentType(Type Ty, bool IsVariadic,
                                     unsigned CallingConvention) const {
  Ty = useFirstFieldIfTransparentUnion(Ty);

  // TODO(cir): check for illegal vector types.
  if (MissingFeature::vectorType())
    llvm_unreachable("NYI");

  if (!isAggregateTypeForABI(Ty)) {
    // NOTE(cir): Enum is IntType in CIR. Skip enum handling here.

    if (MissingFeature::fixedSizeIntType())
      llvm_unreachable("NYI");

    return (isPromotableIntegerTypeForABI(Ty) && isDarwinPCS()
                ? ABIArgInfo::getExtend(Ty)
                : ABIArgInfo::getDirect());
  }

  llvm_unreachable("NYI");
}

std::unique_ptr<TargetLoweringInfo>
createAArch64TargetLoweringInfo(LowerModule &CGM, AArch64ABIKind Kind) {
  return std::make_unique<AArch64TargetLoweringInfo>(CGM.getTypes(), Kind);
}

} // namespace cir
} // namespace mlir
