//===---- AArch64.cpp - AArch64-specific CIR CodeGen ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This provides AArch64-specific CIR CodeGen logic.
//
//===----------------------------------------------------------------------===//

#include "ABIInfo.h"
#include "CIRGenFunction.h"
#include "CIRGenModule.h"
#include "TargetInfo.h"
#include "clang/AST/Decl.h"
#include "clang/CIR/MissingFeatures.h"

using namespace clang;
using namespace clang::CIRGen;

namespace {

class AArch64ABIInfo : public ABIInfo {
public:
  AArch64ABIInfo(CIRGenTypes &cgt) : ABIInfo(cgt) {}
};

class AArch64TargetCIRGenInfo : public TargetCIRGenInfo {
public:
  AArch64TargetCIRGenInfo(CIRGenTypes &cgt)
      : TargetCIRGenInfo(std::make_unique<AArch64ABIInfo>(cgt)) {}

  void setTargetAttributes(const Decl *d, mlir::Operation *gv,
                           CIRGenModule &cgm) const override {
    auto fn = mlir::dyn_cast<cir::FuncOp>(gv);
    if (!fn)
      return;
    assert(!cir::MissingFeatures::branchProtection());
    assert(!cir::MissingFeatures::pointerAuthentication());
  }

  bool isScalarizableAsmOperand(CIRGenFunction &cgf,
                                mlir::Type ty) const override {
    if (cgf.getTarget().hasFeature("ls64")) {
      cgf.cgm.errorNYI("AArch64 LS64 scalarizable asm operand");
      return true;
    }
    return TargetCIRGenInfo::isScalarizableAsmOperand(cgf, ty);
  }

  bool wouldInliningViolateFunctionCallABI(
      const FunctionDecl *caller, const FunctionDecl *callee) const override;
};

} // namespace

// TODO(cir): Find a way to share this with classic codegen.
enum class ArmSMEInlinability : uint8_t {
  Ok = 0,
  ErrorCalleeRequiresNewZA = 1 << 0,
  ErrorCalleeRequiresNewZT0 = 1 << 1,
  WarnIncompatibleStreamingModes = 1 << 2,
  ErrorIncompatibleStreamingModes = 1 << 3,

  IncompatibleStreamingModes =
      WarnIncompatibleStreamingModes | ErrorIncompatibleStreamingModes,

  LLVM_MARK_AS_BITMASK_ENUM(/*LargestValue=*/ErrorIncompatibleStreamingModes),
};

static bool isStreamingCompatible(const FunctionDecl *fd) {
  if (const auto *fpt = fd->getType()->getAs<FunctionProtoType>())
    return fpt->getAArch64SMEAttributes() &
           clang::FunctionType::SME_PStateSMCompatibleMask;
  return false;
}

/// Determines if there are any Arm SME ABI issues with inlining \p Callee into
/// \p Caller. Returns the issue (if any) in the ArmSMEInlinability bit enum.
static ArmSMEInlinability getArmSMEInlinability(const FunctionDecl *caller,
                                                const FunctionDecl *callee) {
  bool callerIsStreaming =
      clang::IsArmStreamingFunction(caller, /*IncludeLocallyStreaming=*/true);
  bool calleeIsStreaming =
      clang::IsArmStreamingFunction(callee, /*IncludeLocallyStreaming=*/true);
  bool callerIsStreamingCompatible = isStreamingCompatible(caller);
  bool calleeIsStreamingCompatible = isStreamingCompatible(callee);

  ArmSMEInlinability inlinability = ArmSMEInlinability::Ok;

  if (!calleeIsStreamingCompatible &&
      (callerIsStreaming != calleeIsStreaming || callerIsStreamingCompatible)) {
    if (calleeIsStreaming)
      inlinability |= ArmSMEInlinability::ErrorIncompatibleStreamingModes;
    else
      inlinability |= ArmSMEInlinability::WarnIncompatibleStreamingModes;
  }
  if (auto *newAttr = callee->getAttr<ArmNewAttr>()) {
    if (newAttr->isNewZA())
      inlinability |= ArmSMEInlinability::ErrorCalleeRequiresNewZA;
    if (newAttr->isNewZT0())
      inlinability |= ArmSMEInlinability::ErrorCalleeRequiresNewZT0;
  }

  return inlinability;
}

bool AArch64TargetCIRGenInfo::wouldInliningViolateFunctionCallABI(
    const FunctionDecl *caller, const FunctionDecl *callee) const {
  return caller && callee &&
         getArmSMEInlinability(caller, callee) != ArmSMEInlinability::Ok;
}

std::unique_ptr<TargetCIRGenInfo>
clang::CIRGen::createAArch64TargetCIRGenInfo(CIRGenTypes &cgt) {
  return std::make_unique<AArch64TargetCIRGenInfo>(cgt);
}
