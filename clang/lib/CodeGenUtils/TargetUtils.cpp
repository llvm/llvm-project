//===--- TargetUtils.cpp - Shared target-specific AST queries ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/CodeGenUtils/TargetUtils.h"
#include "clang/AST/Attr.h"
#include "clang/AST/Decl.h"
#include "clang/Basic/TargetBuiltins.h"
#include "clang/Basic/TargetInfo.h"

namespace clang::CodeGenUtils {

static bool isStreamingCompatible(const FunctionDecl *F) {
  if (const auto *T = F->getType()->getAs<FunctionProtoType>())
    return T->getAArch64SMEAttributes() &
           FunctionType::SME_PStateSMCompatibleMask;
  return false;
}

ArmSMEInlinability getArmSMEInlinability(const FunctionDecl *Caller,
                                         const FunctionDecl *Callee) {
  bool CallerIsStreaming =
      IsArmStreamingFunction(Caller, /*IncludeLocallyStreaming=*/true);
  bool CalleeIsStreaming =
      IsArmStreamingFunction(Callee, /*IncludeLocallyStreaming=*/true);
  bool CallerIsStreamingCompatible = isStreamingCompatible(Caller);
  bool CalleeIsStreamingCompatible = isStreamingCompatible(Callee);

  ArmSMEInlinability Inlinability = ArmSMEInlinability::Ok;

  if (!CalleeIsStreamingCompatible &&
      (CallerIsStreaming != CalleeIsStreaming || CallerIsStreamingCompatible)) {
    if (CalleeIsStreaming)
      Inlinability |= ArmSMEInlinability::ErrorIncompatibleStreamingModes;
    else
      Inlinability |= ArmSMEInlinability::WarnIncompatibleStreamingModes;
  }
  if (auto *NewAttr = Callee->getAttr<ArmNewAttr>()) {
    if (NewAttr->isNewZA())
      Inlinability |= ArmSMEInlinability::ErrorCalleeRequiresNewZA;
    if (NewAttr->isNewZT0())
      Inlinability |= ArmSMEInlinability::ErrorCalleeRequiresNewZT0;
  }

  return Inlinability;
}

bool hasExtraNeonArgument(unsigned BuiltinID) {
  // Required by the headers included below, but not in this particular
  // function.
  [[maybe_unused]] int PtrArgNum = -1;
  [[maybe_unused]] bool HasConstPtr = false;

  // The mask encodes the type. We don't care about the actual value. Instead,
  // we just check whether its been set.
  uint64_t mask = 0;
  switch (BuiltinID) {
#define GET_NEON_OVERLOAD_CHECK
#include "clang/Basic/arm_fp16.inc"
#include "clang/Basic/arm_neon.inc"
#undef GET_NEON_OVERLOAD_CHECK
  // Non-neon builtins for controling VFP that take extra argument for
  // discriminating the type.
  case ARM::BI__builtin_arm_vcvtr_f:
  case ARM::BI__builtin_arm_vcvtr_d:
    mask = 1;
  }

  return mask != 0;
}

bool isAAPCS(const TargetInfo &TargetInfo) {
  return TargetInfo.getABI().starts_with("aapcs");
}

bool requiresAMDGPUProtectedVisibility(const Decl *D,
                                       bool HasHiddenVisibility) {
  if (!HasHiddenVisibility)
    return false;

  return !D->hasAttr<OMPDeclareTargetDeclAttr>() &&
         (D->hasAttr<DeviceKernelAttr>() ||
          (isa<FunctionDecl>(D) && D->hasAttr<CUDAGlobalAttr>()) ||
          (isa<VarDecl>(D) &&
           (D->hasAttr<CUDADeviceAttr>() || D->hasAttr<CUDAConstantAttr>() ||
            cast<VarDecl>(D)->getType()->isCUDADeviceBuiltinSurfaceType() ||
            cast<VarDecl>(D)->getType()->isCUDADeviceBuiltinTextureType())));
}

} // namespace clang::CodeGenUtils
