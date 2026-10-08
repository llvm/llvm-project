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

namespace clang::CodeGenUtils {

//===----------------------------------------------------------------------===//
// Arm and AArch64
//===----------------------------------------------------------------------===//

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

} // namespace clang::CodeGenUtils
