//===---- TargetLoweringInfo.cpp - Encapsulate target details ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file partially mimics the TargetCodeGenInfo class from the file
// clang/lib/CodeGen/TargetInfo.cpp.
//
//===----------------------------------------------------------------------===//

#include "TargetLoweringInfo.h"
#include "clang/CIR/MissingFeatures.h"
#include "llvm/TargetParser/Triple.h"

namespace cir {

TargetLoweringInfo::~TargetLoweringInfo() = default;

cir::SyncScopeKind
TargetLoweringInfo::convertSyncScope(cir::SyncScopeKind syncScope) const {
  // By default, targets don't deal with sync scopes other than system scope.
  return cir::SyncScopeKind::System;
}

std::unique_ptr<TargetLoweringInfo>
createTargetLoweringInfo(const llvm::Triple &triple) {
  switch (triple.getArch()) {
  case llvm::Triple::amdgpu:
    return createAMDGPUTargetLoweringInfo();
  case llvm::Triple::nvptx:
  case llvm::Triple::nvptx64:
    return createNVPTXTargetLoweringInfo();
  case llvm::Triple::spir:
  case llvm::Triple::spir64:
    return createCommonSPIRTargetLoweringInfo();
  case llvm::Triple::spirv:
  case llvm::Triple::spirv32:
  case llvm::Triple::spirv64:
    return createSPIRVTargetLoweringInfo();
  default:
    assert(!cir::MissingFeatures::targetLoweringInfo());
    return std::make_unique<TargetLoweringInfo>();
  }
}

} // namespace cir
