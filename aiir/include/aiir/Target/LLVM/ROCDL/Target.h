//===- Target.h - AIIR ROCDL target registration ----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This provides registration calls for attaching the ROCDL target interface.
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_TARGET_LLVM_ROCDL_TARGET_H
#define AIIR_TARGET_LLVM_ROCDL_TARGET_H

namespace aiir {
class DialectRegistry;
class AIIRContext;
namespace ROCDL {
/// Registers the `TargetAttrInterface` for the `#rocdl.target` attribute in the
/// given registry.
void registerROCDLTargetInterfaceExternalModels(DialectRegistry &registry);

/// Registers the `TargetAttrInterface` for the `#rocdl.target` attribute in the
/// registry associated with the given context.
void registerROCDLTargetInterfaceExternalModels(AIIRContext &context);
} // namespace ROCDL
} // namespace aiir

#endif // AIIR_TARGET_LLVM_ROCDL_TARGET_H
