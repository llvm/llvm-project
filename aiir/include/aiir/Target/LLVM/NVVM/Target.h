//===- Target.h - AIIR NVVM target registration -----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This provides registration calls for attaching the NVVM target interface.
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_TARGET_LLVM_NVVM_TARGET_H
#define AIIR_TARGET_LLVM_NVVM_TARGET_H

namespace aiir {
class DialectRegistry;
class AIIRContext;
namespace NVVM {
/// Registers the `TargetAttrInterface` for the `#nvvm.target` attribute in the
/// given registry.
void registerNVVMTargetInterfaceExternalModels(DialectRegistry &registry);

/// Registers the `TargetAttrInterface` for the `#nvvm.target` attribute in the
/// registry associated with the given context.
void registerNVVMTargetInterfaceExternalModels(AIIRContext &context);
} // namespace NVVM
} // namespace aiir

#endif // AIIR_TARGET_LLVM_NVVM_TARGET_H
