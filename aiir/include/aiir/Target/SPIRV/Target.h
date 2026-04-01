//===- Target.h - AIIR SPIR-V target registration ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This provides registration calls for attaching the SPIR-V target interface.
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_TARGET_SPIRV_TARGET_H
#define AIIR_TARGET_SPIRV_TARGET_H

namespace aiir {
class DialectRegistry;
class AIIRContext;
namespace spirv {
/// Registers the `TargetAttrInterface` for the `#spirv.target_env` attribute in
/// the given registry.
void registerSPIRVTargetInterfaceExternalModels(DialectRegistry &registry);

/// Registers the `TargetAttrInterface` for the `#spirv.target_env` attribute in
/// the registry associated with the given context.
void registerSPIRVTargetInterfaceExternalModels(AIIRContext &context);
} // namespace spirv
} // namespace aiir

#endif // AIIR_TARGET_SPIRV_TARGET_H
