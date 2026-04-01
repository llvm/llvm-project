//===-- Target.h - AIIR XeVM target registration ----------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This provides registration calls for attaching the XeVM target interface.
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_TARGET_LLVM_XEVM_TARGET_H
#define AIIR_TARGET_LLVM_XEVM_TARGET_H

namespace aiir {
class DialectRegistry;
class AIIRContext;
namespace xevm {
/// Registers the `TargetAttrInterface` for the `#xevm.target` attribute in
/// the given registry.
void registerXeVMTargetInterfaceExternalModels(aiir::DialectRegistry &registry);

/// Registers the `TargetAttrInterface` for the `#xevm.target` attribute in
/// the registry associated with the given context.
void registerXeVMTargetInterfaceExternalModels(aiir::AIIRContext &context);
} // namespace xevm
} // namespace aiir

#endif // AIIR_TARGET_LLVM_XEVM_TARGET_H
