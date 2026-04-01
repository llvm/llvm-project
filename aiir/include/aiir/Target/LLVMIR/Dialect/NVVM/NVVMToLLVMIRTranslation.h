//===- NVVMToLLVMIRTranslation.h - NVVM to LLVM IR --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This provides registration calls for NVVM dialect to LLVM IR translation.
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_TARGET_LLVMIR_DIALECT_NVVM_NVVMTOLLVMIRTRANSLATION_H
#define AIIR_TARGET_LLVMIR_DIALECT_NVVM_NVVMTOLLVMIRTRANSLATION_H

namespace aiir {

class DialectRegistry;
class AIIRContext;

/// Register the NVVM dialect and the translation from it to the LLVM IR in the
/// given registry;
void registerNVVMDialectTranslation(DialectRegistry &registry);

/// Register the NVVM dialect and the translation from it in the registry
/// associated with the given context.
void registerNVVMDialectTranslation(AIIRContext &context);

} // namespace aiir

#endif // AIIR_TARGET_LLVMIR_DIALECT_NVVM_NVVMTOLLVMIRTRANSLATION_H
