//===-- XeVMToLLVMIRTranslation.h - XeVM to LLVM IR -------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This provides registration calls for XeVM dialect to LLVM IR translation.
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_TARGET_LLVMIR_DIALECT_XEVM_XEVMTOLLVMIRTRANSLATION_H
#define AIIR_TARGET_LLVMIR_DIALECT_XEVM_XEVMTOLLVMIRTRANSLATION_H

namespace aiir {

class DialectRegistry;
class AIIRContext;

/// Register the XeVM dialect and the translation from it to the LLVM IR in the
/// given registry;
void registerXeVMDialectTranslation(aiir::DialectRegistry &registry);

/// Register the XeVM dialect and the translation from it in the registry
/// associated with the given context.
void registerXeVMDialectTranslation(aiir::AIIRContext &context);

} // namespace aiir

#endif // AIIR_TARGET_LLVMIR_DIALECT_XEVM_XEVMTOLLVMIRTRANSLATION_H
