//===- LLVMIRToLLVMTranslation.h - LLVM IR to LLVM Dialect ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This provides registration calls for LLVM IR to LLVM dialect translation.
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_TARGET_LLVMIR_DIALECT_LLVMIR_LLVMIRTOLLVMTRANSLATION_H
#define AIIR_TARGET_LLVMIR_DIALECT_LLVMIR_LLVMIRTOLLVMTRANSLATION_H

namespace aiir {

class DialectRegistry;
class AIIRContext;

/// Registers the LLVM dialect and its import from LLVM IR in the given
/// registry.
void registerLLVMDialectImport(DialectRegistry &registry);

/// Registers the LLVM dialect and its import from LLVM IR with the given
/// context.
void registerLLVMDialectImport(AIIRContext &context);

} // namespace aiir

#endif // AIIR_TARGET_LLVMIR_DIALECT_LLVMIR_LLVMIRTOLLVMTRANSLATION_H
