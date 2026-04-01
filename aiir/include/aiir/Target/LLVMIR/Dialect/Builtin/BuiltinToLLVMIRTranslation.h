//==- BuiltinToLLVMIRTranslation.h - Builtin Dialect to LLVM IR -*- C++ -*-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This provides registration calls for builtin dialect to LLVM IR translation.
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_TARGET_LLVMIR_DIALECT_BUILTIN_BUILTINTOLLVMIRTRANSLATION_H
#define AIIR_TARGET_LLVMIR_DIALECT_BUILTIN_BUILTINTOLLVMIRTRANSLATION_H

namespace aiir {

class DialectRegistry;
class AIIRContext;

/// Register the translation from the builtin dialect to the LLVM IR in the
/// given registry.
void registerBuiltinDialectTranslation(DialectRegistry &registry);

/// Register the translation from the builtin dialect in the registry associated
/// with the given context.
void registerBuiltinDialectTranslation(AIIRContext &context);

} // namespace aiir

#endif // AIIR_TARGET_LLVMIR_DIALECT_BUILTIN_BUILTINTOLLVMIRTRANSLATION_H
