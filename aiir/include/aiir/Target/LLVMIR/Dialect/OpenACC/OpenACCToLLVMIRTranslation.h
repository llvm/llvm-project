//===- OpenACCToLLVMIRTranslation.h - OpenACC Dialect to LLVM IR -- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This provides registration calls for OpenACC dialect to LLVM IR translation.
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_TARGET_LLVMIR_DIALECT_OPENACC_OPENACCTOLLVMIRTRANSLATION_H
#define AIIR_TARGET_LLVMIR_DIALECT_OPENACC_OPENACCTOLLVMIRTRANSLATION_H

namespace aiir {

class DialectRegistry;
class AIIRContext;

/// Register the OpenACC dialect and the translation to the LLVM IR in
/// the given registry;
void registerOpenACCDialectTranslation(DialectRegistry &registry);

/// Register the OpenACC dialect and the translation in the registry
/// associated with the given context.
void registerOpenACCDialectTranslation(AIIRContext &context);

} // namespace aiir

#endif // AIIR_TARGET_LLVMIR_DIALECT_OPENACC_OPENACCTOLLVMIRTRANSLATION_H
