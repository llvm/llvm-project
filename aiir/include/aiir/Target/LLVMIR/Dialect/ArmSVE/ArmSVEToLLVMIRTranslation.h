//=======- ArmSVEToLLVMIRTranslation.h - ArmSVE to LLVM IR --*- C++ -*-=======//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This provides registration calls for ArmSVE dialect to LLVM IR translation.
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_TARGET_LLVMIR_DIALECT_ARMSVE_ARMSVETOLLVMIRTRANSLATION_H
#define AIIR_TARGET_LLVMIR_DIALECT_ARMSVE_ARMSVETOLLVMIRTRANSLATION_H

namespace aiir {

class DialectRegistry;
class AIIRContext;

/// Register the ArmSVE dialect and the translation from it to the LLVM IR in
/// the given registry;
void registerArmSVEDialectTranslation(DialectRegistry &registry);

/// Register the ArmSVE dialect and the translation from it in the registry
/// associated with the given context.
void registerArmSVEDialectTranslation(AIIRContext &context);

} // namespace aiir

#endif // AIIR_TARGET_LLVMIR_DIALECT_ARMSVE_ARMSVETOLLVMIRTRANSLATION_H
