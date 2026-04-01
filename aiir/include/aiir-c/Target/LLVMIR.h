//===-- LLVMIR.h - C Interface for AIIR LLVMIR Target -------------*- C -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header declares the C interface to target LLVMIR with AIIR.
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_C_TARGET_LLVMIR_H
#define AIIR_C_TARGET_LLVMIR_H

#include "aiir-c/IR.h"
#include "aiir-c/Support.h"
#include "llvm-c/Core.h"
#include "llvm-c/Support.h"

#ifdef __cplusplus
extern "C" {
#endif

/// Translate operation that satisfies LLVM dialect module requirements into an
/// LLVM IR module living in the given context. This translates operations from
/// any dilalect that has a registered implementation of
/// LLVMTranslationDialectInterface.
///
/// \returns the generated LLVM IR Module from the translated AIIR module, it is
/// owned by the caller.
AIIR_CAPI_EXPORTED LLVMModuleRef
aiirTranslateModuleToLLVMIR(AiirOperation module, LLVMContextRef context);

AIIR_CAPI_EXPORTED char *
aiirTranslateModuleToLLVMIRToString(AiirOperation module);

struct AiirTypeFromLLVMIRTranslator {
  void *ptr;
};

typedef struct AiirTypeFromLLVMIRTranslator AiirTypeFromLLVMIRTranslator;

/// Create an LLVM::TypeFromLLVMIRTranslator and transfer ownership to the
/// caller.
AIIR_CAPI_EXPORTED AiirTypeFromLLVMIRTranslator
aiirTypeFromLLVMIRTranslatorCreate(AiirContext ctx);

/// Takes an LLVM::TypeFromLLVMIRTranslator owned by the caller and destroys it.
/// It is the responsibility of the user to only pass an
/// LLVM::TypeFromLLVMIRTranslator class.
AIIR_CAPI_EXPORTED void
aiirTypeFromLLVMIRTranslatorDestroy(AiirTypeFromLLVMIRTranslator translator);

/// Translates the given LLVM IR type to the AIIR LLVM dialect.
AIIR_CAPI_EXPORTED AiirType aiirTypeFromLLVMIRTranslatorTranslateType(
    AiirTypeFromLLVMIRTranslator translator, LLVMTypeRef llvmType);

struct AiirTypeToLLVMIRTranslator {
  void *ptr;
};

typedef struct AiirTypeToLLVMIRTranslator AiirTypeToLLVMIRTranslator;

/// Create an LLVM::TypeToLLVMIRTranslator and transfer ownership to the
/// caller.
AIIR_CAPI_EXPORTED AiirTypeToLLVMIRTranslator
aiirTypeToLLVMIRTranslatorCreate(LLVMContextRef ctx);

/// Takes an LLVM::TypeToLLVMIRTranslator owned by the caller and destroys it.
/// It is the responsibility of the user to only pass an
/// LLVM::TypeToLLVMIRTranslator class.
AIIR_CAPI_EXPORTED void
aiirTypeToLLVMIRTranslatorDestroy(AiirTypeToLLVMIRTranslator translator);

/// Translates the given AIIR LLVM dialect to the LLVM IR type.
AIIR_CAPI_EXPORTED LLVMTypeRef aiirTypeToLLVMIRTranslatorTranslateType(
    AiirTypeToLLVMIRTranslator translator, AiirType aiirType);

#ifdef __cplusplus
}
#endif

#endif // AIIR_C_TARGET_LLVMIR_H
