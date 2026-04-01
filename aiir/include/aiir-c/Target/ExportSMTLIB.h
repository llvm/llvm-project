//===- aiir-c/Target/ExportSMTLIB.h - C API for emitting SMTLIB ---*- C -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header declares the C interface for emitting SMTLIB from an AIIR module.
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_C_EXPORTSMTLIB_H
#define AIIR_C_EXPORTSMTLIB_H

#include "aiir-c/IR.h"

#ifdef __cplusplus
extern "C" {
#endif

/// Emits SMTLIB for the specified module using the provided callback and user
/// data
AIIR_CAPI_EXPORTED AiirLogicalResult aiirTranslateModuleToSMTLIB(
    AiirModule, AiirStringCallback, void *userData, bool inlineSingleUseValues,
    bool indentLetBody, bool emitReset);

AIIR_CAPI_EXPORTED AiirLogicalResult aiirTranslateOperationToSMTLIB(
    AiirOperation, AiirStringCallback, void *userData,
    bool inlineSingleUseValues, bool indentLetBody, bool emitReset);

#ifdef __cplusplus
}
#endif

#endif // AIIR_C_EXPORTSMTLIB_H
