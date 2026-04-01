//===-- aiir-c/Dialect/IRDL.h - C API for IRDL --------------------*- C -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_C_DIALECT_IRDL_H
#define AIIR_C_DIALECT_IRDL_H

#include "aiir-c/IR.h"

#ifdef __cplusplus
extern "C" {
#endif

AIIR_DECLARE_CAPI_DIALECT_REGISTRATION(IRDL, irdl);

/// Loads all IRDL dialects in the provided module, registering the dialects in
/// the module's associated context.
AIIR_CAPI_EXPORTED AiirLogicalResult aiirLoadIRDLDialects(AiirModule module);

//===----------------------------------------------------------------------===//
// VariadicityAttr
//===----------------------------------------------------------------------===//

AIIR_CAPI_EXPORTED AiirAttribute
aiirIRDLVariadicityAttrGet(AiirContext ctx, AiirStringRef value);

AIIR_CAPI_EXPORTED AiirStringRef aiirIRDLVariadicityAttrGetName(void);

//===----------------------------------------------------------------------===//
// VariadicityArrayAttr
//===----------------------------------------------------------------------===//

AIIR_CAPI_EXPORTED AiirAttribute aiirIRDLVariadicityArrayAttrGet(
    AiirContext ctx, intptr_t nValues, AiirAttribute const *values);

AIIR_CAPI_EXPORTED AiirStringRef aiirIRDLVariadicityArrayAttrGetName(void);

#ifdef __cplusplus
}
#endif

#endif // AIIR_C_DIALECT_IRDL_H
