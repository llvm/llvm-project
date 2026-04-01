//===- Dialects.h - CAPI for dialects -----------------------------*- C -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef STANDALONE_C_DIALECTS_H
#define STANDALONE_C_DIALECTS_H

#include "aiir-c/IR.h"

#ifdef __cplusplus
extern "C" {
#endif

AIIR_DECLARE_CAPI_DIALECT_REGISTRATION(Standalone, standalone);

AIIR_CAPI_EXPORTED AiirType aiirStandaloneCustomTypeGet(AiirContext ctx,
                                                        AiirStringRef value);

AIIR_CAPI_EXPORTED bool aiirStandaloneTypeIsACustomType(AiirType t);

AIIR_CAPI_EXPORTED AiirTypeID aiirStandaloneCustomTypeGetTypeID(void);

#ifdef __cplusplus
}
#endif

#endif // STANDALONE_C_DIALECTS_H
