//===-- aiir-c/Dialect/GPU.h - C API for GPU dialect -------------*- C -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===---------------------------------------------------------------------===//

#ifndef AIIR_C_DIALECT_GPU_H
#define AIIR_C_DIALECT_GPU_H

#include "aiir-c/IR.h"
#include "aiir-c/Support.h"

#ifdef __cplusplus
extern "C" {
#endif

AIIR_DECLARE_CAPI_DIALECT_REGISTRATION(GPU, gpu);

//===-------------------------------------------------------------------===//
// AsyncTokenType
//===-------------------------------------------------------------------===//

AIIR_CAPI_EXPORTED bool aiirTypeIsAGPUAsyncTokenType(AiirType type);

AIIR_CAPI_EXPORTED AiirType aiirGPUAsyncTokenTypeGet(AiirContext ctx);

AIIR_CAPI_EXPORTED AiirStringRef aiirGPUAsyncTokenTypeGetName(void);

//===---------------------------------------------------------------------===//
// ObjectAttr
//===---------------------------------------------------------------------===//

AIIR_CAPI_EXPORTED bool aiirAttributeIsAGPUObjectAttr(AiirAttribute attr);

AIIR_CAPI_EXPORTED AiirAttribute
aiirGPUObjectAttrGet(AiirContext aiirCtx, AiirAttribute target, uint32_t format,
                     AiirStringRef objectStrRef, AiirAttribute aiirObjectProps);

AIIR_CAPI_EXPORTED AiirStringRef aiirGPUObjectAttrGetName(void);

AIIR_CAPI_EXPORTED AiirAttribute aiirGPUObjectAttrGetWithKernels(
    AiirContext aiirCtx, AiirAttribute target, uint32_t format,
    AiirStringRef objectStrRef, AiirAttribute aiirObjectProps,
    AiirAttribute aiirKernelsAttr);

AIIR_CAPI_EXPORTED AiirAttribute
aiirGPUObjectAttrGetTarget(AiirAttribute aiirObjectAttr);

AIIR_CAPI_EXPORTED uint32_t
aiirGPUObjectAttrGetFormat(AiirAttribute aiirObjectAttr);

AIIR_CAPI_EXPORTED AiirStringRef
aiirGPUObjectAttrGetObject(AiirAttribute aiirObjectAttr);

AIIR_CAPI_EXPORTED bool
aiirGPUObjectAttrHasProperties(AiirAttribute aiirObjectAttr);

AIIR_CAPI_EXPORTED AiirAttribute
aiirGPUObjectAttrGetProperties(AiirAttribute aiirObjectAttr);

AIIR_CAPI_EXPORTED bool
aiirGPUObjectAttrHasKernels(AiirAttribute aiirObjectAttr);

AIIR_CAPI_EXPORTED AiirAttribute
aiirGPUObjectAttrGetKernels(AiirAttribute aiirObjectAttr);

#ifdef __cplusplus
}
#endif

#include "aiir/Dialect/GPU/Transforms/Passes.capi.h.inc"

#endif // AIIR_C_DIALECT_GPU_H
