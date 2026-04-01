//===-- aiir-c/Dialect/AMDGPU.h - C API for AMDGPU dialect --*- C -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_C_DIALECT_AMDGPU_H
#define AIIR_C_DIALECT_AMDGPU_H

#include "aiir-c/IR.h"

#ifdef __cplusplus
extern "C" {
#endif

AIIR_DECLARE_CAPI_DIALECT_REGISTRATION(AMDGPU, amdgpu);

//===---------------------------------------------------------------------===//
// TDMBaseType
//===---------------------------------------------------------------------===//

AIIR_CAPI_EXPORTED bool aiirTypeIsAAMDGPUTDMBaseType(AiirType type);

AIIR_CAPI_EXPORTED AiirTypeID aiirAMDGPUTDMBaseTypeGetTypeID();

AIIR_CAPI_EXPORTED AiirType aiirAMDGPUTDMBaseTypeGet(AiirContext ctx,
                                                     AiirType elementType);

AIIR_CAPI_EXPORTED AiirStringRef aiirAMDGPUTDMBaseTypeGetName(void);

//===---------------------------------------------------------------------===//
// TDMDescriptorType
//===---------------------------------------------------------------------===//

AIIR_CAPI_EXPORTED bool aiirTypeIsAAMDGPUTDMDescriptorType(AiirType type);

AIIR_CAPI_EXPORTED AiirTypeID aiirAMDGPUTDMDescriptorTypeGetTypeID();

AIIR_CAPI_EXPORTED AiirType aiirAMDGPUTDMDescriptorTypeGet(AiirContext ctx);

AIIR_CAPI_EXPORTED AiirStringRef aiirAMDGPUTDMDescriptorTypeGetName(void);

//===---------------------------------------------------------------------===//
// TDMGatherBaseType
//===---------------------------------------------------------------------===//

AIIR_CAPI_EXPORTED bool aiirTypeIsAAMDGPUTDMGatherBaseType(AiirType type);

AIIR_CAPI_EXPORTED AiirTypeID aiirAMDGPUTDMGatherBaseTypeGetTypeID();

AIIR_CAPI_EXPORTED AiirType aiirAMDGPUTDMGatherBaseTypeGet(AiirContext ctx,
                                                           AiirType elementType,
                                                           AiirType indexType);

AIIR_CAPI_EXPORTED AiirStringRef aiirAMDGPUTDMGatherBaseTypeGetName(void);

#ifdef __cplusplus
}
#endif

#include "aiir/Dialect/AMDGPU/Transforms/Passes.capi.h.inc"

#endif // AIIR_C_DIALECT_AMDGPU_H
